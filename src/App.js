import { useState, useEffect, useRef } from "react";
import * as mobilenet from "@tensorflow-models/mobilenet";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import "@tensorflow/tfjs-backend-cpu";
import * as tf from "@tensorflow/tfjs";

function App() {
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [model, setModel] = useState({ mobilenet: null, cocoSsd: null });
  const [imageURL, setImageURL] = useState(null);
  const [results, setResults] = useState([]);
  const [history, setHistory] = useState([]);

  const imageRef = useRef();
  const fileInputRef = useRef();
  const textInputRef = useRef();
  const canvasRef = useRef();

  // Load both MobileNet and COCO-SSD models
  const loadModel = async () => {
    setIsModelLoading(true);
    try {
      await tf.ready();
      const mobilenetModel = await mobilenet.load();
      const cocoSsdModel = await cocoSsd.load();
      setModel({ mobilenet: mobilenetModel, cocoSsd: cocoSsdModel });
      setIsModelLoading(false);
    } catch (error) {
      console.log("Error loading models:", error);
      setIsModelLoading(false);
    }
  };

  // Upload image
  const uploadImage = (e) => {
    const { files } = e.target;
    if (files.length > 0) {
      const url = URL.createObjectURL(files[0]);
      setImageURL(url);
    } else {
      setImageURL(null);
    }
    setResults([]); // Reset results when a new image is uploaded
  };

  // Identify objects using both models
  const identify = async () => {
    if (!model.mobilenet || !model.cocoSsd) {
      console.log("Models are not loaded yet");
      return;
    }

    if (!imageRef.current) {
      console.log("Image not found");
      return;
    }

    try {
      // Classify the image using MobileNet
      const classificationResults = await model.mobilenet.classify(imageRef.current);
      setResults(classificationResults);

      // Detect objects using COCO-SSD
      const detectionResults = await model.cocoSsd.detect(imageRef.current);
      drawBoundingBoxes(detectionResults);

      // Update history
      setHistory((prevHistory) => [imageURL, ...prevHistory]);
    } catch (error) {
      console.error("Error during identification:", error);
    }
  };

  // Draw bounding boxes on the canvas
  const drawBoundingBoxes = (predictions) => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const image = imageRef.current;

    // Set canvas dimensions to match the image's natural dimensions
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;

    // Scale the canvas to the visible dimensions of the image
    const scaleX = image.clientWidth / image.naturalWidth;
    const scaleY = image.clientHeight / image.naturalHeight;

    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Draw bounding boxes
    predictions.forEach((prediction) => {
      const [x, y, width, height] = prediction.bbox;

      // Scale bounding box coordinates to match the displayed size
      const scaledX = x * scaleX;
      const scaledY = y * scaleY;
      const scaledWidth = width * scaleX;
      const scaledHeight = height * scaleY;

      // Draw the bounding box
      ctx.strokeStyle = "red";
      ctx.lineWidth = 2;
      ctx.strokeRect(scaledX, scaledY, scaledWidth, scaledHeight);

      // Draw labels
      ctx.fillStyle = "red";
      ctx.font = "16px Arial";
      ctx.fillText(
        `${prediction.class} (${(prediction.score * 100).toFixed(2)}%)`,
        scaledX,
        scaledY > 10 ? scaledY - 5 : 10
      );
    });
  };

  // Handle input change
  const handleOnChange = (e) => {
    setImageURL(e.target.value);
    setResults([]);
  };

  // Trigger file upload
  const triggerUpload = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  // Load the models when the component mounts
  useEffect(() => {
    loadModel();
  }, []);

  // Track image history
  useEffect(() => {
    if (imageURL) {
      setHistory((prevHistory) => [imageURL, ...prevHistory]);
    }
  }, [imageURL]);

  // Display loading message if models are still loading
  if (isModelLoading) {
    return <h2>Model Loading...</h2>;
  }

  return (
    <div className="App">
      <h1 className="header">Image Identification</h1>
      <div className="inputHolder">
        <input
          type="file"
          accept="image/*"
          capture="camera"
          className="uploadInput"
          onChange={uploadImage}
          ref={fileInputRef}
        />
        <button className="uploadImage" onClick={triggerUpload}>
          Upload Image
        </button>
        <span className="or">OR</span>
        <input
          type="text"
          placeholder="Paste image URL"
          ref={textInputRef}
          onChange={handleOnChange}
        />
      </div>
      <div className="mainWrapper">
        <div className="mainContent">
          <div className="imageHolder" style={{ position: "relative" }}>
            {imageURL && (
              <>
                <img
                  src={imageURL}
                  alt="Upload Preview"
                  crossOrigin="anonymous"
                  ref={imageRef}
                  style={{ maxWidth: "100%", height: "auto" }}
                />
                <canvas
                  ref={canvasRef}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    pointerEvents: "none",
                  }}
                ></canvas>
              </>
            )}
          </div>
          <button
            className="button"
            onClick={identify}
            disabled={!imageURL}
            style={{
              marginTop: "10px",
              padding: "10px 20px",
              fontSize: "16px",
              cursor: imageURL ? "pointer" : "not-allowed",
            }}
          >
            Identify Image
          </button>
          {results.length > 0 && (
            <div className="resultsHolder">
              <h2>Classification Results</h2>
              {results.map((result, index) => (
                <div className="result" key={result.className}>
                  <span className="name">{result.className}</span>
                  <span className="confidence">
                    Confidence level: {(result.probability * 100).toFixed(2)}%
                    {index === 0 && <span className="bestGuess">Best Guess</span>}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
      {history.length > 0 && (
        <div className="recentPredictions">
          <h2>Recent Images</h2>
          <div className="recentImages">
            {history.map((image, index) => (
              <div className="recentPrediction" key={`${image}${index}`}>
                <img
                  src={image}
                  alt="Recent Prediction"
                  onClick={() => setImageURL(image)}
                />
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
