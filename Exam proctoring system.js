import React, { useState, useEffect, useRef } from 'react';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import '@tensorflow/tfjs'; // Required for TensorFlow.js
import * as posenet from '@tensorflow-models/posenet'; // Required for PoseNet

const App = () => {
  const [model, setModel] = useState(null);
  const [poseModel, setPoseModel] = useState(null);
  const [capturedImage, setCapturedImage] = useState(null); // State to store captured image
  const [alertMessage, setAlertMessage] = useState(null); // State to store alert messages
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  // Load the TensorFlow object detection model and PoseNet model
  useEffect(() => {
    const loadModels = async () => {
      const cocoModel = await cocoSsd.load();
      const poseNetModel = await posenet.load();
      setModel(cocoModel);
      setPoseModel(poseNetModel);
    };

    loadModels();
  }, []);

  // Access webcam and display video
  useEffect(() => {
    if (videoRef.current) {
      navigator.mediaDevices
        .getUserMedia({ video: true })
        .then((stream) => {
          videoRef.current.srcObject = stream;
        })
        .catch((err) => console.log('Error accessing webcam:', err));
    }
  }, []);

  // Object detection and capturing image
  useEffect(() => {
    if (model && poseModel && videoRef.current) {
      const intervalId = setInterval(async () => {
        const predictions = await model.detect(videoRef.current);
        const pose = await poseModel.estimateSinglePose(videoRef.current, {
          flipHorizontal: false,
        });

        const detectedItems = [];

        predictions.forEach((prediction) => {
          // Check for specific objects and assign alerts and bounding box colors
          if (prediction.class === 'cell phone') {
            detectedItems.push({ ...prediction, color: 'red' });
            setAlertMessage('Red Alert: Cell phone detected!');
            captureImage();
          } else if (prediction.class === 'book') {
            detectedItems.push({ ...prediction, color: 'red' });
            setAlertMessage('Red Alert: Book detected!');
            captureImage();
          } else if (prediction.class === 'person') {
            detectedItems.push({ ...prediction, color: 'green' });
            setAlertMessage('Green Alert: Person detected!');
          } else if (prediction.class === 'head') {
            detectedItems.push({ ...prediction, color: 'red' });
            setAlertMessage('Red Alert: Head movement detected!');
            captureImage();
          } else {
            detectedItems.push({ ...prediction, color: 'red' });
            setAlertMessage('Red Alert: Undefined object detected!');
          }
        });

        // Check if head turn is detected and trigger red alert
        detectHeadTurn(pose);

        drawDetectionResults(detectedItems);
      }, 1000);

      return () => clearInterval(intervalId); // Clean up on unmount
    }
  }, [model, poseModel]); // Trigger when the model is loaded

  // Detect head turn by analyzing the positions of the eyes or shoulders
  const detectHeadTurn = (pose) => {
    if (pose.keypoints) {
      const leftEye = pose.keypoints.find((point) => point.part === 'leftEye');
      const rightEye = pose.keypoints.find((point) => point.part === 'rightEye');

      if (leftEye && rightEye) {
        const distanceX = Math.abs(leftEye.position.x - rightEye.position.x);

        // If the distance between eyes changes significantly, we infer a head turn
        if (distanceX > 30) {
          setAlertMessage('Red Alert: Head turn detected!');
          captureImage(); // Capture image when head turn is detected
        }
      }
    }
  };

  // Capture image from canvas and update state to display it
  const captureImage = () => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const video = videoRef.current;

    // Set canvas size to video size
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    // Draw the current video frame onto the canvas
    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Capture the image as base64 and update the state
    const imageUrl = canvas.toDataURL('image/png');
    setCapturedImage(imageUrl);
    console.log('Captured image:', imageUrl); // Log the captured image

    // Send captured image to the backend server
    saveImageToServer(imageUrl);
  };

  // Send the captured image to the backend server
  const saveImageToServer = async (imageData) => {
    try {
      const response = await fetch('http://localhost:5000/save-image', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          image: imageData,
        }),
      });

      const data = await response.json();
      if (data.success) {
        console.log('Image saved successfully!');
      } else {
        console.log('Failed to save image');
      }
    } catch (error) {
      console.error('Error saving image:', error);
    }
  };

  // Drawing detection results on canvas
  const drawDetectionResults = (predictions) => {
    const canvas = canvasRef.current;
    const context = canvas.getContext('2d');
    const video = videoRef.current;

    // Clear canvas before drawing new predictions
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    context.clearRect(0, 0, canvas.width, canvas.height);
    predictions.forEach((prediction) => {
      if (prediction.bbox && prediction.bbox.length === 4) {
        const [x, y, width, height] = prediction.bbox;

        // Draw bounding box
        context.beginPath();
        context.rect(x, y, width, height);
        context.lineWidth = 4;
        context.strokeStyle = prediction.color;
        context.fillStyle = prediction.color;
        context.stroke();

        // Draw label text
        context.fillText(
          ${prediction.class} (${Math.round(prediction.score * 100)}%),
          x,
          y > 10 ? y - 5 : 10
        );
      }
    });
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center bg-gray-100 p-4">
      <h1 className="text-3xl font-bold text-center text-gray-900 mb-6">
        Exam Malpractice Monitoring
      </h1>

      <div className="relative">
        <video
          ref={videoRef}
          width="640"
          height="480"
          autoPlay
          muted
          className="border-4 border-gray-800"
        />
        <canvas
          ref={canvasRef}
          className="absolute top-0 left-0"
          style={{ pointerEvents: 'none' }}
        />
      </div>

      {/* Display Captured Image Below the Video */}
      <div className="mt-6">
        {capturedImage && (
          <div className="max-w-sm mx-auto">
            <h3 className="text-xl font-semibold text-gray-800">Captured Image:</h3>
            <img src={capturedImage} alt="Captured Object" className="mt-4 border-4 border-gray-800" />
          </div>
        )}
      </div>

      {/* Display Alert Message */}
      <div className="mt-4 text-xl font-semibold text-center text-red-500">
        {alertMessage && <p>{alertMessage}</p>}
      </div>
    </div>
  );
};

export default App; 
