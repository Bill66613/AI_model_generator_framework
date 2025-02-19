import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [file, setFile] = useState(null);
  const [uploadStatus, setUploadStatus] = useState("");
  const [fileName, setFileName] = useState("");
  const [trainingProgress, setTrainingProgress] = useState(null);
  const [isTraining, setIsTraining] = useState(false);

  // Handle file selection
  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  // Upload the selected dataset to the backend
  const uploadDataset = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch('http://127.0.0.1:8000/upload-dataset', {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setUploadStatus(`Uploaded: ${data.filename}`);
      setFileName(data.filename);
    } catch (error) {
      console.error("Upload error: ", error);
      setUploadStatus("Upload failed");
    }
  };

  // Trigger training on the uploaded dataset
  const startTraining = async () => {
    if (!fileName) return;
    try {
      const response = await fetch(`http://127.0.0.1:8000/train-model?file_name=${fileName}`, {
        method: 'POST'
      });
      const data = await response.json();
      console.log(data.message);
      setIsTraining(true);
    } catch (error) {
      console.error("Training error: ", error);
    }
  };

  // Poll the training-progress endpoint every 2 seconds while training is active
  useEffect(() => {
    let interval;
    if (isTraining) {
      interval = setInterval(async () => {
        try {
          const response = await fetch(`http://127.0.0.1:8000/training-progress/${fileName}`);
          if(response.ok) {
            const progressData = await response.json();
            setTrainingProgress(progressData);
            if (progressData.status === "complete" || progressData.error) {
              setIsTraining(false);
              clearInterval(interval);
            }
          }
        } catch (error) {
          console.error("Progress error: ", error);
          setIsTraining(false);
          clearInterval(interval);
        }
      }, 2000);
    }
    return () => clearInterval(interval);
  }, [isTraining, fileName]);

  return (
    <div className="App">
      <h1>Local AI Model Trainer for Human Motion Tracking</h1>
      <div>
        <input type="file" onChange={handleFileChange} />
        <button onClick={uploadDataset}>Upload Dataset</button>
      </div>
      {uploadStatus && <p>{uploadStatus}</p>}
      {fileName && !isTraining && (
        <div>
          <button onClick={startTraining}>Start Training</button>
        </div>
      )}
      {isTraining && trainingProgress && (
        <div>
          <h2>Training Progress</h2>
          <p>Status: {trainingProgress.status}</p>
          <p>Progress: {trainingProgress.progress}%</p>
        </div>
      )}
    </div>
  );
}

export default App;
