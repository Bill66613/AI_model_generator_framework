from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time

import tensorflow as tf
import numpy as np

# Directory to store datasets
DATASET_DIR = "datasets"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

app = FastAPI(title="Local AI Model Trainer for Human Motion Tracking")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify the actual origin(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global dictionary to track training progress for each dataset
training_status = {}


# Ensure there's a directory for saving models
MODELS_DIR = "models"
if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def train_model_tf(dataset_filename: str):
    """
    A sample training function using TensorFlow/Keras.
    This function creates a simple CNN model, trains it on dummy data,
    and saves the trained model. Replace the dummy data generation with your
    dataset loading and preprocessing logic.
    """
    # Dummy data: replace these with real data loading logic
    # For example, load images or sensor data associated with `dataset_filename`
    x_train = np.random.rand(100, 64, 64, 3)  # 100 samples, 64x64 RGB images
    y_train = np.random.randint(0, 2, size=(100, 1))  # Binary classification

    # Define a simple CNN model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train the model and simulate progress updates per epoch
    epochs = 5
    for epoch in range(1, epochs + 1):
        history = model.fit(x_train, y_train, epochs=1, batch_size=10, verbose=0)
        # Log or update training progress (you might integrate this with your tracking mechanism)
        progress = int((epoch / epochs) * 100)
        print(f"Epoch {epoch}/{epochs} completed. Progress: {progress}%")
        # Optionally, update a global status dictionary (similar to the dummy progress)
        training_status[dataset_filename] = {"status": "training", "progress": progress}
    
    # Save the model
    model_save_path = os.path.join(MODELS_DIR, f"{dataset_filename}_model.h5")
    model.save(model_save_path)
    training_status[dataset_filename] = {"status": "complete", "progress": 100}
    print(f"Training complete. Model saved to {model_save_path}")


@app.get("/health")
async def health_check():
    """
    Health Check Endpoint to verify that the server is running.
    """
    return {"status": "healthy"}

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads a dataset file and saves it in the designated datasets folder.
    """
    try:
        contents = await file.read()
        file_path = os.path.join(DATASET_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)
        return {"filename": file.filename, "status": "uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def dummy_train_model(dataset_filename: str):
    """
    Dummy training function to simulate model training.
    This function simulates training over 5 epochs, updating progress each epoch.
    In a real application, replace this with your actual model training logic.
    """
    file_path = os.path.join(DATASET_DIR, dataset_filename)
    if not os.path.exists(file_path):
        training_status[dataset_filename] = {"error": "Dataset not found."}
        return

    # Initialize training progress
    training_status[dataset_filename] = {"status": "training", "progress": 0}
    print(f"Starting training on dataset: {dataset_filename}")

    # Simulate training process over 5 epochs
    for epoch in range(1, 6):
        print(f"Epoch {epoch}: Training in progress...")
        time.sleep(2)  # Simulate time delay per epoch
        progress = int((epoch / 5) * 100)
        training_status[dataset_filename] = {"status": "training", "progress": progress}
    
    # Mark training as complete
    training_status[dataset_filename] = {"status": "complete", "progress": 100}
    print("Training complete.")

@app.post("/train-model")
async def train_model(file_name: str, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger model training on a specified dataset.
    The training process is executed in the background.
    """
    file_path = os.path.join(DATASET_DIR, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Dataset not found")
    
    # Schedule the TensorFlow training process as a background task
    background_tasks.add_task(train_model_tf, file_name)
    return {"message": f"Training for {file_name} has started in the background."}

@app.get("/training-progress/{dataset_filename}")
async def get_training_progress(dataset_filename: str):
    """
    Endpoint to retrieve the training progress for a given dataset.
    """
    status = training_status.get(dataset_filename)
    if status is None:
        raise HTTPException(status_code=404, detail="No training found for this dataset.")
    return status

if __name__ == "__main__":
    # Run the app on localhost at port 8000 with auto-reload enabled for development
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
