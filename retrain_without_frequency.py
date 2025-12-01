"""Retrain model without frequency features for Arduino deployment"""
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.model_selection import train_test_split
from collections import Counter
import glob
import pandas as pd
from utils.model_training import EdgeMLModel, prepare_training_data
from datetime import datetime

print("="*70)
print("RETRAINING MODEL WITHOUT FREQUENCY FEATURES")
print("="*70)

# Get all dragged window files
window_files = glob.glob('persistent_data/dragged_window_*.csv')
print(f"\nFound {len(window_files)} window files")

# Extract labels from filenames
labels = []
for f in window_files:
    # Extract label from filename: dragged_window_X_LABEL.csv
    label = f.split('_')[-1].replace('.csv', '')
    labels.append(label)

# Count samples per class
label_counts = Counter(labels)
print(f"\nClass distribution:")
for label, count in sorted(label_counts.items()):
    print(f"  {label}: {count} windows")

# Prepare training data WITHOUT frequency features
print(f"\nPreparing training data (time-domain features only)...")
X, y = prepare_training_data(
    window_files,
    labels,
    sensor_cols=['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ'],
    sampling_rate=100,
    include_frequency=False  # Only time-domain features (90 total)
)

print(f"Feature matrix shape: {X.shape}")
# Should be 90 (15 per axis * 6 axes)
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")

# Train Neural Network model
print(f"\nTraining Neural Network model...")
model = EdgeMLModel(model_type='neural_network')

# Split data manually
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Train set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Train model
train_results = model.train(X_train, y_train)

# Train model
train_results = model.train(X_train, y_train)

# Evaluate on test set
X_test_scaled = model.scaler.transform(X_test)
y_test_encoded = model.label_encoder.transform(y_test)
y_pred = model.model.predict(X_test_scaled)

accuracy = accuracy_score(y_test_encoded, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test_encoded, y_pred, average='weighted')

print(f"\nTest Set Results:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test_encoded, y_pred,
      target_names=model.label_encoder.classes_))

# Save model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f'persistent_data/neural_network_time_only_{timestamp}.joblib'
model.save_model(model_filename)

print(f"\n" + "="*70)
print(f"✅ Model saved to: {model_filename}")
print(f"✅ Features: {X.shape[1]} (time-domain only, no frequency features)")
print(f"✅ This model can be accurately deployed to Arduino!")
print("="*70)
print(f"\nTo use this model:")
print(f"1. Update regenerate_deployment.py to use: {model_filename}")
print(f"2. Run regeneration")
print(f"3. Upload to Arduino")
