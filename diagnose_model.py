"""Diagnostic script to analyze the latest deployed model and training data."""
import joblib
import os
import glob
import numpy as np
import pandas as pd

# Load the latest pytorch_mlp model (most likely deployed)
models = sorted(glob.glob('persistent_data/models/pytorch_mlp*.joblib'))
if not models:
    models = sorted(glob.glob('persistent_data/models/*.joblib'))
latest = models[-1]
print(f"=== Analyzing model: {os.path.basename(latest)} ===\n")

data = joblib.load(latest)
print(f"Model type: {data.get('model_type', 'unknown')}")
print(f"Labels: {list(data['label_encoder'].classes_)}")
print(f"Num features: {len(data['feature_names'])}")
print(f"Feature names: {data['feature_names']}")
print()

# Performance metrics
if 'performance_metrics' in data:
    pm = data['performance_metrics']
    for k in ['train_accuracy', 'val_accuracy', 'test_accuracy']:
        print(f"  {k}: {pm.get(k, 'N/A')}")
    if 'confusion_matrix' in pm:
        cm = np.array(pm['confusion_matrix'])
        labels = list(data['label_encoder'].classes_)
        print(f"\n  Confusion Matrix (rows=true, cols=pred):")
        print(f"  Labels: {labels}")
        for i, row in enumerate(cm):
            print(f"  {labels[i]:>20s}: {row}")
    if 'classification_report' in pm:
        print(f"\n  Classification Report:\n{pm['classification_report']}")
print()

# Scaler info
scaler = data.get('scaler')
if scaler:
    print("=== Scaler (StandardScaler) ===")
    feature_names = data['feature_names']
    for i, fn in enumerate(feature_names):
        print(f"  {fn:>30s}: mean={scaler.mean_[i]:.4f}, std={scaler.scale_[i]:.4f}")
print()

# Check training data
print("=== Training Data Analysis ===")
train_files = sorted(glob.glob('persistent_data/training/*_train.csv'))
for tf in train_files:
    df = pd.read_csv(tf)
    print(f"\nFile: {os.path.basename(tf)}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)}")
    if 'label' in df.columns:
        print(f"  Label distribution:\n{df['label'].value_counts().to_string()}")
        # Check key features per class
        for label in sorted(df['label'].unique()):
            subset = df[df['label'] == label]
            if 'acc_mag_min' in df.columns:
                print(f"  {label}: acc_mag_min={subset['acc_mag_min'].mean():.3f}, "
                      f"acc_mag_max={subset['acc_mag_max'].mean():.3f}, "
                      f"acc_mag_mean={subset['acc_mag_mean'].mean():.3f}")
            if 'gyro_mag_mean' in df.columns:
                print(f"  {label}: gyro_mag_mean={subset['gyro_mag_mean'].mean():.3f}, "  
                      f"gyro_mag_max={subset['gyro_mag_max'].mean():.3f}")

# Check window files
print("\n=== Window Files ===")
window_files = sorted(glob.glob('persistent_data/windows/*.csv'))
print(f"Total window files: {len(window_files)}")
for wf in window_files[:30]:
    df = pd.read_csv(wf)
    basename = os.path.basename(wf)
    # Detect label from filename
    print(f"  {basename}: {len(df)} rows")
