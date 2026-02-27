"""
Comprehensive diagnostic for device prediction issues.
Analyzes: padding impact, class separability, model confidence, and feature importance.
"""
import joblib
import os
import glob
import numpy as np
import pandas as pd
from utils.feature_extraction import create_feature_vector
from config.config import SENSOR_COLUMNS

# ───────────────────────────────────────────────
# 1. Load model and training data
# ───────────────────────────────────────────────
models = sorted(glob.glob('persistent_data/models/pytorch_mlp*.joblib'))
latest = models[-1]
print(f"=== Model: {os.path.basename(latest)} ===")
data = joblib.load(latest)
scaler = data['scaler']
label_encoder = data['label_encoder']
feature_names = data['feature_names']
labels = list(label_encoder.classes_)
print(f"Classes: {labels}")
print(f"Features: {len(feature_names)}")

# Load the actual model for predictions
model_obj = data['model']
has_pytorch = hasattr(model_obj, '_pytorch_trainer')
print(f"PyTorch trainer available: {has_pytorch}")
print()

# ───────────────────────────────────────────────
# 2. Analyze window files and padding impact
# ───────────────────────────────────────────────
print("=" * 60)
print("=== WINDOW PADDING ANALYSIS ===")
print("=" * 60)
window_files = sorted(glob.glob('persistent_data/windows/*.csv'))
print(f"Total windows: {len(window_files)}")

TARGET = 150
window_info = []
for wf in window_files:
    df = pd.read_csv(wf)
    basename = os.path.basename(wf)
    # Extract label from filename
    parts = basename.replace('.csv', '').split('_')
    # The label is typically everything after the number prefix
    window_info.append({
        'file': basename,
        'rows': len(df),
        'pad_pct': max(0, (TARGET - len(df)) / TARGET * 100),
    })

for wi in window_info:
    pad_str = f"  PADDED {wi['pad_pct']:.0f}%" if wi['pad_pct'] > 0 else "  OK"
    print(f"  {wi['file']:>50s}: {wi['rows']:>4d} rows{pad_str}")

print()

# ───────────────────────────────────────────────
# 3. Quantify padding impact on features
# Pick a real window and compare features: original vs padded
# ───────────────────────────────────────────────
print("=" * 60)
print("=== PADDING IMPACT ON FEATURES (example window) ===")
print("=" * 60)

# Find a window with significant padding
for wf in window_files:
    df = pd.read_csv(wf)
    if len(df) < TARGET * 0.8:  # At least 20% padding needed
        # Compute features WITHOUT padding (original size)
        sensor_cols = [c for c in SENSOR_COLUMNS if c in df.columns]
        features_orig = create_feature_vector(
            df, sensor_cols, 100,
            include_frequency=False,
            orientation_robust=True,
            include_per_axis=False
        )
        
        # Now pad with edge replication (as the pipeline does)
        padding_needed = TARGET - len(df)
        last_row = df.iloc[[-1]]
        padding_df = pd.concat([last_row] * padding_needed, ignore_index=True)
        df_padded = pd.concat([df, padding_df], ignore_index=True)
        
        features_padded = create_feature_vector(
            df_padded, sensor_cols, 100,
            include_frequency=False,
            orientation_robust=True,
            include_per_axis=False
        )
        
        print(f"Window: {os.path.basename(wf)} ({len(df)} rows → padded to {TARGET})")
        print(f"Padding: {padding_needed} rows ({padding_needed/TARGET*100:.0f}%)")
        print()
        print(f"{'Feature':>35s} | {'Original':>12s} | {'Padded':>12s} | {'Diff%':>8s}")
        print("-" * 75)
        
        for col in features_orig.columns:
            v_orig = features_orig[col].values[0]
            v_pad = features_padded[col].values[0]
            if abs(v_orig) > 1e-6:
                diff_pct = (v_pad - v_orig) / abs(v_orig) * 100
            else:
                diff_pct = 0 if abs(v_pad) < 1e-6 else float('inf')
            marker = " <<<" if abs(diff_pct) > 10 else ""
            print(f"{col:>35s} | {v_orig:>12.4f} | {v_pad:>12.4f} | {diff_pct:>7.1f}%{marker}")
        
        print("\nFeatures with >10% change marked with <<<")
        break

print()

# ───────────────────────────────────────────────
# 4. Analyze class separability in feature space
# ───────────────────────────────────────────────
print("=" * 60)
print("=== CLASS SEPARABILITY ANALYSIS ===")
print("=" * 60)

train_files = sorted(glob.glob('persistent_data/training/*_train.csv'))
if train_files:
    df_train = pd.read_csv(train_files[0])
    feat_cols = [c for c in df_train.columns if c != 'label']
    
    # Fisher's discriminant ratio for each feature pair
    print("\nPer-feature discrimination (higher = better separability):")
    print(f"{'Feature':>35s} | ", end="")
    for lbl in sorted(df_train['label'].unique()):
        print(f"{lbl[:8]:>10s}", end=" | ")
    print(" Fisher_ratio")
    print("-" * (35 + 3 + (10 + 3) * len(sorted(df_train['label'].unique())) + 15))
    
    for feat in feat_cols:
        means = {}
        stds = {}
        for lbl in sorted(df_train['label'].unique()):
            subset = df_train[df_train['label'] == lbl]
            means[lbl] = subset[feat].mean()
            stds[lbl] = subset[feat].std()
        
        # Fisher ratio: variance of class means / mean of class variances
        class_means = list(means.values())
        class_vars = [s**2 for s in stds.values()]
        between_var = np.var(class_means)
        within_var = np.mean(class_vars)
        fisher = between_var / (within_var + 1e-10)
        
        print(f"{feat:>35s} | ", end="")
        for lbl in sorted(df_train['label'].unique()):
            print(f"{means[lbl]:>10.3f}", end=" | ")
        print(f" {fisher:.4f}")

print()

# ───────────────────────────────────────────────
# 5. Model confidence analysis per class
# ───────────────────────────────────────────────
print("=" * 60)
print("=== MODEL PREDICTION CONFIDENCE ===")
print("=" * 60)

test_files = sorted(glob.glob('persistent_data/training/*_test.csv'))
if test_files and has_pytorch:
    df_test = pd.read_csv(test_files[0])
    feat_cols = [c for c in df_test.columns if c != 'label']
    X_test = df_test[feat_cols].values
    y_test = df_test['label'].values
    
    # Scale features
    X_scaled = scaler.transform(X_test)
    
    # Get predictions through PyTorch
    import torch
    trainer = model_obj._pytorch_trainer
    trainer.model.eval()
    X_tensor = torch.FloatTensor(X_scaled)
    with torch.no_grad():
        logits = trainer.model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()
    
    print("\nPer-sample prediction confidence:")
    print(f"{'True Label':>20s} | {'Predicted':>15s} | {'Confidence':>10s} | Probabilities per class")
    print(f"{'':>20s} | {'':>15s} | {'':>10s} | {' | '.join([f'{l[:8]:>8s}' for l in labels])}")
    print("-" * 120)
    
    for i in range(len(X_test)):
        pred_idx = np.argmax(probs[i])
        pred_label = labels[pred_idx]
        conf = probs[i][pred_idx]
        prob_str = " | ".join([f"{p:>8.4f}" for p in probs[i]])
        marker = " ✓" if pred_label == y_test[i] else " ✗"
        print(f"{y_test[i]:>20s} | {pred_label:>15s} | {conf:>10.4f} | {prob_str}{marker}")
    
    # Summary
    print("\nClass-level confidence summary:")
    for lbl in labels:
        mask = y_test == lbl
        if mask.sum() == 0:
            continue
        class_probs = probs[mask]
        pred_indices = np.argmax(class_probs, axis=1)
        pred_labels = [labels[j] for j in pred_indices]
        correct = sum(1 for pl in pred_labels if pl == lbl)
        avg_conf = np.mean([class_probs[i][pred_indices[i]] for i in range(len(class_probs))])
        print(f"  {lbl}: {correct}/{mask.sum()} correct, avg confidence: {avg_conf:.4f}")

print()

# ───────────────────────────────────────────────
# 6. Simulate device-like data features
# ───────────────────────────────────────────────
print("=" * 60)
print("=== SIMULATED DEVICE vs TRAINING FEATURE COMPARISON ===")
print("=" * 60)
print("""
Training features are computed from PADDED windows (edge replication).
Device features are computed from FULL 150-sample real sensor readings.
Key question: does padding shift features enough to cross decision boundaries?
""")

# Show the scaler parameters to understand normalization
print("Scaler sensitivity (features with small std are most sensitive to shifts):")
print(f"{'Feature':>35s} | {'Scaler Mean':>12s} | {'Scaler Std':>12s} | {'1-Std shift → scaled':>20s}")
print("-" * 90)
for i, fn in enumerate(feature_names):
    shifted = 1.0 / scaler.scale_[i]  # How much 1-raw-unit shift becomes in scaled space
    print(f"{fn:>35s} | {scaler.mean_[i]:>12.4f} | {scaler.scale_[i]:>12.4f} | {shifted:>20.4f}")

# ───────────────────────────────────────────────
# 7. Summary and recommendations
# ───────────────────────────────────────────────
print()
print("=" * 60)
print("=== DIAGNOSIS SUMMARY ===")
print("=" * 60)
print(f"""
PROBLEM: Model never predicts 'walking' (Class 2) or 'walking_upstairs' (Class 4) on device.

ROOT CAUSES IDENTIFIED:

1. TINY DATASET ({len(df_train) if train_files else '?'} training samples, {len(feature_names)} features)
   - Only ~21 samples per class for 33 features
   - Severely underdetermined → overfits to exact training distribution
   - Rule of thumb: need 5-10x features per class = 165-330 samples/class

2. SIMILAR WALKING CLASSES  
   - walking vs walking_downstairs have nearly identical feature distributions
   - Small dataset means decision boundaries are unstable
   - Model may have learned noise rather than real class differences

3. WINDOW PADDING DISTRIBUTION SHIFT
   - All training windows are shorter than 150 samples
   - Edge replication reduces: std, kurtosis, mean_crossing_rate, jerk features
   - Device collects full 150 real samples → different feature distribution
   - Even small shifts can cross decision boundaries with an overfit model

4. JERK FEATURE CONTAMINATION
   - Padded region is constant → jerk = 0 in padded section  
   - Reduces jerk_mean, jerk_std significantly
   - Device jerk features will be much higher → unexpected by model

RECOMMENDATIONS (in priority order):

1. COLLECT MORE & LONGER DATA
   - Record at least 3-5 seconds per activity at 100Hz (300-500 samples)
   - This eliminates padding AND gives enough data for reliable training
   - Aim for 50+ windows per class (use sliding window)
   
2. OR REDUCE WINDOW SIZE
   - Set window size to 70 samples (0.7s) to eliminate all padding
   - Tradeoff: shorter window = less context for classification
   
3. USE SLIDING WINDOW AUGMENTATION
   - From a 5-second recording, extract overlapping 1.5s windows
   - Step size 0.5s → ~7 windows per recording
   - Dramatically increases training data
   
4. TRY RANDOM FOREST
   - Handles small datasets better than neural networks
   - More robust to feature distribution shifts
   - Less prone to overfitting with limited data
""")
