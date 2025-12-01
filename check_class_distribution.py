"""Quick script to check class distribution in training data."""
import pandas as pd
import os
import glob
from collections import Counter

data_dir = 'persistent_data'

print("\n" + "="*60)
print("TRAINING DATA ANALYSIS - Dragged Windows")
print("="*60)

# Find all dragged window files
window_files = glob.glob(os.path.join(data_dir, 'dragged_window_*_*.csv'))

if not window_files:
    print("❌ No dragged window files found!")
    print("   Expected files: persistent_data/dragged_window_X_ACTIVITY.csv")
    exit(1)

print(f"Found {len(window_files)} dragged window files\n")

# Count samples per class
class_counts = Counter()

for window_file in window_files:
    # Extract activity from filename: dragged_window_0_running.csv -> running
    basename = os.path.basename(window_file)
    activity = basename.split('_', 3)[-1].replace('.csv', '')

    df = pd.read_csv(window_file)
    sample_count = len(df)
    class_counts[activity] += sample_count

    print(f"  {basename:45s}: {sample_count:5d} samples -> {activity}")

print("\n" + "="*60)
print("CLASS DISTRIBUTION IN TRAINING DATA:")
print("="*60)

total_samples = sum(class_counts.values())

for activity in sorted(class_counts.keys()):
    count = class_counts[activity]
    percentage = (count / total_samples * 100) if total_samples > 0 else 0
    print(f"{activity:25s}: {count:8d} samples ({percentage:5.1f}%)")

print("="*60)
print(f"{'TOTAL':25s}: {total_samples:8d} samples")
print("="*60)

# Calculate imbalance ratio
if class_counts:
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    imbalance_ratio = max_count / min_count if min_count > 0 else 0

    min_class = min(class_counts, key=class_counts.get)
    max_class = max(class_counts, key=class_counts.get)

    print(f"\nImbalance Analysis:")
    print(f"  Smallest class: {min_class} ({min_count} samples)")
    print(f"  Largest class:  {max_class} ({max_count} samples)")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}x")

    if imbalance_ratio > 2.0:
        print(f"\n⚠️  WARNING: Severe class imbalance detected!")
        print(f"  The model may be biased toward '{max_class}'")
        print(f"\n  SOLUTIONS:")
        print(f"  1. Use Random Forest or SVM (they support class_weight='balanced')")
        print(f"  2. For Neural Network: Balance the dataset manually before training")
        print(
            f"     - Undersample majority classes to match {min_class} ({min_count} samples)")
        print(f"     - Or oversample minority classes using SMOTE/augmentation")
