"""
Data Balancing Utility for Human Activity Recognition
Balances class distribution by undersampling or oversampling
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from config.config import PERSISTENT_DIR, METADATA_FILE
import json


def analyze_class_distribution(metadata_file: str = METADATA_FILE) -> Dict[str, int]:
    """
    Analyze the distribution of samples across activity classes.

    Returns:
        Dictionary mapping activity class names to sample counts
    """
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        return {}

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    class_counts = {}

    for dataset_name, dataset_info in metadata.items():
        if 'dragged_samples' in dataset_info:
            dataset_label = dataset_info.get(
                'label', dataset_name.replace('.csv', ''))
            sample_count = len(dataset_info['dragged_samples'])

            if dataset_label in class_counts:
                class_counts[dataset_label] += sample_count
            else:
                class_counts[dataset_label] = sample_count

    return class_counts


def undersample_to_min_class(metadata_file: str = METADATA_FILE,
                             output_suffix: str = '_balanced') -> Dict[str, List[str]]:
    """
    Balance dataset by undersampling majority classes to match the smallest class.

    This creates new metadata entries pointing to a subset of the window files.

    Args:
        metadata_file: Path to the metadata JSON file
        output_suffix: Suffix to add to balanced dataset names

    Returns:
        Dictionary mapping class names to lists of selected window files
    """
    print("=" * 60)
    print("DATA BALANCING UTILITY - Undersampling Strategy")
    print("=" * 60)

    # Load metadata
    if not os.path.exists(metadata_file):
        print(f"❌ Metadata file not found: {metadata_file}")
        return {}

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    # Organize windows by class
    class_windows = {}
    for dataset_name, dataset_info in metadata.items():
        if 'dragged_samples' in dataset_info:
            dataset_label = dataset_info.get(
                'label', dataset_name.replace('.csv', ''))

            if dataset_label not in class_windows:
                class_windows[dataset_label] = []

            class_windows[dataset_label].extend(
                dataset_info['dragged_samples'])

    # Display current distribution
    print("\n📊 Current Class Distribution:")
    print("-" * 60)
    for class_name in sorted(class_windows.keys()):
        print(
            f"  {class_name:25s}: {len(class_windows[class_name]):5d} windows")

    # Find minimum class size
    min_class_size = min(len(windows) for windows in class_windows.values())
    print(f"\n🎯 Target balanced size: {min_class_size} windows per class")

    # Undersample each class
    balanced_windows = {}
    np.random.seed(42)  # For reproducibility

    print("\n✂️ Undersampling to balance classes:")
    print("-" * 60)

    for class_name, windows in class_windows.items():
        if len(windows) > min_class_size:
            # Randomly select min_class_size windows
            selected_indices = np.random.choice(
                len(windows), min_class_size, replace=False)
            balanced_windows[class_name] = [windows[i]
                                            for i in selected_indices]
            print(
                f"  {class_name:25s}: {len(windows):5d} → {min_class_size:5d} windows (removed {len(windows) - min_class_size})")
        else:
            balanced_windows[class_name] = windows
            print(f"  {class_name:25s}: {len(windows):5d} windows (unchanged)")

    # Calculate total samples
    total_original = sum(len(windows) for windows in class_windows.values())
    total_balanced = sum(len(windows) for windows in balanced_windows.values())

    print("\n" + "=" * 60)
    print(f"📊 Balancing Summary:")
    print(f"  Original total: {total_original} windows")
    print(f"  Balanced total: {total_balanced} windows")
    print(f"  Removed: {total_original - total_balanced} windows ({100 * (total_original - total_balanced) / total_original:.1f}%)")
    print(f"  Classes: {len(balanced_windows)}")
    print(f"  Samples per class: {min_class_size}")
    print("=" * 60)

    # Save balanced window lists to a new metadata entry
    balanced_metadata_file = metadata_file.replace(
        '.json', '_balanced_info.json')

    with open(balanced_metadata_file, 'w') as f:
        json.dump(balanced_windows, f, indent=2)

    print(f"\n💾 Balanced dataset info saved to: {balanced_metadata_file}")
    print("\n✅ Data balancing complete!")
    print("\n📝 Next Steps:")
    print("  1. The system will now automatically use 'balanced' class weighting")
    print("  2. Go to your GUI app and retrain the model")
    print("  3. The model will treat all classes equally during training")
    print("  4. Generate new Arduino code and test on your device")

    return balanced_windows


def show_class_distribution_report(metadata_file: str = METADATA_FILE):
    """Display a detailed report of class distribution."""
    print("\n" + "=" * 60)
    print("CLASS DISTRIBUTION ANALYSIS REPORT")
    print("=" * 60)

    class_counts = analyze_class_distribution(metadata_file)

    if not class_counts:
        print("❌ No data found to analyze.")
        return

    total_samples = sum(class_counts.values())
    min_class = min(class_counts.items(), key=lambda x: x[1])
    max_class = max(class_counts.items(), key=lambda x: x[1])

    print(f"\n📊 Sample Distribution:")
    print("-" * 60)

    for class_name in sorted(class_counts.keys()):
        count = class_counts[class_name]
        percentage = 100 * count / total_samples
        bar_length = int(40 * count / max_class[1])
        bar = '█' * bar_length

        print(f"{class_name:25s}: {count:5d} ({percentage:5.1f}%) {bar}")

    print("-" * 60)
    print(f"{'Total':25s}: {total_samples:5d} (100.0%)")

    print(f"\n📈 Statistics:")
    print(f"  Minimum class: {min_class[0]} ({min_class[1]} samples)")
    print(f"  Maximum class: {max_class[0]} ({max_class[1]} samples)")
    print(f"  Imbalance ratio: {max_class[1] / min_class[1]:.2f}x")

    # Recommendation
    imbalance_ratio = max_class[1] / min_class[1]
    print(f"\n💡 Recommendation:")
    if imbalance_ratio > 2.0:
        print("  ⚠️  SEVERE CLASS IMBALANCE DETECTED!")
        print(
            f"  The largest class has {imbalance_ratio:.1f}x more samples than the smallest.")
        print("  This can cause models to be biased toward majority classes.")
        print("\n  Solutions:")
        print("  1. ✅ Use class_weight='balanced' in model training (NOW ENABLED)")
        print("  2. Run undersample_to_min_class() to balance your data")
        print("  3. Collect more data for minority classes")
    elif imbalance_ratio > 1.5:
        print("  ⚠️  Moderate class imbalance detected.")
        print(
            f"  The largest class has {imbalance_ratio:.1f}x more samples than the smallest.")
        print("  ✅ The class_weight='balanced' parameter will help mitigate this.")
    else:
        print("  ✅ Classes are reasonably balanced.")
        print("  The class_weight='balanced' parameter will ensure optimal training.")

    print("=" * 60)


if __name__ == "__main__":
    """Run the data balancing analysis and optionally perform undersampling."""

    print("\n" + "=" * 60)
    print("HAR DATA BALANCING UTILITY")
    print("=" * 60)

    # Show distribution report
    show_class_distribution_report()

    # Ask user if they want to balance the data
    print("\n" + "=" * 60)
    print("Would you like to perform undersampling to balance the classes?")
    print("This will NOT delete your original data.")
    print("=" * 60)

    response = input(
        "\nProceed with undersampling? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        balanced_windows = undersample_to_min_class()

        if balanced_windows:
            print("\n✅ Undersampling completed successfully!")
            print("\nYou can now retrain your model. The new model training includes:")
            print("  • class_weight='balanced' parameter (automatically applied)")
            print("  • Equal treatment of all activity classes")
            print("  • Better prediction accuracy across all activities")
    else:
        print("\n✅ No changes made to your data.")
        print("\nGood news: Your model training now includes class_weight='balanced'")
        print("This means the model will automatically compensate for class imbalance!")
        print("\nJust retrain your model in the GUI app to see improved results.")
