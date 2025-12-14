"""
Migration Script for Reorganizing Persistent Data Storage
Moves files from flat structure to organized subdirectories
"""

import os
import shutil
import json
import glob
from pathlib import Path

# Import config to get paths
from config.config import (
    PERSISTENT_DIR, DATASETS_DIR, WINDOWS_DIR, TRAINING_DIR, MODELS_DIR, METADATA_FILE
)

def migrate_storage(dry_run=True):
    """
    Migrate files from flat persistent_data structure to organized subdirectories.
    
    Args:
        dry_run: If True, only print what would be moved without actually moving files
    """
    print("=" * 70)
    print("STORAGE MIGRATION SCRIPT")
    print("=" * 70)
    print(f"Mode: {'DRY RUN (no changes)' if dry_run else 'LIVE (files will be moved)'}")
    print()
    
    moved_files = {
        'datasets': [],
        'windows': [],
        'training': [],
        'models': []
    }
    
    # 1. Migrate cleaned dataset files (cleaned_smoothed_*.csv)
    print("📁 Migrating cleaned dataset files...")
    cleaned_files = glob.glob(os.path.join(PERSISTENT_DIR, "cleaned_smoothed_*.csv"))
    for file_path in cleaned_files:
        if os.path.dirname(file_path) == PERSISTENT_DIR:  # Only move if in root
            filename = os.path.basename(file_path)
            new_path = os.path.join(DATASETS_DIR, filename)
            print(f"   {filename} -> datasets/")
            if not dry_run:
                shutil.move(file_path, new_path)
            moved_files['datasets'].append(filename)
    
    # 2. Migrate raw dataset files (activity CSV files not in subdirectories)
    print("\n📁 Migrating raw dataset files...")
    # Pattern: files like laying_1.csv, walking_2.csv, etc. (not dragged_window or cleaned)
    raw_files = [f for f in glob.glob(os.path.join(PERSISTENT_DIR, "*.csv"))
                 if os.path.dirname(f) == PERSISTENT_DIR and 
                 not os.path.basename(f).startswith('dragged_window_') and
                 not os.path.basename(f).startswith('cleaned_smoothed_')]
    
    for file_path in raw_files:
        filename = os.path.basename(file_path)
        new_path = os.path.join(DATASETS_DIR, filename)
        print(f"   {filename} -> datasets/")
        if not dry_run:
            shutil.move(file_path, new_path)
        moved_files['datasets'].append(filename)
    
    # 3. Migrate dragged window files
    print("\n🪟 Migrating dragged window files...")
    window_files = glob.glob(os.path.join(PERSISTENT_DIR, "dragged_window_*.csv"))
    for file_path in window_files:
        if os.path.dirname(file_path) == PERSISTENT_DIR:  # Only move if in root
            filename = os.path.basename(file_path)
            new_path = os.path.join(WINDOWS_DIR, filename)
            print(f"   {filename} -> windows/")
            if not dry_run:
                shutil.move(file_path, new_path)
            moved_files['windows'].append(filename)
    
    # 4. Migrate training data files
    print("\n🎓 Migrating training data files...")
    old_training_dir = os.path.join(PERSISTENT_DIR, 'training_data')
    if os.path.exists(old_training_dir):
        training_files = glob.glob(os.path.join(old_training_dir, "*"))
        for file_path in training_files:
            filename = os.path.basename(file_path)
            new_path = os.path.join(TRAINING_DIR, filename)
            print(f"   {filename} -> training/")
            if not dry_run:
                shutil.move(file_path, new_path)
            moved_files['training'].append(filename)
        
        # Remove old training_data directory if empty
        if not dry_run and not os.listdir(old_training_dir):
            os.rmdir(old_training_dir)
            print(f"   Removed empty directory: training_data/")
    
    # 5. Migrate model files (.joblib)
    print("\n🤖 Migrating model files...")
    model_files = glob.glob(os.path.join(PERSISTENT_DIR, "*.joblib"))
    for file_path in model_files:
        if os.path.dirname(file_path) == PERSISTENT_DIR:  # Only move if in root
            filename = os.path.basename(file_path)
            new_path = os.path.join(MODELS_DIR, filename)
            print(f"   {filename} -> models/")
            if not dry_run:
                shutil.move(file_path, new_path)
            moved_files['models'].append(filename)
    
    # 6. Migrate trained_models.json metadata
    print("\n📋 Migrating model metadata...")
    old_models_metadata = os.path.join(PERSISTENT_DIR, "trained_models.json")
    if os.path.exists(old_models_metadata) and os.path.dirname(old_models_metadata) == PERSISTENT_DIR:
        new_models_metadata = os.path.join(MODELS_DIR, "trained_models.json")
        print(f"   trained_models.json -> models/")
        if not dry_run:
            shutil.move(old_models_metadata, new_models_metadata)
        moved_files['models'].append("trained_models.json")
    
    # 7. Update metadata.json file paths
    print("\n📝 Updating metadata.json file references...")
    if os.path.exists(METADATA_FILE) and not dry_run:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        updated = False
        for dataset_name, dataset_info in metadata.items():
            # Update cleaned_data_path if it exists
            if 'cleaned_data_path' in dataset_info:
                old_path = dataset_info['cleaned_data_path']
                if os.path.dirname(old_path) == PERSISTENT_DIR:
                    filename = os.path.basename(old_path)
                    new_path = os.path.join(DATASETS_DIR, filename)
                    dataset_info['cleaned_data_path'] = new_path
                    updated = True
                    print(f"   Updated {dataset_name} cleaned_data_path")
            
            # Update dragged_samples paths
            if 'dragged_samples' in dataset_info:
                new_samples = []
                for sample_path in dataset_info['dragged_samples']:
                    if os.path.dirname(sample_path) == PERSISTENT_DIR:
                        filename = os.path.basename(sample_path)
                        new_path = os.path.join(WINDOWS_DIR, filename)
                        new_samples.append(new_path)
                        updated = True
                    else:
                        new_samples.append(sample_path)
                dataset_info['dragged_samples'] = new_samples
                if updated:
                    print(f"   Updated {dataset_name} dragged_samples paths")
        
        if updated:
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
            print("   ✅ Metadata file updated")
    
    # Print summary
    print("\n" + "=" * 70)
    print("MIGRATION SUMMARY")
    print("=" * 70)
    print(f"📁 Datasets:  {len(moved_files['datasets'])} files")
    print(f"🪟 Windows:   {len(moved_files['windows'])} files")
    print(f"🎓 Training:  {len(moved_files['training'])} files")
    print(f"🤖 Models:    {len(moved_files['models'])} files")
    print(f"\nTotal: {sum(len(files) for files in moved_files.values())} files")
    
    if dry_run:
        print("\n⚠️  This was a DRY RUN - no files were actually moved")
        print("Run with dry_run=False to perform the migration")
    else:
        print("\n✅ Migration completed successfully!")
        print("\nNew directory structure:")
        print(f"   {DATASETS_DIR}/  - Raw and cleaned dataset files")
        print(f"   {WINDOWS_DIR}/   - Dragged window samples")
        print(f"   {TRAINING_DIR}/  - Train/val/test split files")
        print(f"   {MODELS_DIR}/    - Trained model files and metadata")
    
    print("=" * 70)
    
    return moved_files

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--execute":
        print("\n⚠️  WARNING: This will move files in your persistent_data directory!")
        response = input("Are you sure you want to proceed? (yes/no): ")
        if response.lower() == 'yes':
            migrate_storage(dry_run=False)
        else:
            print("Migration cancelled.")
    else:
        print("\nRunning in DRY RUN mode...\n")
        migrate_storage(dry_run=True)
        print("\nTo execute the migration, run:")
        print("  python migrate_storage.py --execute")
