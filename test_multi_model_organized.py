#!/usr/bin/env python3
"""
Quick test to demonstrate organized generation with multiple model types
"""

import os
import json
from pathlib import Path
from utils.model_training import EdgeMLModel
from deployment import generate_and_save_deployment_code, get_deployment_info
from config.config import PERSISTENT_DIR

def test_multiple_models():
    """Test organized generation with different model types."""
    print("🧪 Testing Multiple Model Types - Organized Generation")
    print("=" * 60)
    
    # Find available model files
    model_files = [f for f in os.listdir(PERSISTENT_DIR) if f.endswith('.joblib')]
    
    print(f"📁 Found {len(model_files)} model files:")
    for f in model_files[:5]:  # Show first 5
        print(f"   - {f}")
    
    # Test with different model types
    test_models = []
    
    # Find models of different types
    for model_file in model_files:
        if 'random_forest' in model_file and not any('random_forest' in m[0] for m in test_models):
            test_models.append((model_file, 'random_forest'))
        elif 'neural_network' in model_file and not any('neural_network' in m[0] for m in test_models):
            test_models.append((model_file, 'neural_network'))
        
        if len(test_models) >= 2:  # Test with 2 different model types
            break
    
    if not test_models:
        test_models = [(model_files[0], 'unknown')]  # Use at least one model
    
    print(f"\n🎯 Testing {len(test_models)} different model types:")
    for model_file, expected_type in test_models:
        print(f"   - {model_file} (expected: {expected_type})")
    
    output_dir = "multi_model_deployment"
    
    for model_file, expected_type in test_models:
        print(f"\n{'='*50}")
        print(f"🚀 Testing: {model_file}")
        print(f"{'='*50}")
        
        model_path = os.path.join(PERSISTENT_DIR, model_file)
        
        try:
            # Load the model
            model = EdgeMLModel.load_model(model_path)
            print(f"✅ Model loaded: {model.model_type}")
            
            # Prepare model data
            model_data = {
                'model_type': model.model_type,
                'feature_names': model.feature_names or [],
                'classes': list(model.label_encoder.classes_) if model.label_encoder else ['activity_1', 'activity_2'],
                'model_object': model
            }
            
            print(f"📊 Model Info: {model.model_type} | {len(model_data['feature_names'])} features | {len(model_data['classes'])} classes")
            
            # Test with Arduino platform for this model
            platform = 'arduino'
            
            # Show what will be generated
            info = get_deployment_info(model.model_type, model_data, platform)
            print(f"📂 Will create folder: {info['folder_structure']}")
            print(f"📄 Files to generate:")
            for file_info in info['files']:
                print(f"   - {file_info['filename']}")
            
            # Generate and save
            saved_files = generate_and_save_deployment_code(
                model.model_type, model_data, platform, output_dir
            )
            
            print(f"💾 Generated files:")
            for filepath, status in saved_files.items():
                rel_path = os.path.relpath(filepath, output_dir)
                print(f"   - {rel_path}")
                print(f"     {status}")
                
        except Exception as e:
            print(f"❌ Error with {model_file}: {str(e)}")
    
    # Show final folder structure
    print(f"\n📁 Final Generated Structure:")
    print("=" * 40)
    
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            level = root.replace(output_dir, '').count(os.sep)
            indent = '  ' * level
            folder_name = os.path.basename(root) or output_dir
            print(f"{indent}{folder_name}/")
            
            subindent = '  ' * (level + 1)
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                print(f"{subindent}{file} ({file_size:,} bytes)")
    
    print(f"\n✅ Multi-model organized generation completed!")
    print(f"📍 Files saved in: {os.path.abspath(output_dir)}")

if __name__ == "__main__":
    test_multiple_models()