#!/usr/bin/env python3
"""
Test script for organized deployment code generation
Demonstrates the new folder structure and descriptive naming system
"""

import os
import json
from pathlib import Path
from utils.model_training import EdgeMLModel
from deployment import (
    generate_deployment_code, 
    generate_and_save_deployment_code, 
    get_deployment_info
)
from config.config import PERSISTENT_DIR

def test_organized_generation():
    """Test the new organized code generation system."""
    print("🧪 Testing Organized Deployment Code Generation")
    print("=" * 60)
    
    # Load metadata to find available models
    metadata_path = os.path.join(PERSISTENT_DIR, 'trained_models.json')
    if not os.path.exists(metadata_path):
        print("❌ No trained models metadata file found.")
        # Try to find a model file directly
        model_files = [f for f in os.listdir(PERSISTENT_DIR) if f.endswith('.joblib')]
        if model_files:
            print(f"📁 Found model files: {model_files}")
            # Use the first model file
            model_filename = model_files[0]
            model_path = os.path.join(PERSISTENT_DIR, model_filename)
        else:
            print("❌ No model files found.")
            return
    else:
        with open(metadata_path, 'r') as f:
            models_metadata = json.load(f)
        
        if not models_metadata:
            print("❌ No models in metadata file.")
            return
        
        # Use the first available model
        model_filename = list(models_metadata.keys())[0]
        model_path = os.path.join(PERSISTENT_DIR, model_filename)
    
    print(f"📁 Testing with model: {model_filename}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Load the model
        print("📤 Loading model...")
        model = EdgeMLModel.load_model(model_path)
        print(f"✅ Model loaded: {model.model_type}")
        
        # Prepare model data
        model_data = {
            'model_type': model.model_type,
            'feature_names': model.feature_names or [],
            'classes': list(model.label_encoder.classes_) if model.label_encoder else ['activity_1', 'activity_2'],
            'model_params': model.model_params,
            'performance_metrics': model.performance_metrics,
            'model_object': model  # Add the actual model object for parameter extraction
        }
        
        print(f"📊 Model Information:")
        print(f"   - Type: {model_data['model_type']}")
        print(f"   - Features: {len(model_data['feature_names'])}")
        print(f"   - Classes: {len(model_data['classes'])}")
        
        # Test different platforms
        platforms = ['arduino', 'seeed_xiao', 'arm_cortex_m']
        
        print(f"\n🎯 Testing New Organized Generation System")
        print("-" * 40)
        
        for platform in platforms:
            print(f"\n🚀 Testing {platform}...")
            
            # 1. Get deployment info (preview what would be generated)
            print("   📋 Getting deployment info...")
            info = get_deployment_info(model.model_type, model_data, platform)
            
            print(f"   📂 Folder: {info['folder_structure']}")
            print(f"   📄 Files to generate:")
            for file_info in info['files']:
                print(f"      - {file_info['filename']} ({file_info['type']})")
            
            # 2. Generate code with organized naming (in memory)
            print("   🔧 Generating organized code...")
            generated_code = generate_deployment_code(model.model_type, model_data, platform)
            
            print(f"   ✅ Generated {len(generated_code)} files with organized names:")
            for filename, content in generated_code.items():
                print(f"      - {filename} ({len(content)} characters)")
            
            # 3. Generate and save to organized folders
            print("   💾 Saving to organized folders...")
            output_dir = "generated_deployment_code"
            saved_files = generate_and_save_deployment_code(
                model.model_type, model_data, platform, output_dir
            )
            
            print(f"   📁 Saved files:")
            for filepath, status in saved_files.items():
                print(f"      - {filepath}")
                print(f"        {status}")
        
        print(f"\n📁 Generated Code Structure:")
        print("-" * 40)
        
        # Show the created folder structure
        output_dir = "generated_deployment_code"
        if os.path.exists(output_dir):
            for root, dirs, files in os.walk(output_dir):
                level = root.replace(output_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"{subindent}{file} ({file_size} bytes)")
        
        print(f"\n✅ Organized deployment code generation test completed!")
        print(f"📍 Generated files are saved in: {os.path.abspath(output_dir)}")
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_organized_generation()