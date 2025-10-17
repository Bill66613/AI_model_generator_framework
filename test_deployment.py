#!/usr/bin/env python3
"""
Test script to debug deployment code generation
"""

import os
import sys
import json

# Add the app directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import PERSISTENT_DIR
from utils.model_training import EdgeMLModel
from deployment import generate_deployment_code

def test_deployment_generation():
    """Test the deployment code generation with a real model."""
    print("🧪 Testing Deployment Code Generation")
    print("=" * 50)
    
    # Check if we have trained models
    model_metadata_file = os.path.join(PERSISTENT_DIR, "trained_models.json")
    
    if not os.path.exists(model_metadata_file):
        print("❌ No trained models found. Please train a model first.")
        return
    
    with open(model_metadata_file, 'r') as f:
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
        
        print(f"📊 Model data prepared:")
        print(f"   - Type: {model_data['model_type']}")
        print(f"   - Features: {len(model_data['feature_names'])}")
        print(f"   - Classes: {len(model_data['classes'])}")
        
        # Test code generation for different platforms
        platforms = ['arduino', 'seeed_xiao', 'arm_cortex_m']
        
        for platform in platforms:
            print(f"\n🚀 Generating code for {platform}...")
            try:
                generated_code = generate_deployment_code(model.model_type, model_data, platform)
                print(f"✅ Generated {len(generated_code)} files:")
                for filename in generated_code.keys():
                    print(f"   - {filename}")
                    # Show first 200 characters of each file
                    content = generated_code[filename]
                    preview = content[:200] + "..." if len(content) > 200 else content
                    print(f"     Preview: {preview}")
                    print()
            except Exception as e:
                print(f"❌ Failed to generate code for {platform}: {e}")
                import traceback
                traceback.print_exc()
        
        print("\n✅ Deployment code generation test completed!")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_deployment_generation()