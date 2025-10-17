#!/usr/bin/env python3
"""
Test script to simulate the callback behavior with organized folder generation
"""

import os
import json
from utils.model_training import EdgeMLModel
from deployment import generate_and_save_deployment_code, generate_deployment_code
from config.config import PERSISTENT_DIR

def test_callback_simulation():
    """Test the callback behavior with organized folder structure."""
    print("🧪 Testing Callback Integration with Organized Structure")
    print("=" * 60)
    
    # Find a model to test with
    model_files = [f for f in os.listdir(PERSISTENT_DIR) if f.endswith('.joblib')]
    
    if not model_files:
        print("❌ No model files found for testing")
        return
    
    model_filename = model_files[0]  # Use first available model
    model_path = os.path.join(PERSISTENT_DIR, model_filename)
    
    print(f"📁 Testing with model: {model_filename}")
    
    try:
        # Load the model (simulating callback behavior)
        print("📤 Loading model...")
        model = EdgeMLModel.load_model(model_path)
        print(f"✅ Model loaded: {model.model_type}")
        
        # Prepare model data (same as in callback)
        model_data = {
            'model_type': model.model_type,
            'feature_names': model.feature_names or [],
            'classes': list(model.label_encoder.classes_) if model.label_encoder else ['activity_1', 'activity_2'],
            'model_params': model.model_params,
            'performance_metrics': model.performance_metrics,
            'model_object': model  # Add the actual model object for parameter extraction
        }
        
        # Test different platforms (simulating user selections)
        platforms = ['arduino', 'seeed_xiao', 'arm_cortex_m']
        
        for platform in platforms:
            print(f"\n🚀 Simulating callback for {platform}...")
            
            # This simulates the callback function behavior
            output_dir = "generated"  # Base directory for organized structure
            
            # Generate and save to organized folders (new behavior)
            saved_files = generate_and_save_deployment_code(
                model_data['model_type'], model_data, platform, output_dir)
            
            print(f"   ✅ Generated and saved {len(saved_files)} files:")
            for filepath, status in saved_files.items():
                rel_path = os.path.relpath(filepath, output_dir)
                print(f"      📂 {rel_path}")
                print(f"         {status}")
            
            # Also generate for display (existing behavior for UI)
            generated_code = generate_deployment_code(
                model_data['model_type'], model_data, platform)
            
            print(f"   📄 Generated code files for display:")
            for filename in generated_code.keys():
                print(f"      - {filename}")
        
        # Show final organized structure
        print(f"\n📁 Final Generated Structure (as seen in callback):")
        print("=" * 50)
        
        output_dir = "generated"
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
        
        print(f"\n✅ Callback simulation completed!")
        print(f"📍 When you click 'Generate Deployment Code', files will be saved to:")
        print(f"    {os.path.abspath(output_dir)}/[model_type]_models/[platform]/")
        
    except Exception as e:
        print(f"❌ Error during callback simulation: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_callback_simulation()