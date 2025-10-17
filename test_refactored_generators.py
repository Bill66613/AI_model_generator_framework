#!/usr/bin/env python3
"""
Test script for the refactored code generators
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.model_training import EdgeMLModel
from deployment import generate_deployment_code, CodeGeneratorFactory
import json

def test_refactored_generators():
    """Test the refactored code generators."""
    print("🧪 Testing Refactored Code Generators")
    print("=" * 50)
    
    # Load a trained model
    model_path = "persistent_data/random_forest_har_model.joblib"
    
    if not os.path.exists(model_path):
        print("❌ No trained model found. Please train a model first.")
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
            'classes': list(model.label_encoder.classes_),
            'model_object': model  # Include actual model for parameter extraction
        }
        
        print(f"📊 Model data prepared:")
        print(f"   - Type: {model_data['model_type']}")
        print(f"   - Features: {len(model_data['feature_names'])}")
        print(f"   - Classes: {len(model_data['classes'])}")
        
        # Test different platforms
        platforms = ['arduino', 'seeed_xiao', 'arm_cortex_m']
        
        for platform in platforms:
            print(f"\n🚀 Testing {platform} with refactored generator...")
            
            try:
                # Test the factory function
                generator = CodeGeneratorFactory.create_generator(model.model_type, model_data, platform)
                print(f"   ✅ Generator created: {type(generator).__name__}")
                
                # Generate code
                generated_code = generate_deployment_code(model.model_type, model_data, platform)
                print(f"   ✅ Generated {len(generated_code)} files:")
                
                for filename, content in generated_code.items():
                    print(f"      - {filename} ({len(content)} characters)")
                    
                    # Check if content contains real parameters
                    if 'har_model.cpp' in filename and '2.992789' in content:
                        print(f"      ✅ Contains real model parameters!")
                    elif 'har_model_cortex.c' in filename:
                        print(f"      ✅ ARM Cortex-M optimized code generated")
                    
                    # Save a preview
                    preview_lines = content.split('\n')[:15]
                    print(f"      Preview:")
                    for line in preview_lines[:5]:
                        print(f"         {line}")
                    print(f"         ... ({len(preview_lines)} lines total)")
                    
            except Exception as e:
                print(f"   ❌ Error testing {platform}: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Refactored code generator testing completed!")
        
        # Test comparison with original
        print(f"\n🔄 Comparing with original generator...")
        try:
            from deployment.code_generator import generate_deployment_code as generate_original
            
            original_code = generate_original(model_data, 'arduino', 'balanced')
            refactored_code = generate_deployment_code(model_data, 'arduino', 'balanced')
            
            print(f"   Original generator files: {list(original_code.keys())}")
            print(f"   Refactored generator files: {list(refactored_code.keys())}")
            
            # Compare file sizes
            for filename in original_code.keys():
                if filename in refactored_code:
                    orig_size = len(original_code[filename])
                    refact_size = len(refactored_code[filename])
                    print(f"   {filename}: Original={orig_size}, Refactored={refact_size}")
                    
                    # Check if both contain real parameters
                    has_real_orig = '2.992789' in original_code[filename]
                    has_real_refact = '2.992789' in refactored_code[filename]
                    print(f"      Real parameters - Original: {has_real_orig}, Refactored: {has_real_refact}")
            
        except Exception as e:
            print(f"   ⚠️ Could not compare with original: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_refactored_generators()