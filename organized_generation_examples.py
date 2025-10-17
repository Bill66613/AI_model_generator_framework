#!/usr/bin/env python3
"""
Usage Examples for Organized Deployment Code Generation

This script demonstrates how to use the new organized code generation system
that creates descriptive filenames and organized folder structures.
"""

from deployment import (
    generate_deployment_code,
    generate_and_save_deployment_code,
    get_deployment_info
)

def example_usage():
    """Examples of how to use the organized code generation system."""
    
    print("📖 Organized Code Generation Usage Examples")
    print("=" * 50)
    
    # Example model data (this would come from your actual trained model)
    example_model_data = {
        'model_type': 'random_forest',
        'feature_names': [f'feature_{i}' for i in range(138)],  # 138 features
        'classes': ['walking', 'running', 'standing', 'sitting'],  # 4 classes
        'model_object': None  # Would contain actual trained model
    }
    
    print("📊 Example Model Data:")
    print(f"   - Type: {example_model_data['model_type']}")
    print(f"   - Features: {len(example_model_data['feature_names'])}")
    print(f"   - Classes: {len(example_model_data['classes'])}")
    
    print("\n🎯 Usage Examples:")
    print("-" * 30)
    
    # Example 1: Get information about what would be generated
    print("\n1️⃣ Get Deployment Information (Preview)")
    print("   Code: get_deployment_info(model_type, model_data, platform)")
    
    info = get_deployment_info('random_forest', example_model_data, 'arduino')
    print(f"   📂 Folder: {info['folder_structure']}")
    print(f"   📄 Files that would be generated:")
    for file_info in info['files']:
        print(f"      - {file_info['filename']} ({file_info['type']})")
    
    # Example 2: Generate code with organized naming (in memory)
    print("\n2️⃣ Generate Code with Organized Names (In Memory)")
    print("   Code: generate_deployment_code(model_type, model_data, platform)")
    
    generated_code = generate_deployment_code('random_forest', example_model_data, 'arduino')
    print(f"   ✅ Generated {len(generated_code)} files:")
    for filename, content in generated_code.items():
        print(f"      - {filename} ({len(content):,} characters)")
    
    # Example 3: Generate and save to organized folders
    print("\n3️⃣ Generate and Save to Organized Folders")
    print("   Code: generate_and_save_deployment_code(model_type, model_data, platform, output_dir)")
    
    saved_files = generate_and_save_deployment_code(
        'random_forest', 
        example_model_data, 
        'arduino', 
        'example_output'
    )
    print(f"   💾 Saved files:")
    for filepath, status in saved_files.items():
        print(f"      - {filepath}")
        print(f"        {status}")
    
    print("\n📁 Filename Format Explanation:")
    print("-" * 30)
    print("   Format: har_{model_type}_{platform}_f{features}_c{classes}.{ext}")
    print("   Example: har_random_forest_arduino_f138_c4.cpp")
    print("   ")
    print("   Components:")
    print("   - har: Human Activity Recognition prefix")
    print("   - random_forest: Model type")
    print("   - arduino: Target platform")  
    print("   - f138: 138 features")
    print("   - c4: 4 classes")
    print("   - .cpp/.h: File extension")
    
    print("\n📂 Folder Structure:")
    print("-" * 20)
    print("   output_dir/")
    print("   ├── random_forest_models/")
    print("   │   ├── arduino/")
    print("   │   ├── seeed_xiao/")
    print("   │   └── arm_cortex_m/")
    print("   ├── neural_network_models/")
    print("   │   ├── arduino/")
    print("   │   └── arm_cortex_m/")
    print("   └── svm_models/")
    print("       ├── arduino/")
    print("       └── arm_cortex_m/")
    
    print("\n✅ Usage examples completed!")
    print("💡 Tip: Use get_deployment_info() first to preview what files will be created")

if __name__ == "__main__":
    example_usage()