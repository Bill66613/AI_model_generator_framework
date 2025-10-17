# Organized Deployment Code Generation System

## 🎯 Overview

The deployment code generation system now creates **organized folder structures** with **descriptive filenames** that clearly identify which model and platform each generated file belongs to.

## 📁 Folder Structure

```
output_directory/
├── random_forest_models/
│   ├── arduino/
│   ├── seeed_xiao/
│   └── arm_cortex_m/
├── neural_network_models/
│   ├── arduino/
│   ├── seeed_xiao/
│   └── arm_cortex_m/
└── svm_models/
    ├── arduino/
    ├── seeed_xiao/
    └── arm_cortex_m/
```

## 🏷️ Filename Convention

**Format:** `har_{model_type}_{platform}_f{features}_c{classes}.{extension}`

**Examples:**
- `har_random_forest_arduino_f138_c4.h` - Header file for Random Forest on Arduino (138 features, 4 classes)
- `har_neural_network_seeed_xiao_f138_c4.cpp` - Neural Network source for Seeed Xiao
- `har_svm_arm_cortex_m_f138_c4.c` - SVM source for ARM Cortex-M

**Components:**
- `har` - Human Activity Recognition prefix
- `{model_type}` - Model type (random_forest, neural_network, svm)
- `{platform}` - Target platform (arduino, seeed_xiao, arm_cortex_m)
- `f{features}` - Number of features (e.g., f138 = 138 features)
- `c{classes}` - Number of classes (e.g., c4 = 4 classes)
- `{extension}` - File type (.h for headers, .cpp/.c for source)

## 🛠️ Available Functions

### 1. `get_deployment_info(model_type, model_data, platform)`
**Purpose:** Preview what files will be generated without actually creating them

**Returns:** Information about folder structure and files that would be created

**Example:**
```python
info = get_deployment_info('random_forest', model_data, 'arduino')
print(f"Folder: {info['folder_structure']}")
print(f"Files: {[f['filename'] for f in info['files']]}")
```

### 2. `generate_deployment_code(model_type, model_data, platform)`
**Purpose:** Generate code with organized naming (in memory only)

**Returns:** Dictionary with descriptive filenames as keys and code content as values

**Example:**
```python
generated_code = generate_deployment_code('random_forest', model_data, 'arduino')
for filename, content in generated_code.items():
    print(f"{filename}: {len(content)} characters")
```

### 3. `generate_and_save_deployment_code(model_type, model_data, platform, output_dir)`
**Purpose:** Generate code AND save to organized folder structure

**Returns:** Dictionary with full file paths and save status

**Example:**
```python
saved_files = generate_and_save_deployment_code(
    'random_forest', model_data, 'arduino', 'my_generated_code'
)
for filepath, status in saved_files.items():
    print(f"{filepath}: {status}")
```

## 🔧 Usage Examples

### Basic Usage (In Memory)
```python
from deployment import generate_deployment_code

# Generate code with organized naming
code_files = generate_deployment_code('random_forest', model_data, 'arduino')

# Files will have names like:
# - har_random_forest_arduino_f138_c4.h
# - har_random_forest_arduino_f138_c4.cpp
```

### Save to Organized Folders
```python
from deployment import generate_and_save_deployment_code

# Generate and save to organized folders
saved_files = generate_and_save_deployment_code(
    model_type='random_forest',
    model_data=your_model_data,
    platform='arduino',
    output_dir='deployment_code'
)

# Creates: deployment_code/random_forest_models/arduino/har_random_forest_arduino_f138_c4.*
```

### Preview Before Generation
```python
from deployment import get_deployment_info

# Preview what will be generated
info = get_deployment_info('neural_network', model_data, 'seeed_xiao')

print(f"Will create folder: {info['folder_structure']}")
for file_info in info['files']:
    print(f"File: {file_info['filename']} ({file_info['type']})")
```

## ✅ Benefits

1. **Clear Identification:** Filenames immediately tell you:
   - Which model type generated the code
   - Which platform it targets
   - How many features and classes it handles

2. **Organized Structure:** 
   - Models are separated by type
   - Platforms are separated within each model type
   - No naming conflicts between different models

3. **Easy Management:**
   - Find specific generated code quickly
   - Compare different model implementations
   - Deploy specific versions without confusion

4. **Scalable:**
   - Easy to add new model types
   - Easy to add new platforms
   - Maintains organization as project grows

## 📍 Generated File Locations

After running the examples, generated files are saved in:
- `generated_deployment_code/` - Single model, multiple platforms
- `multi_model_deployment/` - Multiple model types
- `example_output/` - Usage example files

Each location maintains the organized folder structure with descriptive filenames.