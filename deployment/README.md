# Deployment Module

This directory contains the **source code** for the deployment code generation system, not the generated output files.

## Directory Contents

- `base_generator.py` - Abstract base class for all code generators
- `neural_network_generator.py` - Neural network specific code generator
- `random_forest_generator.py` - Random forest specific code generator  
- `svm_generator.py` - SVM specific code generator
- `arm_cortex_generator.py` - ARM Cortex-M specific code generator
- `code_generator_factory.py` - Factory pattern implementation
- `deployment_config.toml` - Configuration file
- `har_example_usr1.ino` - Template Arduino sketch

## Generated Files Location

**Generated deployment code files are saved to the organized directory structure:**

```text
generated/
├── neural_network_models/
│   ├── arduino/
│   ├── seeed_xiao/
│   └── arm_cortex_m/
├── random_forest_models/
│   ├── arduino/
│   ├── seeed_xiao/
│   └── arm_cortex_m/
└── svm_models/
    ├── arduino/
    ├── seeed_xiao/
    └── arm_cortex_m/
```

## Important Notes

- **DO NOT** manually place generated files in this directory
- Generated files are automatically organized by model type, platform, and optimization level
- This directory should only contain the generator source code and configuration files
- Generated code files include optimization level in their filenames (e.g., `_speed`, `_accuracy`, `_power`, `_balanced`)

## Usage

The deployment module is used automatically by the HAR framework when generating deployment code through the web interface.
