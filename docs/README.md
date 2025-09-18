# Human Activity Recognition Framework Documentation

## Table of Contents

1. [Installation Guide](#installation-guide)
2. [User Manual](#user-manual)
3. [API Reference](#api-reference)
4. [Developer Guide](#developer-guide)
5. [Deployment Guide](#deployment-guide)

## Installation Guide

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Minimum 4GB RAM recommended
- 1GB free disk space

### Quick Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/har-edge-framework.git
cd har-edge-framework

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch application
python app.py
```

### Detailed Installation

For detailed installation instructions, see [INSTALL.md](INSTALL.md).

## User Manual

### Getting Started

1. **Upload Data**: Import your IMU sensor data (CSV format)
2. **Preprocess**: Clean and segment your data using interactive tools
3. **Train Model**: Select and train machine learning models
4. **Deploy**: Generate code for your target edge device

### Data Format

Your CSV files should contain the following columns:
- `Timestamp`: Time information (optional)
- `aX`, `aY`, `aZ`: Accelerometer readings (m/s²)
- `gX`, `gY`, `gZ`: Gyroscope readings (rad/s)
- `Label`: Activity label (optional, for supervised learning)

### Interactive Preprocessing

The framework's unique draggable time window feature allows you to:
- Select precise activity boundaries by dragging rectangles
- Adjust window sizes in real-time
- Validate selections with live statistical feedback
- Export selected segments for training

## API Reference

### Core Classes

#### `EdgeMLModel`
Main class for machine learning operations.

```python
from utils.model_training import EdgeMLModel

# Create model
model = EdgeMLModel('random_forest')

# Train model
results = model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

#### Data Processing Functions

```python
from utils.data_processing import clean_data, low_pass_filter

# Clean data
cleaned_df = clean_data(df, method='remove_missing')

# Apply filtering
filtered_df = low_pass_filter(df, cutoff=5, fs=100)
```

### Callback Functions

The framework uses Dash callbacks for interactivity:
- `data_callbacks.py`: File upload and data management
- `preprocessing_callbacks.py`: Signal processing and windowing
- `training_callbacks.py`: Model training and evaluation

## Developer Guide

### Project Structure

```
├── app.py                      # Main application
├── requirements.txt            # Dependencies
├── config/                     # Configuration
├── layouts/                    # UI layouts
├── callbacks/                  # Interactive callbacks
├── utils/                      # Utility functions
├── tests/                      # Test suite
├── docs/                       # Documentation
├── deployment/                 # Deployment templates
├── persistent_data/            # Local storage
└── models/                     # Trained models
```

### Adding New Models

To add a new machine learning model:

1. Extend the `EdgeMLModel` class
2. Add model initialization in `_initialize_model()`
3. Update parameter grids in `_get_default_param_grid()`
4. Add model option to training layout

### Custom Preprocessing

To add custom preprocessing steps:

1. Add function to `utils/data_processing.py`
2. Create callback in `preprocessing_callbacks.py`
3. Add UI component to `layouts/preprocessing.py`

## Deployment Guide

### Supported Platforms

- **Arduino**: Generate C++ sketches
- **ARM Cortex-M**: Optimized C code
- **Seeed XIAO nRF52840**: Platform-specific optimizations
- **TensorFlow Lite Micro**: Cross-platform inference

### Edge Optimization

The framework provides several optimization strategies:
- **Model Quantization**: INT8/INT16 conversion
- **Feature Selection**: Reduce computational load
- **Memory Optimization**: Minimize RAM usage
- **Power Efficiency**: Optimize for battery life

### Code Generation

```python
# Example deployment code generation
model = EdgeMLModel.load_model('trained_model.joblib')
deployment_code = generate_deployment_code(
    model, 
    platform='arduino',
    optimization='balanced'
)
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.
