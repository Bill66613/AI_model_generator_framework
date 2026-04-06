# Deployment Approaches: From Trained Model to Running Device

This document explains the three deployment approaches supported by the framework, what happens at each stage of the pipeline, and the technical rationale behind each approach.

## High-Level Pipeline

```
┌──────────────┐     ┌───────────────────┐     ┌────────────────┐     ┌──────────────┐
│  Train Model │────▶│  Code Generation  │────▶│  Compile/Build │────▶│ Flash/Deploy │
│  (Python)    │     │  (Framework)      │     │  (Toolchain)   │     │ (Device)     │
└──────────────┘     └───────────────────┘     └────────────────┘     └──────────────┘
     .joblib          .h / .cpp / .ino           .bin / .elf           Running on MCU
                      .tflite / .onnx
```

All three approaches share the same **feature extraction** code — the parity-critical C++ implementation that computes identical statistical features as Python. They differ in how the **model inference** step is handled.

---

## Approach 1: Direct C++ Code Generation (Default)

### What Happens

```
Trained Model (.joblib)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ CodeGeneratorFactory                             │
│                                                  │
│  1. Load model weights, biases, thresholds       │
│  2. Reorder from Python (alphabetical) order     │
│     to C++ (computation) order                   │
│  3. Embed all parameters as C arrays/constants   │
│  4. Generate model-specific prediction function  │
└─────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────┐
│ Generated Files                           │
│                                           │
│  har_model.h    ← Constants, declarations │
│  har_model.cpp  ← Full implementation     │
│  har_model.ino  ← Arduino sketch          │
└───────────────────────────────────────────┘
```

### What's Inside the Generated Code

**Header file (`.h`)** — compile-time constants:

```cpp
#define NUM_FEATURES 33
#define NUM_CLASSES 5
#define WINDOW_SIZE 150
#define SAMPLING_RATE 100
#define N_CHANNELS 6

// Scaler parameters (from sklearn's StandardScaler)
extern const float FEATURE_MEANS[NUM_FEATURES];  // mean of each feature from training
extern const float FEATURE_STDS[NUM_FEATURES];   // std of each feature from training

// Model parameters (depends on model type)
// Random Forest: tree structures with split thresholds
// Neural Network: weight matrices and bias vectors
// SVM: support vectors and dual coefficients
```

**Implementation file (`.cpp`)** — the actual logic:

```cpp
// 1. Feature extraction — identical to Python's create_feature_vector()
void extract_features(float buffer[][6], int window_size, float features[]) {
    // Compute acc_magnitude = sqrt(aX² + aY² + aZ²) per sample
    // Compute gyro_magnitude, jerk_magnitude
    // For each magnitude signal: compute 15 statistics
    //   (mean, std, min, max, range, median, q25, q75, iqr,
    //    skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate)
}

// 2. Feature scaling — reproduces StandardScaler.transform()
void scale_features(float features[]) {
    for (int i = 0; i < NUM_FEATURES; i++)
        features[i] = (features[i] - FEATURE_MEANS[i]) / FEATURE_STDS[i];
}

// 3. Model-specific prediction
int har_predict_internal(float features[], float probabilities[]) {
    // Random Forest: traverse each tree, aggregate votes
    // Neural Network: forward pass through layers (matmul + ReLU/softmax)
    // SVM: compute kernel values, accumulate decision function
}
```

### Why No Runtime Needed

The model's learned parameters (weights, thresholds, support vectors) are converted into **plain C arrays and arithmetic operations**. The prediction function is just nested `if` statements (RF), matrix multiplications (NN), or dot products (SVM). There's no model file to parse — everything is compiled directly into the firmware's `.text` and `.rodata` sections.

### Pros and Cons

| Advantage | Disadvantage |
|-----------|-------------|
| Smallest binary size | Each model type needs its own generator |
| Fastest inference (no interpreter overhead) | Generated code can be large for big models |
| No external dependencies | Changing model requires regenerating + reflashing |
| Full transparency — every operation visible | Complex models (deep CNNs) hard to generate |
| Works on the smallest MCUs (2KB RAM) | |

---

## Approach 2: TensorFlow Lite for Microcontrollers (TFLite Micro)

### What Happens

```
Trained Model (.joblib)
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│ TFLiteConverter                                          │
│                                                          │
│  sklearn MLP / PyTorch MLP:                              │
│    → Reconstruct as Keras Sequential model               │
│    → Copy weights layer-by-layer                         │
│    → tf.lite.TFLiteConverter.from_keras_model()          │
│                                                          │
│  PyTorch CNN:                                            │
│    → Reconstruct Conv1D → Pool → Dense as Keras          │
│    → Transpose weight axes (PyTorch ↔ TF convention)     │
│    → tf.lite.TFLiteConverter.from_keras_model()          │
│                                                          │
│  sklearn RF / SVM:                                       │
│    → skl2onnx: convert to ONNX format                   │
│    → onnx-tf: convert ONNX to TF SavedModel             │
│    → tf.lite.TFLiteConverter.from_saved_model()          │
│                                                          │
│  Optional quantization:                                  │
│    → float16: reduce weight precision (2x smaller)       │
│    → int8: full quantization with calibration data       │
│    → int16: intermediate precision                       │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ .tflite file (FlatBuffer binary)                         │
│                                                          │
│  Contains:                                               │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Schema version                                     │  │
│  │ Operator codes: [FULLY_CONNECTED, RELU, SOFTMAX..] │  │
│  │ Subgraphs:                                         │  │
│  │   └─ Tensors: shapes, types, buffer indices        │  │
│  │   └─ Operators: which op, input/output tensor IDs  │  │
│  │ Buffers:                                           │  │
│  │   └─ Weight data (float32 or quantized int8)       │  │
│  │   └─ Bias data                                     │  │
│  │   └─ Quantization parameters (scale, zero_point)   │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────────┐
│ TFLiteMicroCodeGenerator                                  │
│                                                          │
│  1. Convert .tflite bytes to C hex array:                │
│     const unsigned char g_har_model[] = {                │
│       0x20, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33,  │
│       ...                                                │
│     };                                                   │
│                                                          │
│  2. Generate TFLite Micro interpreter wrapper:           │
│     - tflite_init(): load model, allocate arena          │
│     - tflite_predict(): copy features → invoke → read    │
│                                                          │
│  3. Feature extraction: reuse base C++ implementation    │
└──────────────────────────────────────────────────────────┘
    │
    ▼
┌───────────────────────────────────────────────┐
│ Generated Files                               │
│                                               │
│  har_tflite_model.h     ← Constants + model   │
│  har_tflite_model.cpp   ← Feature extraction  │
│                            + interpreter code │
│  har_tflite_model.ino   ← Arduino sketch      │
│  har_tflite_model.tflite ← Raw model file     │
└───────────────────────────────────────────────┘
```

### What the .tflite File Contains

A `.tflite` file is a [FlatBuffer](https://google.github.io/flatbuffers/) — a serialized binary format designed for efficient access without parsing/unpacking. It contains:

1. **Model schema version** — compatibility check
2. **Operator list** — which operations the model uses (e.g., `FULLY_CONNECTED`, `CONV_2D`, `RELU`, `SOFTMAX`)
3. **Computation graph** — a directed acyclic graph of operators, with each operator specifying its input and output tensors
4. **Tensor metadata** — for each tensor: name, data type (float32, int8, etc.), shape, and which buffer holds its data
5. **Weight buffers** — the actual learned parameters, stored as binary data
6. **Quantization parameters** — if quantized: scale factor and zero point per tensor, so `real_value = (int_value - zero_point) * scale`

### What the Runtime Interpreter Does

The TFLite Micro interpreter is a ~20KB library that:

1. **Parses the model graph** from the FlatBuffer at initialization
2. **Allocates a tensor arena** — a single pre-allocated memory block where all intermediate computation results are stored (no dynamic `malloc`)
3. **For each inference call**:
   - Copies input features into the input tensor
   - Walks the computation graph in topological order
   - For each operator: reads input tensors → executes the operation (matmul, ReLU, etc.) → writes output tensor
   - The output tensor contains class probabilities

```cpp
// Simplified interpreter internals:
for (int node_idx = 0; node_idx < graph.num_nodes; node_idx++) {
    TfLiteNode* node = &graph.nodes[node_idx];
    TfLiteRegistration* op = &graph.registrations[node->op_index];

    // op->invoke() does the actual computation:
    //   FULLY_CONNECTED: output = matmul(input, weights) + bias
    //   RELU: output = max(0, input)
    //   SOFTMAX: output = exp(input) / sum(exp(input))
    op->invoke(context, node);
}
```

### Why TFLite Micro Needs a Runtime

Unlike direct code generation, the model's structure is **not hardcoded** into the firmware. Instead:

- The **model** (weights + graph) is stored as data (the byte array)
- The **interpreter** (the runtime) is generic code that can execute any model

This is analogous to the difference between:

- **Compiled code** (direct): `y = w1*x1 + w2*x2 + b` — the formula is in the instructions
- **Interpreted code** (TFLite): "read the formula from this data blob, then execute it" — the formula is in the data

### Pros and Cons

| Advantage | Disadvantage |
|-----------|-------------|
| Supports any TF-compatible model | Larger binary (~20KB interpreter overhead) |
| Quantization built-in (int8, float16) | Requires more RAM (tensor arena) |
| Model can be updated without recompiling | Slower inference (interpreter dispatch) |
| Validated by Google's testing infrastructure | Needs TensorFlow installed for conversion |
| Standard format used by Edge Impulse, etc. | RF/SVM conversion is indirect (→ ONNX → TF → TFLite) |

---

## Approach 3: ONNX Runtime

### What Happens

```
Trained Model (.joblib)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│ ONNXConverter                                        │
│                                                      │
│  sklearn (RF / SVM / MLP):                           │
│    → skl2onnx: Pipeline(scaler, model) → ONNX       │
│    → Scaler is included in the ONNX graph!           │
│                                                      │
│  PyTorch (MLP / CNN):                                │
│    → torch.onnx.export(model, dummy_input)           │
│    → Traces the forward() method                     │
│    → Captures all operations as ONNX ops             │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ .onnx file (Protocol Buffer binary)                   │
│                                                      │
│  Contains:                                           │
│  ┌────────────────────────────────────────────────┐  │
│  │ IR version + opset version                     │  │
│  │ Graph:                                         │  │
│  │   └─ Nodes: [MatMul, Add, Relu, Softmax, ...]  │  │
│  │   └─ Initializers: weight tensors (numpy data) │  │
│  │   └─ Inputs: name, shape, data type            │  │
│  │   └─ Outputs: name, shape, data type           │  │
│  │ Metadata: model type, feature names, classes   │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────────────┐
│ ONNXRuntimeCodeGenerator                              │
│                                                      │
│  MCU mode:                                           │
│    → Embed .onnx as C byte array                     │
│    → Generate placeholder inference (runtime needed)  │
│                                                      │
│  Full runtime mode (Linux/RPi):                      │
│    → Save .onnx file alongside code                  │
│    → Generate Ort::Session loading + inference code   │
└──────────────────────────────────────────────────────┘
```

### What the .onnx File Contains

ONNX (Open Neural Network Exchange) uses Protocol Buffers to store:

1. **IR version** — format version for backward compatibility
2. **Opset version** — which operator set (determines available ops and their semantics)
3. **Computation graph** — nodes, each with:
   - Operator type (`MatMul`, `Gemm`, `Conv`, `Relu`, `TreeEnsembleClassifier`, etc.)
   - Input and output tensor names
   - Attributes (e.g., kernel sizes, activation types)
4. **Initializers** — the learned weight tensors, stored as numpy-compatible arrays
5. **Metadata** — custom properties (model type, feature names, class labels)

**Key difference from TFLite**: ONNX has native operators for tree ensembles (`TreeEnsembleClassifier`) and SVMs (`SVMClassifier`), so RF and SVM models convert cleanly without intermediate steps.

### When to Use ONNX Runtime

- **Full ONNX Runtime** (Ort::Session): for Linux-based devices (Raspberry Pi, Jetson Nano, embedded Linux)
- **ONNX Runtime Micro**: experimental, for Cortex-M MCUs (limited operator support)
- **Validation**: useful for verifying model conversion correctness regardless of target

### Pros and Cons

| Advantage | Disadvantage |
|-----------|-------------|
| Native RF/SVM operators (clean conversion) | ONNX Runtime Micro is less mature than TFLite Micro |
| Cross-framework standard (PyTorch, sklearn, TF) | Full runtime is large (~10MB+) |
| Good for Linux-based edge devices | Limited MCU support |
| sklearn pipelines convert as-is (scaler included) | Fewer quantization options than TFLite |

---

## Comparison: Which Approach to Choose

```
                    ┌─────────────────────────────────┐
                    │     Choose Deployment Approach   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │  Target device type?         │
                    └──────┬───────────┬──────────┘
                           │           │
                    Bare-metal MCU    Linux-based
                    (ESP32, nRF,     (RPi, Jetson)
                     STM32, AVR)          │
                           │              ▼
                           │      ┌──────────────────┐
                           │      │  ONNX Runtime     │
                           │      │  (full runtime)   │
                           │      └──────────────────┘
                           │
                    ┌──────▼──────────────┐
                    │  Model type?         │
                    └──────┬─────┬────────┘
                           │     │
                      RF/SVM/   CNN (deep) or
                      small NN  need quantization
                           │          │
                           ▼          ▼
                    ┌───────────┐ ┌───────────────┐
                    │  Direct   │ │  TFLite Micro  │
                    │  C++ code │ │  (interpreter)  │
                    └───────────┘ └───────────────┘
```

| Criterion | Direct C++ | TFLite Micro | ONNX Runtime |
|-----------|-----------|-------------|-------------|
| **Binary size** | Smallest | +20KB interpreter | +10MB runtime |
| **Inference speed** | Fastest | ~2-5x slower | ~2-3x slower |
| **Min RAM** | 2 KB | 4-64 KB arena | 256+ KB |
| **Quantization** | Manual (in generator) | Built-in (int8, float16) | Limited |
| **Model update** | Requires reflash | Can swap model blob | Can swap .onnx file |
| **Supported models** | RF, SVM, MLP, CNN | Any TF-compatible | Any ONNX-compatible |
| **Conversion complexity** | None (direct extraction) | Medium (Keras rebuild) | Low (skl2onnx/torch) |
| **Best for** | Production MCU | Flexible MCU | Edge Linux |

---

## Feature Extraction: The Common Foundation

Regardless of deployment approach, **feature extraction is always the same C++ code**. This is the framework's critical parity guarantee:

```
Python (training)                    C++ (deployment)
─────────────────                    ────────────────
pandas DataFrame                     float buffer[150][6]
    │                                     │
    ▼                                     ▼
create_feature_vector()              extract_features()
    │                                     │
    ├─ acc_magnitude stats (11)      ├─ acc_magnitude stats (11)
    ├─ gyro_magnitude stats (11)     ├─ gyro_magnitude stats (11)
    ├─ jerk_magnitude stats (11)     ├─ jerk_magnitude stats (11)
    └─ DFT features (14)            └─ DFT features (14)
    │                                     │
    ▼                                     ▼
StandardScaler.transform()           scale_features()
    │                                     │
    ▼                                     ▼
model.predict()                      {direct / tflite / onnx}_predict()
```

The **feature extraction + scaling** are always generated as hand-coded C++. Only the final `predict()` call differs between approaches:

- **Direct**: hand-coded C++ arithmetic
- **TFLite**: `interpreter->Invoke()`
- **ONNX**: `session->Run()`
