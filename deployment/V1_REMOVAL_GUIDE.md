# V1 Code Generator Removal Guide

This document describes how to fully remove the v1 (monolithic) code generators
and replace them with the v2 (modular) architecture already in `deployment/v2/`.

---

## Current state

The factory in `code_generator_factory.py` routes requests to v2 or v1 based on:

| Condition | Route |
|---|---|
| Model type in `_V2_SUPPORTED_MODELS` AND platform not in `{micropython, zephyr}` | **→ v2** |
| Platform = `micropython` | → v1 `MicroPythonCodeGenerator` |
| Platform = `zephyr` | → v1 `ZephyrCodeGenerator` |
| Model type = `pytorch_cnn` / `pytorch_cnn2d` | → v1 `CNNCodeGenerator` |
| `create_generator()` called directly (internal/legacy) | → v1 class dispatch |

`_V2_SUPPORTED_MODELS = frozenset(['random_forest', 'neural_network', 'pytorch_mlp', 'svm'])`

---

## Files that CAN be deleted once v1 migration is complete

These v1 files have a direct v2 replacement already:

| V1 file | Replaced by |
|---|---|
| `base_generator.py` | `v2/feature_block.py` + `v2/classifier_block.py` + `v2/generator.py` |
| `random_forest_generator.py` | `v2/models/random_forest.py` |
| `neural_network_generator.py` | `v2/models/neural_network.py` |
| `svm_generator.py` | `v2/models/svm.py` |
| `arm_cortex_generator.py` | v2 routes `arm_cortex_m` to `ArduinoSketch` with generic config |
| `tflite_generator.py` | `v2/models/tflite.py` |
| `onnx_generator.py` | `v2/models/onnx_model.py` |
| `har_example_usr1.ino` | Generated per-model by v2 as `{sketch_name}.ino` |
| `har_example_usr2.ino` | Generated per-model by v2 as `{sketch_name}.ino` |

---

## Files that MUST be kept until migrated

These v1 files have no v2 replacement yet:

| V1 file | Why still needed | Migration task |
|---|---|---|
| `micropython_generator.py` | `micropython` platform still routes to v1 | Create `v2/platforms/micropython.py` + `v2/models/` equivalents |
| `zephyr_generator.py` | `zephyr` platform still routes to v1 | Create `v2/platforms/zephyr.py` |
| `cnn_generator.py` | `pytorch_cnn` / `pytorch_cnn2d` still route to v1 | Create `v2/models/cnn.py` using existing `v2/models/neural_network.py` as template |

The following support files are used by both v1 and v2 — keep them:

| File | Used by |
|---|---|
| `converters/` (directory) | `v2/models/tflite.py`, `v2/models/onnx_model.py` |
| `validation.py` | Factory validation checks |
| `quantization.py` | Factory and v1 generators |
| `deployment_config.toml` | Reference documentation only |

---

## Step-by-step removal plan

### Step 1 — Migrate `pytorch_cnn` / `pytorch_cnn2d` to v2

1. Create `deployment/v2/models/cnn.py` with a `CNNModelBlock` class.
   - Extract the CNN layer structure from `cnn_generator.py` `_generate_prediction_function()`
   - Implement `generate() -> Tuple[str, str]` returning `(har_model.h, har_model.cpp)`
2. Register it in `v2/models/__init__.py`:
   ```python
   from .cnn import CNNModelBlock
   # in create_model_block():
   if model_type in ('pytorch_cnn', 'pytorch_cnn2d'):
       return CNNModelBlock(...)
   ```
3. Add `'pytorch_cnn'` and `'pytorch_cnn2d'` to `_V2_SUPPORTED_MODELS` in the factory.
4. Verify with the existing test suite that generated C++ still compiles.
5. Delete `cnn_generator.py`.

### Step 2 — Migrate `micropython` platform to v2

1. Create `deployment/v2/platforms/micropython.py` with a `MicroPythonSketch` class.
   - Port the MicroPython feature extraction from `micropython_generator.py`
   - **CRITICAL**: Any formula in `v2/feature_block.py` that was also fixed in
     `micropython_generator.py` must be verified for parity.
   - `generate()` returns the `.py` file content (not `.ino`).
2. Update `v2/platforms/__init__.py`:
   ```python
   from .micropython import MicroPythonSketch
   # in create_platform_sketch():
   if p == 'micropython':
       return MicroPythonSketch(...)
   ```
3. Remove `'micropython'` from the platform exclusion list in `generate_deployment_code()`:
   ```python
   # Change:
   and platform not in ('micropython', 'zephyr')
   # To:
   and platform not in ('zephyr',)    # or remove entirely once zephyr is done too
   ```
4. Delete `micropython_generator.py`.

### Step 3 — Migrate `zephyr` platform to v2

1. Create `deployment/v2/platforms/zephyr.py` with a `ZephyrSketch` class.
   - Port Zephyr RTOS sensor loop from `zephyr_generator.py`
   - `generate()` returns `main.c` content (Zephyr uses CMake + C, not Arduino)
   - Optionally return a `CMakeLists.txt` stub
2. Update `v2/platforms/__init__.py` similarly to Step 2.
3. Remove `'zephyr'` from the platform exclusion list.
4. Delete `zephyr_generator.py`.

### Step 4 — Remove v1 factory routing

Once Steps 1–3 are complete, simplify `code_generator_factory.py`:

1. Remove all v1 imports (lines 10–19):
   ```python
   # DELETE these imports:
   from .base_generator import BaseCodeGenerator, ...
   from .random_forest_generator import RandomForestCodeGenerator
   from .neural_network_generator import NeuralNetworkCodeGenerator
   from .svm_generator import SVMCodeGenerator
   from .arm_cortex_generator import ARMCortexMCodeGenerator
   from .micropython_generator import MicroPythonCodeGenerator
   from .zephyr_generator import ZephyrCodeGenerator
   from .cnn_generator import CNNCodeGenerator
   from .tflite_generator import TFLiteMicroCodeGenerator
   from .onnx_generator import ONNXRuntimeCodeGenerator
   ```
2. Remove `CodeGeneratorFactory._generators` dict and `create_generator()` method.
3. Remove `get_supported_models()` / `register_generator()` or reimplement using v2 data.
4. Remove the `# --- Direct code generation (default) ---` fallback block in
   `generate_deployment_code()`.
5. Update `_V2_SUPPORTED_MODELS` to include all supported model types.

### Step 5 — Delete v1 files

After all tests pass:

```powershell
# From deployment/ directory:
Remove-Item base_generator.py
Remove-Item random_forest_generator.py
Remove-Item neural_network_generator.py
Remove-Item svm_generator.py
Remove-Item arm_cortex_generator.py
Remove-Item tflite_generator.py
Remove-Item onnx_generator.py
Remove-Item cnn_generator.py          # after Step 1
Remove-Item micropython_generator.py  # after Step 2
Remove-Item zephyr_generator.py       # after Step 3
Remove-Item har_example_usr1.ino
Remove-Item har_example_usr2.ino
```

---

## Test checklist before deletion

Run after each step to confirm nothing is broken:

```bash
uv run pytest tests/test_main.py -v
```

Additionally, manually verify:
- [ ] Code Gen tab in the Dash app generates files without errors for each model type
- [ ] `validate_deployment.py` passes for a generated sketch
- [ ] Device Test tab can parse serial output from a flashed device
- [ ] `micropython` output is syntactically valid Python (if migrated)
- [ ] `zephyr` output has correct CMake + Zephyr RTOS includes (if migrated)

---

## Risk notes

**MicroPython parity risk**: `micropython_generator.py` has its own copy of every
statistical formula (it cannot `#include` C headers). When migrating, compare each
formula in `micropython_generator.py` against `v2/feature_block.py` line by line.
Any discrepancy that was already present in v1 should be fixed in v2's MicroPython
platform, not silently carried over.

**CNN weight format**: `cnn_generator.py` reads `pytorch_state_dict` directly from the
`.joblib` file. `v2/models/neural_network.py` reads `pytorch_coefs` (pre-exported via
`PyTorchTrainer.export_mlp_weights()`). For CNN, the analogous `export_cnn_weights()`
output lives under `pytorch_cnn_weights` in the model dict. Confirm the key name before
implementing `v2/models/cnn.py`.
