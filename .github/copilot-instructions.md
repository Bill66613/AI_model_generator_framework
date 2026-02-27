# HAR Edge Deployment Framework — Copilot Instructions

## Thesis Context & Documentation Policy

This is a **master's thesis project** (student: Nguyen Truong Minh Hoang, MSSV: 2270757, supervisor: Dr. Le Trong Nhan). The thesis report is in `academic-paper-vietnamese/` and must be written in Vietnamese.

**IMPORTANT — When making any code change or discovering any technical issue:**
- Ask yourself: *"Is this a novel finding, interesting bug, or differentiating feature vs commercial platforms (Edge Impulse, SensiML)?"*
- If yes → document it in `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` and note it in `academic-paper-vietnamese/THESIS_REPORT_INSTRUCTIONS.md`
- These files are structured for cross-session continuity. Always read them before working on thesis-related tasks.
- `TECHNICAL_FINDINGS.md` = detailed technical evidence (code, data, math). Structured by numbered findings with subsections: problem, evidence, impact, solution.
- `THESIS_REPORT_INSTRUCTIONS.md` = TODO list, chapter mapping, LaTeX snippets, defense prep, Vietnamese terminology glossary.

## Architecture Overview

Dash 2.14+ single-page app with 6 sequential tabs (Data → Preprocess → Feature Engineering → Train → Code Gen → Device Test). State flows via filesystem (`persistent_data/`) and `dcc.Store` components. `suppress_callback_exceptions=True` is required — all layouts are eagerly loaded.

```
app.py                  ← Entry point, registers layouts then callbacks
layouts/                ← One file per tab, each exports `layout` variable
callbacks/              ← One file per tab, each exports `register_callbacks(app)`
utils/                  ← Core logic (see Utils section below)
deployment/             ← Code generators: BaseCodeGenerator → model/platform subclasses
config/config.py        ← Central constants: SENSOR_COLUMNS, paths, sampling rate
persistent_data/        ← Runtime data: datasets/, windows/, training/, models/, generated/
academic-paper-vietnamese/ ← Thesis LaTeX source + instruction files for report/defense
```

## Critical Patterns

**Callback registration**: Each `callbacks/*.py` defines a `register_callbacks(app)` function. All `@app.callback` decorators live *inside* that function (closure over `app`). Called explicitly in `app.py` after layout is set.

**Config imports**: Always use named imports: `from config.config import PERSISTENT_DIR, SENSOR_COLUMNS, MODELS_DIR`. The canonical sensor columns are `['aX','aY','aZ','gX','gY','gZ']`.

**Model serialization**: `joblib` exclusively. Models are saved as dicts with keys: `model`, `scaler`, `label_encoder`, `feature_names`, `model_type`, `model_params`, `performance_metrics`. PyTorch models additionally store `pytorch_state_dict` and `pytorch_weights`. Stored in `persistent_data/models/` as `*.joblib`.

**Feature types** (6 modes set in Feature Engineering tab):
- `orientation_invariant_time_only` → 33 features (acc_mag + gyro_mag + jerk × 11 stats each)
- `orientation_invariant` → 47 features (above + DFT magnitude features)
- `time_domain` → 90 features (15 stats × 6 axes)
- `all` → 138 features (time + frequency per axis)
- `frequency_domain` → 48 | `raw` → 6

**Feature reordering at code-gen**: Python trains on alphabetically-sorted DataFrame columns. C++ extracts in fixed order (acc_mag stats → gyro_mag stats → jerk). `CodeGeneratorFactory.reorder_model_parameters()` remaps weights/scaler at generation time — never change feature extraction order in C++ without updating this.

**Training-deployment parity** (CRITICAL): The framework guarantees identical feature computation between Python (`utils/feature_extraction.py`) and generated C++ (`deployment/base_generator.py`). Any change to a statistical formula in one MUST be mirrored in the other. Key verified formulas: kurtosis, skewness (use population std for z-scores), zero-crossing rate, mean-crossing rate. See `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` for the full parity checklist.

## Data Flow Between Tabs

1. **Data** → CSVs uploaded to `persistent_data/datasets/`, registered in `metadata.json`
2. **Preprocess** → Cleaned files (`cleaned_smoothed_*`), dragged windows saved to `persistent_data/windows/`
3. **Feature Engineering** → Reads windows, pads with edge-value replication, extracts features, splits train/val/test → `persistent_data/training/*.csv` + `*_fe_metadata.json`
4. **Training** → Reads `*_train.csv`, trains model → `persistent_data/models/*.joblib`
5. **Code Gen** → Loads `.joblib`, reorders features to C++ order, generates `.h`/`.cpp`/`.ino` → `persistent_data/generated/{model_dir}/`
6. **Device Test** → Serial communication with flashed device

## Utils Module

| File | Purpose |
|------|---------|
| `feature_extraction.py` | Core FE: `create_feature_vector()` dispatches to orientation-invariant, per-axis, frequency modes. 15 stats per signal: mean, std, min, max, range, median, q25, q75, iqr, skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate |
| `edge_ml_model.py` | `EdgeMLModel` class — unified train/predict/save/load for RF, SVM, MLP, PyTorch MLP/CNN. All models default to `class_weight='balanced'` |
| `training_pipeline.py` | `prepare_training_data()` (loads window CSVs → feature matrix), `create_model()` factory |
| `model_training.py` | **Backward-compat shim** — re-exports everything from the 3 modules above |
| `pytorch_models.py` | `HARMLP`, `HARCNN` model definitions + `PyTorchTrainer` (handles training loop, exports weights via `export_mlp_weights()`/`export_cnn_weights()` in sklearn-compatible format) |
| `data_processing.py` | `parse_csv()`, `clean_data()`, `low_pass_filter()`, `detect_sensor_columns()` |
| `device_reader.py` | Serial device communication for Device Test tab |

## PyTorch Integration Path

`EdgeMLModel(model_type='pytorch_mlp')` → creates `PyTorchTrainer` with `HARMLP` → trains with Adam + CrossEntropy + LR scheduler → `export_mlp_weights()` converts to sklearn-compatible dict (coefs_, intercepts_) → `NeuralNetworkCodeGenerator` reads these weights → generates C++ forward pass. Same path for `pytorch_cnn` → `HARCNN` → `export_cnn_weights()` → `CNNCodeGenerator`.

## Code Generation Hierarchy

```
BaseCodeGenerator (ABC)  ← base_generator.py (~1580 lines, feature extraction + scaler + template methods)
├── NeuralNetworkCodeGenerator   ← Also handles pytorch_mlp
├── RandomForestCodeGenerator
├── SVMCodeGenerator
├── CNNCodeGenerator             ← Handles pytorch_cnn
├── ARMCortexMCodeGenerator      ← Platform-specific wrapper
├── MicroPythonCodeGenerator     ← Separate feature extraction (must stay in sync with base)
└── ZephyrCodeGenerator          ← Platform-specific wrapper
```

`CodeGeneratorFactory.create_generator()` routes by model type and platform. Platform generators (arm_cortex_m, micropython, zephyr) take priority over model-type generators.

**Optimization modes** affect `feature_precision` (decimal places for weights/scaler):
- `accuracy` → 6 decimals, debug enabled
- `balanced` → 3 decimals (default)
- `speed` → 2-3 decimals, buffer optimization
- `power` → 3-4 decimals, buffer optimization

**Note**: `deployment/deployment_config.toml` exists with platform specs (memory, compile flags) but is **NOT loaded by any Python code** — currently aspirational/documentation only. Platform values are hardcoded in generators.

## Naming Conventions

- **Component IDs**: kebab-case, descriptive — `{noun}-{noun}-{role}` (e.g., `model-type-selector`, `start-training-btn`, `working-directory-store`)
- **Layout files**: `layouts/{tab_name}.py` → exports `layout`
- **Callback files**: `callbacks/{tab_name}_callbacks.py` → exports `register_callbacks(app)`
- **Generated code**: `persistent_data/generated/{model_type}_models/{model_name}/`

## Developer Workflows

```bash
# Run the app
python app.py                    # Serves at http://127.0.0.1:8050

# Run tests
pytest tests/test_main.py -v     # pytest, single test file

# Venv activation (Windows)
.\venv\Scripts\Activate.ps1

# Compile thesis (from academic-paper-vietnamese/)
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Known Pitfalls

- **Window padding**: Windows shorter than target size use **edge-value replication** (NOT zero-padding). See `callbacks/feature_engineering_callbacks.py`. Zero-padding creates physically impossible `acc_magnitude=0` values. This was a critical deployment bug — see `academic-paper-vietnamese/TECHNICAL_FINDINGS.md` §1.
- **MicroPython generator** has its OWN feature extraction code (not inherited from base). Any formula fix in `base_generator.py` must also be applied in `micropython_generator.py`.
- **Stale training files**: The FE callback uses `_get_fe_train_files()` to find only training files matching the current FE session. Loading ALL `*_train.csv` causes feature-count union bugs.
- **Kurtosis/skewness formulas**: Must use population std (not sample std) for z-score normalization to match pandas. Already fixed in all generators — see `TECHNICAL_FINDINGS.md` §2.
- **Backward-compat paths**: Config path helpers check old flat `persistent_data/` location before new subdirectory. Don't remove this fallback.
- **Inline styles**: Nearly all CSS is inline Python dicts. Only `assets/sticky.css` exists for the sticky header. No external CSS framework besides dash-bootstrap-components.
