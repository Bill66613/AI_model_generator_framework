# TODO.md

## Active / Planned

- [ ] **Step completion indicator** — Add a persistent workflow progress bar or sidebar showing
      which pipeline stages have been completed (data uploaded → preprocessed → features extracted
      → model trained → code generated → device tested).
- [ ] **Reduce dcc.Store payload** — `engineered-dataset-store` currently holds full
      `X_train.tolist()` / `X_test.tolist()` arrays.  Store file paths instead and load on demand
      so the browser isn't shuttling megabytes per callback.
- [ ] **End-to-end integration test** — Automate a headless run through all 6 tabs with sample
      data to catch regressions.
- [ ] **balance_training_data.py** — Either integrate into the Feature Engineering tab or remove
      (currently unused / not imported anywhere).
- [ ] **Deployment generator sampling rate** — `base_generator.py` still has
      `self.sampling_rate = 100` hardcoded; read from model metadata instead.

## Recently Completed

- [x] Centralized sensor column configuration (`config.py`: `SENSOR_COLUMNS`, `ACCEL_COLUMNS`,
      `GYRO_COLUMNS`, `DEFAULT_SAMPLING_RATE`). All callbacks and utils now derive column lists
      from config rather than hardcoding `['aX','aY','aZ','gX','gY','gZ']`.
- [x] Clarified normalization UX — dropdown label and info box now explain that normalization is
      deferred to training and bundled with the saved model.
- [x] Fixed Device Testing tab — auto-discovers latest `.joblib` model, reads
      `_fe_metadata.json` for correct feature method, window size, and sampling rate.
- [x] Fixed `fs` inconsistency — `clean_and_smooth_data()` now reads actual sampling rate from
      metadata instead of using hardcoded `fs=50`.
- [x] Added split ratio validation — execute button is disabled when train + val ≥ 100 %;
      guard clause prevents execution even if UI is bypassed.
- [x] Removed ~470 lines of orphaned legacy preprocessing callbacks (`preprocess_for_training`,
      `perform_enhanced_train_val_test_split`, `clear_training_data`).
- [x] Fixed inference feature mismatch — Device Testing reads `_fe_metadata.json` and passes
      correct `orientation_robust`, `include_per_axis`, `include_frequency` flags.
- [x] Refactored `utils/` — split monolithic `model_training.py` (981 lines) into:
      `feature_extraction.py`, `edge_ml_model.py`, `training_pipeline.py`.
      `model_training.py` remains as a backward-compat re-export shim.

## Architecture (6-Tab Workflow)

1. **Data Management** — Upload / manage CSV sensor datasets
2. **Signal Preprocessing** — Clean, filter, window, drag-select windows
3. **Feature Engineering** — Extract features, normalize, train/val/test split
4. **Model Training** — Train sklearn / PyTorch models, evaluate, save
5. **Code Generation** — Generate C/C++/MicroPython for target MCU
6. **Device Testing** — Serial connection, real-time plots, live inference
