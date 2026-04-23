# HAR Edge Deployment Framework — Claude Code Instructions

## Project Identity

Master's thesis project: **Human Activity Recognition Edge Deployment Framework**

- Student: Nguyen Truong Minh Hoang (MSSV: 2270757)
- Supervisor: Dr. Le Trong Nhan
- Thesis report language: Vietnamese (`academic-paper-vietnamese/`)

## Architecture

Dash 2.14+ single-page app with 6 sequential tabs: Data → Preprocess → Feature Engineering → Train → Code Gen → Device Test.

```
app.py                  ← Entry point
layouts/                ← One file per tab, exports `layout`
callbacks/              ← One file per tab, exports `register_callbacks(app)`
utils/                  ← Core logic (feature extraction, ML models, training)
deployment/             ← C++ code generators: BaseCodeGenerator → subclasses
config/config.py        ← Central constants
persistent_data/        ← Runtime data: datasets/, windows/, training/, models/, generated/
academic-paper-vietnamese/ ← Thesis LaTeX source
```

## Mandatory Rules

1. **Training-deployment parity**: Any statistical formula change in Python (`utils/feature_extraction.py`) MUST be mirrored in C++ (`deployment/base_generator.py`) and vice versa. This is the #1 invariant.
2. **MicroPython sync**: `deployment/micropython_generator.py` has its own feature extraction — any formula fix in `base_generator.py` must also go there.
3. **Config imports**: Always `from config.config import PERSISTENT_DIR, SENSOR_COLUMNS, MODELS_DIR`
4. **Model serialization**: `joblib` only. Dict keys: `model`, `scaler`, `label_encoder`, `feature_names`, `model_type`, `model_params`, `performance_metrics`.
5. **Callback pattern**: Each `callbacks/*.py` defines `register_callbacks(app)` — all decorators inside that closure.
6. **Feature order**: Python trains on alphabetically-sorted columns. C++ extracts in fixed order. `CodeGeneratorFactory.reorder_model_parameters()` handles reordering at generation time.
7. **No zero-padding**: Window padding uses edge-value replication, never zeros (creates impossible `acc_magnitude=0`).
8. **Kurtosis/skewness**: Must use population std (not sample std) for z-scores to match pandas.

## Documentation Policy

When making any code change, ask: *"Is this a novel finding or differentiating feature vs Edge Impulse/SensiML?"*

- If yes → document in `academic-paper-vietnamese/TECHNICAL_FINDINGS.md`
- Also note in `academic-paper-vietnamese/THESIS_REPORT_INSTRUCTIONS.md`

## Key Commands

```bash
uv run python app.py             # Run app at http://127.0.0.1:8050
uv run pytest tests/test_main.py -v  # Run tests
uv sync                          # Install/sync dependencies
uv sync --extra dev              # + dev tools
.venv\Scripts\Activate.ps1       # Activate venv (Windows)
```

## Feature Types (6 modes)

| Mode | Count | Description |
|------|-------|-------------|
| `orientation_invariant_time_only` | 33 | acc_mag + gyro_mag (15 stats each) + jerk (3 stats: mean/std/max) |
| `orientation_invariant` | 53 | above + DFT magnitude features (10 per signal) |
| `time_domain` | 90 | 15 stats × 6 axes |
| `all` | 156 | time + frequency per axis (11 per axis) |
| `frequency_domain` | 66 | frequency features (11 per axis) |
| `raw` | 6 | raw sensor values |

## Sensor Columns

Canonical: `['aX','aY','aZ','gX','gY','gZ']`

## Git Worktree Convention

This project uses git worktrees for parallel work:

- `main` — stable, tested code
- Feature branches get their own worktree in `../GUI_app-{branch-name}/`
- Each Claude Code session works in its own worktree to avoid conflicts
- Use `.claude/commands/worktree-*.md` commands to manage worktrees

## Agent Routing

Use specialized agents via `.claude/commands/`:

- `/thesis` — Thesis writing and LaTeX tasks
- `/codegen` — Code generator modifications (parity-critical)
- `/frontend` — Dash layout/callback work
- `/deploy` — Device deployment and serial testing
- `/review` — Code review with parity checks
