---
description: "Dash callback registration patterns and state management rules. Use when editing callback functions, adding new callbacks, or debugging UI state flow."
applyTo: "callbacks/**/*.py"
---
# Callback Rules

## Registration Pattern
All `@app.callback` decorators MUST be inside the `register_callbacks(app)` function:
```python
def register_callbacks(app):
    @app.callback(
        Output('id', 'prop'),
        Input('trigger', 'prop'),
        prevent_initial_call=True
    )
    def my_callback(value):
        ...
```

## State Management
- Use `dcc.Store` for cross-callback data
- Use filesystem (`persistent_data/`) for persistent state
- Use `dash.no_update` to skip outputs selectively
- Use `dash.exceptions.PreventUpdate` to cancel entirely

## Component IDs
kebab-case: `{noun}-{noun}-{role}` (e.g., `model-type-selector`, `start-training-btn`)

## Important
- `suppress_callback_exceptions=True` must remain in `app.py`
- Config paths: `from config.config import PERSISTENT_DIR, SENSOR_COLUMNS`
