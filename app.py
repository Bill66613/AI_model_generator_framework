from dash import Dash, html, dcc, Input, Output, State
import os, json
from layouts.data_upload import layout as data_upload_layout
from layouts.preprocessing import layout as preprocessing_layout
from layouts.feature_engineering import layout as feature_engineering_layout
from layouts.training import layout as training_layout
from layouts.code_generation import layout as code_generation_layout
from layouts.device_test import layout as device_test_layout
from config.config import PERSISTENT_DIR

# Initialize the app FIRST
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For deployment

# ---------------------------------------------------------------------------
# Workflow step definitions
# ---------------------------------------------------------------------------
WORKFLOW_STEPS = [
    {'id': 'step-data',    'label': 'Data',       'icon': '📊'},
    {'id': 'step-preproc', 'label': 'Preprocess',  'icon': '🔬'},
    {'id': 'step-fe',      'label': 'Features',    'icon': '⚙️'},
    {'id': 'step-train',   'label': 'Train',       'icon': '🎯'},
    {'id': 'step-code',    'label': 'Code Gen',    'icon': '🔧'},
    {'id': 'step-device',  'label': 'Device Test', 'icon': '📡'},
]

def _make_step_div(step, idx, total):
    """Create one step element for the progress bar."""
    return html.Div([
        html.Div(step['icon'], className='step-icon', id=f"{step['id']}-icon",
                 style={'font-size': '20px', 'text-align': 'center'}),
        html.Div(step['label'], style={
            'font-size': '11px', 'text-align': 'center', 'margin-top': '2px',
            'white-space': 'nowrap'}),
    ], id=step['id'], style={
        'display': 'inline-block', 'text-align': 'center', 'padding': '6px 14px',
        'border-radius': '8px', 'margin-right': '4px' if idx < total - 1 else '0',
        'background': '#f0f0f0', 'color': '#999', 'min-width': '72px',
        'transition': 'background 0.3s, color 0.3s',
    })

progress_bar = html.Div(
    [_make_step_div(s, i, len(WORKFLOW_STEPS)) for i, s in enumerate(WORKFLOW_STEPS)],
    id='workflow-progress-bar',
    style={
        'display': 'flex', 'justify-content': 'center', 'align-items': 'center',
        'padding': '10px 20px', 'background': '#fafafa',
        'border-bottom': '1px solid #e0e0e0',
    }
)

# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------
app.layout = html.Div([
    # Refresh timer for progress bar (fires every 5 s when tab is visible)
    dcc.Interval(id='progress-interval', interval=5000, n_intervals=0),

    # Progress bar (made sticky via assets/sticky.css)
    progress_bar,

    # Tabs — children contain full layouts so all dcc.Store components exist
    # immediately.  The tab header row is made sticky via assets/sticky.css.
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='📊 Data Management', value='tab-1',
                children=data_upload_layout),
        dcc.Tab(label='🔬 Signal Preprocessing', value='tab-2',
                children=preprocessing_layout),
        dcc.Tab(label='⚙️ Feature Engineering', value='tab-3',
                children=feature_engineering_layout),
        dcc.Tab(label='🎯 Model Training', value='tab-4',
                children=training_layout),
        dcc.Tab(label='🔧 Code Generation', value='tab-5',
                children=code_generation_layout),
        dcc.Tab(label='📡 Device Testing', value='tab-6',
                children=device_test_layout),
    ]),
])

# NOW import and register callbacks (after layout is set)
from callbacks import data_callbacks, preprocessing_callbacks, training_callbacks, feature_engineering_callbacks, code_generation_callbacks, device_test_callbacks

# Explicitly register all callbacks
data_callbacks.register_callbacks(app)
preprocessing_callbacks.register_callbacks(app)
feature_engineering_callbacks.register_callbacks(app)
training_callbacks.register_callbacks(app)
code_generation_callbacks.register_callbacks(app)
device_test_callbacks.register_callbacks(app)


# ---------------------------------------------------------------------------
# Workflow progress callback — checks filesystem for pipeline artifacts
# ---------------------------------------------------------------------------
@app.callback(
    [Output(step['id'], 'style') for step in WORKFLOW_STEPS],
    Input('progress-interval', 'n_intervals'),
    State('working-directory-store', 'data'),
    prevent_initial_call=False,
)
def update_workflow_progress(_n, base_dir):
    """Check which pipeline stages have completed artifacts and colour the progress bar."""
    if not base_dir:
        base_dir = PERSISTENT_DIR

    metadata_file = os.path.join(base_dir, 'metadata.json')
    metadata = {}
    if os.path.exists(metadata_file):
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception:
            pass

    # Determine completed steps based on artifacts
    has_data = len(metadata) > 0
    has_preproc = any(
        'dragged_samples' in v and len(v.get('dragged_samples', [])) > 0
        for v in metadata.values()
    )
    fe_dir = os.path.join(base_dir, 'feature_engineering')
    has_fe = os.path.isdir(fe_dir) and any(
        f.endswith('.json') for f in os.listdir(fe_dir)
    ) if os.path.isdir(fe_dir) else False
    models_dir = os.path.join(base_dir, 'models')
    has_model = os.path.isdir(models_dir) and any(
        f.endswith('.joblib') for f in os.listdir(models_dir)
    ) if os.path.isdir(models_dir) else False
    generated_dir = os.path.join(base_dir, 'generated_code')
    has_code = os.path.isdir(generated_dir) and len(os.listdir(generated_dir)) > 0

    completed = [has_data, has_preproc, has_fe, has_model, has_code, False]

    base_style = {
        'display': 'inline-block', 'text-align': 'center', 'padding': '6px 14px',
        'border-radius': '8px', 'min-width': '72px',
        'transition': 'background 0.3s, color 0.3s',
    }

    styles = []
    for i, done in enumerate(completed):
        s = dict(base_style)
        s['margin-right'] = '4px' if i < len(WORKFLOW_STEPS) - 1 else '0'
        if done:
            s['background'] = '#4CAF50'
            s['color'] = 'white'
        else:
            s['background'] = '#f0f0f0'
            s['color'] = '#999'
        styles.append(s)
    return styles


if __name__ == '__main__':
    app.run(debug=True)
