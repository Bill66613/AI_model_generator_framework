from dash import Dash, html, dcc
from layouts.data_upload import layout as data_upload_layout
from layouts.preprocessing import layout as preprocessing_layout
from layouts.feature_engineering import layout as feature_engineering_layout
from layouts.training import layout as training_layout
from layouts.code_generation import layout as code_generation_layout
from layouts.device_test import layout as device_test_layout

# Initialize the app FIRST
app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server  # For deployment

# Define the app layout with tabs
app.layout = html.Div([
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
                children=device_test_layout)
    ])
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

if __name__ == '__main__':
    app.run(debug=True)
