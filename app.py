from dash import Dash, html, dcc
from layouts.data_upload import layout as data_upload_layout
from layouts.preprocessing import layout as preprocessing_layout
from layouts.training import layout as training_layout
from callbacks import data_callbacks, preprocessing_callbacks, training_callbacks

# Initialize the app
app = Dash(__name__, suppress_callback_exceptions=True)

# Define the app layout with tabs
app.layout = html.Div([
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Data Upload', value='tab-1',
                children=data_upload_layout),
        dcc.Tab(label='Preprocessing', value='tab-2',
                children=preprocessing_layout),
        dcc.Tab(label='Training', value='tab-3', children=training_layout)
    ])
])


if __name__ == '__main__':
    app.run(debug=True)
