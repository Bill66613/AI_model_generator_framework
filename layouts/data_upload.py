from dash import dcc, html

layout = html.Div([
    html.H3("Upload Sensor Data"),

    dcc.Upload(
        id='upload-data',
        children=html.Button("Upload Files"),
        multiple=True
    ),

    html.Div(id='upload-output'),

    html.Hr(),

    html.H3("Manage Datasets"),
    dcc.Dropdown(
        id='dataset-selector',
        options=[],  # Populated dynamically
        placeholder="Select a dataset"
    ),

    html.Div([
        html.Label("Dataset Label:"),
        dcc.Textarea(id='assigned-label', readOnly=True,
                     placeholder="No label assigned", style={'height': 15}),
        html.Label("Assign Label:"),
        dcc.Input(id='dataset-label', type='text', placeholder="Enter label"),
        html.Button("Save Label", id='save-label-btn')
    ]),
    html.Button("Clear Data", id='clear-data-btn'),

    html.Hr(),

    html.H3("Sampling Rate"),
    html.Div([
        html.Label("Sampling Rate (Hz):"),
        dcc.Input(
            id='sampling-rate-input',
            type='number',
            placeholder="Enter sampling rate",
            value=100,  # Default value
            min=1,
            step=1
        ),
        html.Button("Save Sampling Rate", id='save-sampling-rate-btn')
    ]),

    html.Hr(),

    html.H3("Preview Data"),
    dcc.Graph(id='data-preview')
])
