from dash import dcc, html
from dash import dash_table

layout = html.Div([
    html.H3("Data Preprocessing"),

    dcc.Store(id='stored-datasets', data={}, storage_type='local'),
    dcc.Store(id='current-windows', data=[], storage_type='session'),  # Store current window positions

    # Dropdown to select dataset
    dcc.Dropdown(
        id='dataset-selector_',
        options=[],  # Populated dynamically
        placeholder="Select a dataset",
        clearable=False
    ),

    html.Hr(),

    # Button for cleaning and smoothing data
    html.Div([
        html.Button("Clean & Smooth Data", id='clean-smooth-btn', n_clicks=0),
        html.Button("Save Processed Data",
                    id='save-cleaned-smoothed-btn', n_clicks=0)
    ], style={'margin-bottom': '20px'}),

    html.Div(id='is-preprocessed'),

    # Graph to show preprocessed data
    dcc.Graph(id='preprocessed-graph'),

    html.Hr(),

    # Input for time window span
    html.Div([
        html.Label("Time Window Span (in milliseconds):"),
        dcc.Input(
            id='time-window-span-input',
            type='number',
            placeholder="Enter time window span",
            min=1,
            step=1,
            value=1000  # Default value
        ),
        html.Button("Apply Time Window",
                    id='apply-time-window-btn', n_clicks=0)
    ], style={'margin-bottom': '20px'}),

    # Graph for interactive selection
    dcc.Graph(
        id='interactive-sample-graph',
        config={
            'editable': True,
            'displayModeBar': True,
            'displaylogo': False,
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'time_windows',
                'height': 600,
                'width': 1000,
                'scale': 1
            }
        }
    ),

    # Window control buttons
    html.Div([
        html.Button("➕ Add Window", 
                   id='add-window-btn', 
                   n_clicks=0,
                   style={
                       'background-color': '#28a745',
                       'color': 'white',
                       'border': 'none',
                       'padding': '8px 16px',
                       'margin-right': '10px',
                       'border-radius': '4px',
                       'cursor': 'pointer',
                       'font-weight': 'bold'
                   }),
        html.Button("➖ Remove Last Window", 
                   id='remove-window-btn', 
                   n_clicks=0,
                   style={
                       'background-color': '#dc3545',
                       'color': 'white',
                       'border': 'none',
                       'padding': '8px 16px',
                       'margin-right': '10px',
                       'border-radius': '4px',
                       'cursor': 'pointer',
                       'font-weight': 'bold'
                   }),
        html.Button("🔄 Reset Windows", 
                   id='reset-windows-btn', 
                   n_clicks=0,
                   style={
                       'background-color': '#6c757d',
                       'color': 'white',
                       'border': 'none',
                       'padding': '8px 16px',
                       'border-radius': '4px',
                       'cursor': 'pointer',
                       'font-weight': 'bold'
                   })
    ], style={'margin': '10px 0', 'text-align': 'left'}),

    html.Div([
        html.Button("Split Selected Windows",
                    id='split-selected-windows-btn', n_clicks=0,
                    style={
                        'background-color': '#007bff',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 20px',
                        'border-radius': '4px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '16px'
                    })
    ], style={'margin-top': '20px'}),

    html.Hr(),

    # Graph to show split samples
    dcc.Graph(id='split-samples-graph'),

    html.Hr(),

    # Section for viewing split datasets
    html.Div([
        html.H4("Split Dataset Viewer", style={
                'margin-top': '20px', 'color': '#2E86AB'}),
        html.P("View and analyze individual split windows from the time-series data.",
               style={'color': '#666', 'font-style': 'italic'}),

        html.Div([
            html.Label("Select Split Window:", style={
                       'font-weight': 'bold', 'margin-right': '10px'}),
            dcc.Dropdown(
                id='split-dataset-selector',
                options=[],  # Populated dynamically
                placeholder="No split windows available - split some windows first",
                clearable=False,
                style={'width': '50%', 'display': 'inline-block'}
            ),
            html.Button(
                "🗑️ Delete Selected Window",
                id='delete-split-window-btn',
                n_clicks=0,
                style={
                    'margin-left': '10px',
                    'background-color': '#dc3545',
                    'color': 'white',
                    'border': 'none',
                    'padding': '8px 16px',
                    'border-radius': '4px',
                    'cursor': 'pointer'
                }
            ),
            html.Button(
                "🧹 Clean All Generated Data",
                id='clean-generated-data-btn',
                n_clicks=0,
                style={
                    'margin-left': '10px',
                    'background-color': '#ffc107',
                    'color': '#212529',
                    'border': 'none',
                    'padding': '8px 16px',
                    'border-radius': '4px',
                    'cursor': 'pointer',
                    'font-weight': 'bold'
                }
            )
        ], style={'display': 'flex', 'align-items': 'center', 'margin-bottom': '20px'}),

        # Info panel for selected split window
        html.Div(id='split-window-info', style={
            'background-color': '#f8f9fa',
            'padding': '15px',
            'border-radius': '5px',
            'border-left': '4px solid #2E86AB',
            'margin-bottom': '20px'
        }),

        # Graph to show selected split window
        dcc.Graph(id='selected-split-graph'),

        # Statistics table for the selected window
        html.Div([
            html.H5("Window Statistics", style={
                    'margin-top': '20px', 'color': '#2E86AB'}),
            dash_table.DataTable(
                id='split-window-stats-table',
                columns=[],  # Populated dynamically
                data=[],
                style_cell={
                    'textAlign': 'center',
                    'padding': '10px',
                    'fontFamily': 'Arial'
                },
                style_header={
                    'backgroundColor': '#2E86AB',
                    'color': 'white',
                    'fontWeight': 'bold'
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#f8f9fa'
                    }
                ]
            )
        ])
    ], style={
        'border': '2px solid #e9ecef',
        'border-radius': '8px',
        'padding': '20px',
        'margin': '20px 0',
        'background-color': '#ffffff'
    }),

    html.Hr(),

    # Model Training Preprocessing Section
    html.Div([
        html.H3("📊 Preprocessing for Model Training", style={'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Prepare your split window data for machine learning model training with proper normalization and train-test splitting.",
               style={'color': '#666', 'font-style': 'italic', 'margin-bottom': '25px'}),

        # Dataset selection for training
        html.Div([
            html.Label("📁 Select Training Dataset:", style={'font-weight': 'bold', 'margin-bottom': '10px'}),
            dcc.Dropdown(
                id='training-dataset-selector',
                options=[],  # Populated dynamically
                placeholder="Choose split windows to use for training",
                multi=True,
                style={'margin-bottom': '20px'}
            ),
            html.Div(id='training-dataset-info', style={
                'background-color': '#e3f2fd',
                'padding': '10px',
                'border-radius': '4px',
                'margin-bottom': '20px',
                'border-left': '4px solid #2196f3'
            })
        ]),

        # Feature extraction and preprocessing options
        html.Div([
            html.H5("🔧 Feature Engineering", style={'color': '#2E86AB', 'margin-bottom': '15px'}),
            
            html.Div([
                html.Label("Normalization Method:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.Dropdown(
                    id='normalization-method',
                    options=[
                        {'label': '📏 Min-Max Scaling (0-1)', 'value': 'minmax'},
                        {'label': '📊 Standard Scaling (Z-score)', 'value': 'standard'},
                        {'label': '🔄 Robust Scaling', 'value': 'robust'},
                        {'label': '❌ No Normalization', 'value': 'none'}
                    ],
                    value='standard',
                    placeholder="Select normalization method",
                    style={'width': '300px', 'display': 'inline-block'}
                )
            ], style={'margin-bottom': '15px'}),

            html.Div([
                html.Label("Feature Selection:", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.Dropdown(
                    id='feature-selection-method',
                    options=[
                        {'label': '🎯 All Features', 'value': 'all'},
                        {'label': '📈 Statistical Features', 'value': 'statistical'},
                        {'label': '🌊 Time-Domain Only', 'value': 'time_domain'},
                        {'label': '📊 Custom Selection', 'value': 'custom'}
                    ],
                    value='all',
                    placeholder="Select features to include",
                    style={'width': '300px', 'display': 'inline-block'}
                )
            ], style={'margin-bottom': '20px'})
        ]),

        # Train-test split configuration
        html.Div([
            html.H5("🎲 Train-Test Split Configuration", style={'color': '#2E86AB', 'margin-bottom': '15px'}),
            
            html.Div([
                html.Label("Train-Test Split Ratio:", style={'font-weight': 'bold', 'margin-bottom': '10px'}),
                dcc.Slider(
                    id='train-test-split',
                    min=0.1, max=0.9, step=0.05,
                    marks={i/10: f"{i*10}%" for i in range(1, 10)},
                    value=0.8,
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
                html.Div(id='split-ratio-info', style={'margin-top': '10px', 'font-size': '14px', 'color': '#666'})
            ], style={'margin-bottom': '15px'}),

            html.Div([
                html.Label("Random State (for reproducibility):", style={'font-weight': 'bold', 'margin-right': '10px'}),
                dcc.Input(
                    id='random-state-input',
                    type='number',
                    value=42,
                    min=0,
                    max=9999,
                    style={'width': '100px'}
                )
            ], style={'margin-bottom': '20px'})
        ]),

        # Action buttons
        html.Div([
            html.Button(
                "🚀 Preprocess for Training",
                id='preprocess-for-training-btn',
                n_clicks=0,
                style={
                    'background-color': '#4CAF50',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'border-radius': '4px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '16px',
                    'margin-right': '15px'
                }
            ),
            html.Button(
                "📊 Perform Train-Test Split",
                id='train-test-split-btn',
                n_clicks=0,
                style={
                    'background-color': '#2196F3',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'border-radius': '4px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '16px',
                    'margin-right': '15px'
                }
            ),
            html.Button(
                "💾 Save Preprocessed Data",
                id='save-preprocessed-training-btn',
                n_clicks=0,
                style={
                    'background-color': '#FF9800',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'border-radius': '4px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '16px'
                }
            )
        ], style={'margin': '20px 0'}),

        # Results display
        html.Div(id='preprocessing-results', style={
            'background-color': '#f8f9fa',
            'padding': '15px',
            'border-radius': '5px',
            'border-left': '4px solid #4CAF50',
            'margin': '20px 0'
        }),

        # Visualization of preprocessed data
        dcc.Graph(id='train-test-split-graph'),

        # Data stores for preprocessed data
        dcc.Store(id='preprocessed-training-data', storage_type='session'),
        dcc.Store(id='train-test-data', storage_type='session')

    ], style={
        'border': '2px solid #e9ecef',
        'border-radius': '8px',
        'padding': '25px',
        'margin': '20px 0',
        'background-color': '#ffffff'
    }),

    html.Hr()
])
