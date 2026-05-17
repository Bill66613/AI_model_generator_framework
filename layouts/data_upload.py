from dash import dcc, html, dash_table

layout = html.Div([
    # Header Section
    html.Div([
        html.H2("📊 Sensor Data Management", style={
            'color': '#2E86AB',
            'text-align': 'center',
            'margin-bottom': '10px',
            'font-weight': 'bold'
        }),
        html.P("Upload, manage, and preview your HAR sensor datasets", style={
            'text-align': 'center',
            'color': '#666',
            'font-style': 'italic',
            'margin-bottom': '30px'
        })
    ], style={'margin-bottom': '30px'}),

    # Working Directory Section
    html.Div([
        html.H3("📂 Working Directory", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Select or create a working directory to organize your datasets",
               style={'color': '#666', 'margin-bottom': '15px'}),

        html.Div([
            html.Div([
                html.Label("Current Directory:", style={
                           'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                html.Div(id='current-working-dir-display', children="Not set", style={
                    'padding': '12px',
                    'background': '#e3f2fd',
                    'border-radius': '6px',
                    'border-left': '4px solid #2196f3',
                    'font-family': 'monospace',
                    'font-size': '13px',
                    'margin-bottom': '8px'
                }),
                html.Div([
                    html.Strong("💡 Portability tip: "),
                    "Use a ",
                    html.Strong("relative path"),
                    " (e.g. ",
                    html.Code("persistent_data"),
                    " or ",
                    html.Code("my_project/data"),
                    ") so the project stays portable when moved to another machine. "
                    "Relative paths are resolved from the application root folder. "
                    "Absolute paths work too but break if you move the project.",
                ], style={
                    'font-size': '12px', 'color': '#856404',
                    'background': '#fff3cd', 'padding': '8px 12px',
                    'border-radius': '5px', 'border-left': '3px solid #ffc107',
                    'margin-bottom': '15px'
                }),
            ]),

            html.Div([
                html.Div([
                    html.Label("Directory Path:", style={
                               'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
                    html.Div([
                        dcc.Input(
                            id='directory-path-input',
                            type='text',
                            value='',
                            placeholder='Absolute: D:\\data  OR  relative to app root: my_data  (default: persistent_data)',
                            style={
                                'flex': '1',
                                'padding': '8px',
                                'border': '1px solid #ddd',
                                'border-right': 'none',
                                'border-radius': '6px 0 0 6px',
                                'font-size': '14px',
                                'box-sizing': 'border-box',
                                'min-width': '0',
                            }
                        ),
                        html.Button(
                            "📁 Browse",
                            id='browse-working-dir-btn',
                            n_clicks=0,
                            title='Open folder picker dialog',
                            style={
                                'flex': '0 0 auto',
                                'padding': '7px 16px',
                                'background-color': '#6c757d',
                                'color': 'white',
                                'border': '1px solid #6c757d',
                                'border-radius': '0 6px 6px 0',
                                'cursor': 'pointer',
                                'font-weight': 'bold',
                                'font-size': '13px',
                                'white-space': 'nowrap',
                            }
                        ),
                    ], style={'display': 'flex', 'align-items': 'stretch', 'width': '100%'}),
                ], style={'margin-bottom': '15px'}),

                html.Div([
                    html.Button(
                        "✅ Apply Directory",
                        id='select-working-dir-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#2196f3',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'margin-right': '10px'
                        }
                    ),
                    html.Button(
                        "🔄 Use Default (persistent_data)",
                        id='use-default-dir-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#4caf50',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'margin-right': '10px'
                        }
                    ),
                    html.Button(
                        "🔧 Migrate Old Files",
                        id='migrate-files-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#ff9800',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 20px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold'
                        }
                    )
                ], style={'margin-bottom': '15px'})
            ]),

            html.Div(id='working-dir-status', style={'margin-top': '10px'})
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Upload Section
    html.Div([
        html.H3("📁 Upload Sensor Data", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Select one or more CSV files containing sensor data (accelerometer, gyroscope, etc.)",
               style={'color': '#666', 'margin-bottom': '15px'}),

        dcc.Upload(
            id='upload-data',
            children=html.Div([
                html.Button(
                    "📁 Upload Files",
                    style={
                        'width': '200px',
                        'height': '45px',
                        'background-color': '#4CAF50',
                        'color': 'white',
                        'border': 'none',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '16px',
                        'transition': 'all 0.3s ease'
                    }
                ),
                html.Div("Drag and drop files here or click to browse", style={
                    'margin-top': '10px',
                    'color': '#999',
                    'font-size': '14px',
                    'text-align': 'center'
                })
            ], style={'text-align': 'center'}),
            multiple=True,
            style={
                'width': '100%',
                'height': '120px',
                'border': '2px dashed #ccc',
                'border-radius': '8px',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'margin': '15px 0',
                'background-color': '#fafafa',
                'transition': 'border-color 0.3s ease'
            }
        ),

        html.Div(id='upload-output', style={
            'margin-top': '15px',
            'padding': '10px',
            'border-radius': '4px',
            'min-height': '20px'
        })
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Dataset Management Section
    html.Div([
        html.H3("🗂️ Dataset Management", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),

        # Dataset Selection Row
        html.Div([
            html.Div([
                html.Label("Select Dataset:", style={
                           'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='dataset-selector',
                    options=[],
                    placeholder="Choose a dataset to manage",
                    style={'margin-bottom': '15px'}
                )
            ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
                html.Label(" ", style={'display': 'block',
                           'margin-bottom': '8px'}),  # Spacer
                html.Button(
                    "🗑️ Delete Selected",
                    id='delete-dataset-btn',
                    style={
                        'background-color': '#dc3545',
                        'color': 'white',
                        'border': 'none',
                        'padding': '10px 15px',
                        'border-radius': '4px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%'
                    }
                )
            ], style={'width': '28%', 'display': 'inline-block', 'margin-left': '2%'})
        ]),

        # Dataset Information Grid
        html.Div([
            # Left Column - Labels and Properties
            html.Div([
                html.H5("📋 Dataset Properties", style={
                        'color': '#2E86AB', 'margin-bottom': '15px'}),

                html.Div([
                    html.Label("Current Label:", style={
                               'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
                    dcc.Textarea(
                        id='assigned-label',
                        readOnly=True,
                        placeholder="No label assigned",
                        style={
                            'width': '100%',
                            'height': '40px',
                            'padding': '8px',
                            'border': '1px solid #ddd',
                            'border-radius': '4px',
                            'background-color': '#f8f9fa',
                            'resize': 'none',
                            'margin-bottom': '15px'
                        }
                    )
                ]),

                html.Div([
                    html.Label("Assign New Label:", style={
                               'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
                    html.Div([
                        dcc.Input(
                            id='dataset-label',
                            type='text',
                            placeholder="Enter activity label (e.g., walking, running)",
                            style={
                                'width': '70%',
                                'padding': '8px',
                                'border': '1px solid #ddd',
                                'border-radius': '4px',
                                'display': 'inline-block'
                            }
                        ),
                        html.Button(
                            "💾 Save",
                            id='save-label-btn',
                            style={
                                'width': '28%',
                                'margin-left': '2%',
                                'background-color': '#007bff',
                                'color': 'white',
                                'border': 'none',
                                'padding': '8px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        )
                    ])
                ])
            ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

            # Right Column - Sampling Rate
            html.Div([
                html.H5("⏱️ Sampling Configuration", style={
                        'color': '#2E86AB', 'margin-bottom': '15px'}),

                html.Div([
                    html.Label("Sampling Rate (Hz):", style={
                               'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
                    html.Div([
                        dcc.Input(
                            id='sampling-rate-input',
                            type='number',
                            placeholder="Enter sampling rate",
                            value=100,
                            min=1,
                            max=10000,
                            step=1,
                            style={
                                'width': '70%',
                                'padding': '8px',
                                'border': '1px solid #ddd',
                                'border-radius': '4px',
                                'display': 'inline-block'
                            }
                        ),
                        html.Button(
                            "💾 Save",
                            id='save-sampling-rate-btn',
                            style={
                                'width': '28%',
                                'margin-left': '2%',
                                'background-color': '#28a745',
                                'color': 'white',
                                'border': 'none',
                                'padding': '8px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        )
                    ])
                ]),

                html.Div([
                    html.P("💡 Common rates: 50Hz (basic), 100Hz (standard), 200Hz (high-precision)",
                           style={
                               'font-size': '12px',
                               'color': '#666',
                               'margin-top': '10px',
                               'font-style': 'italic'
                           })
                ])
            ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
        ], style={'margin': '20px 0'}),

        # Dataset Information Table
        html.Div([
            html.H5("📊 Dataset Information", style={
                    'color': '#2E86AB', 'margin-bottom': '15px'}),
            html.Div(id='dataset-info-table', style={'margin-bottom': '20px'})
        ]),

        # Action Buttons
        html.Div([
            html.Button(
                "🧹 Clear All Data",
                id='clear-data-btn',
                style={
                    'background-color': '#ffc107',
                    'color': '#212529',
                    'border': 'none',
                    'padding': '12px 24px',
                    'border-radius': '6px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '14px',
                    'margin-right': '10px'
                }
            ),
            html.Button(
                "📋 Export Metadata",
                id='export-metadata-btn',
                style={
                    'background-color': '#17a2b8',
                    'color': 'white',
                    'border': 'none',
                    'padding': '12px 24px',
                    'border-radius': '6px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '14px'
                }
            )
        ], style={'text-align': 'center', 'margin-top': '20px'})

    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Preview Section
    html.Div([
        html.H3("📈 Data Preview", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Interactive visualization of the selected dataset",
               style={'color': '#666', 'margin-bottom': '15px'}),
        dcc.Graph(
            id='data-preview',
            style={'height': '500px'},
            config={
                'displayModeBar': True,
                'displaylogo': False,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'sensor_data_preview',
                    'height': 500,
                    'width': 1000,
                    'scale': 1
                }
            }
        )
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'border': '1px solid #e9ecef'
    }),

    # Hidden components for data storage
    dcc.Store(id='dataset-metadata', storage_type='session'),
    dcc.Store(id='working-directory-store', storage_type='local')

], style={
    'max-width': '1200px',
    'margin': '0 auto',
    'padding': '20px',
    'background-color': '#f8f9fa',
    'min-height': '100vh',
    'font-family': 'Arial, sans-serif'
})
