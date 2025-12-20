from dash import dcc, html
from dash import dash_table

layout = html.Div([
    # Header Section
    html.Div([
        html.H2("🔬 Data Preprocessing & Feature Engineering", style={
            'color': '#2E86AB',
            'text-align': 'center',
            'margin-bottom': '10px',
            'font-weight': 'bold'
        }),
        html.P("Advanced preprocessing pipeline for Human Activity Recognition datasets", style={
            'text-align': 'center',
            'color': '#666',
            'font-style': 'italic',
            'margin-bottom': '30px'
        })
    ], style={'margin-bottom': '30px'}),

    # Hidden stores
    dcc.Store(id='stored-datasets', data={}, storage_type='local'),
    dcc.Store(id='current-windows', data=[], storage_type='session'),

    # Dataset Selection Section
    html.Div([
        html.H3("📂 Dataset Selection", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Choose a dataset to begin preprocessing", style={
               'color': '#666', 'margin-bottom': '15px'}),

        html.Div([
            html.Label("Select Dataset:", style={
                       'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
            dcc.Dropdown(
                id='dataset-selector_',
                options=[],
                placeholder="Choose a dataset for preprocessing",
                clearable=False,
                style={'margin-bottom': '15px'}
            ),
            # Enhanced Dataset Status Display
            html.Div(id='dataset-status-display', style={
                'margin-top': '15px'
            })
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Data Cleaning & Smoothing Section
    html.Div([
        html.H3("🧹 Data Preprocessing & Signal Processing", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Apply noise reduction, outlier removal, and signal conditioning techniques to raw sensor data",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div([
            html.Div([
                html.H5("📋 Signal Processing Pipeline", style={
                        'color': '#495057', 'margin-bottom': '15px'}),
                html.Ul([
                    html.Li("🗑️ Remove missing values and anomalies"),
                    html.Li("📊 Filter statistical outliers"),
                    html.Li("🌊 Apply low-pass filter (5Hz cutoff)"),
                    html.Li("📈 Savitzky-Golay smoothing (window=5)")
                ], style={'color': '#666', 'line-height': '1.6'})
            ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
                html.Button(
                    "🔧 Process Signal Data",
                    id='clean-smooth-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#17a2b8',
                        'color': 'white',
                        'border': 'none',
                        'padding': '15px 25px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '16px',
                        'width': '100%',
                        'margin-bottom': '15px'
                    }
                ),
                html.Button(
                    "💾 Save Processed Data",
                    id='save-cleaned-smoothed-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#28a745',
                        'color': 'white',
                        'border': 'none',
                        'padding': '15px 25px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '16px',
                        'width': '100%'
                    }
                )
            ], style={'width': '35%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'})
        ]),

        # Preprocessed data visualization
        html.Div([
            dcc.Graph(
                id='preprocessed-graph',
                style={'height': '450px'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'preprocessed_data',
                        'height': 450,
                        'width': 1000,
                        'scale': 1
                    }
                }
            )
        ], style={'margin-top': '20px'})

    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Time Window Configuration Section
    html.Div([
        html.H3("⏱️ Time Window Configuration", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Define sliding windows for activity segmentation and feature extraction",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Time Window Span (milliseconds):", style={
                           'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Input(
                    id='time-window-span-input',
                    type='number',
                    placeholder="Enter window duration",
                    min=100,
                    max=10000,
                    step=100,
                    value=1500,
                    style={
                        'width': '100%',
                        'padding': '12px',
                        'border': '1px solid #ddd',
                        'border-radius': '4px',
                        'font-size': '16px'
                    }
                ),
                html.Div([
                    html.P("💡 Recommended values (based on HAR research):", style={
                           'margin': '10px 0 5px 0', 'font-weight': 'bold', 'color': '#495057'}),
                    html.P("• 1000ms - Very fast transitions",
                           style={'margin': '2px 0', 'color': '#666', 'font-size': '14px'}),
                    html.P("• 1500ms - Optimal for most activities (walking, running, stairs) ✅",
                           style={'margin': '2px 0', 'color': '#28a745', 'font-size': '14px', 'font-weight': 'bold'}),
                    html.P("• 2000ms - Complex motion sequences",
                           style={'margin': '2px 0', 'color': '#666', 'font-size': '14px'})
                ])
            ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
                html.Button(
                    "🎯 Apply Time Windows",
                    id='apply-time-window-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#007bff',
                        'color': 'white',
                        'border': 'none',
                        'padding': '15px 25px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '16px',
                        'width': '100%',
                        'margin-bottom': '20px'
                    }
                ),
                html.Div([
                    html.P("🎛️ Window Controls", style={
                           'font-weight': 'bold', 'margin-bottom': '10px', 'color': '#495057'}),
                    html.P("Use buttons below the graph to add, remove, or reset windows",
                           style={'font-size': '14px', 'color': '#666', 'line-height': '1.4'})
                ])
            ], style={'width': '35%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'})
        ])

    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Interactive Window Selection Section
    html.Div([
        html.H3("🎮 Interactive Window Selection", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Drag windows to optimal positions and extract segments for model training",
               style={'color': '#666', 'margin-bottom': '20px'}),

        # Interactive graph with enhanced user experience
        dcc.Graph(
            id='interactive-sample-graph',
            style={
                'height': '800px',  # Increased height for better visibility
                'width': '100%',    # Full width utilization
                'border': '2px solid #e9ecef',
                'border-radius': '8px',
                'background-color': '#fafafa'
            },
            config={
                'editable': True,
                'displayModeBar': True,
                'displaylogo': False,
                'modeBarButtonsToAdd': [
                    'drawrect',
                    'eraseshape',
                    'pan2d',
                    'zoom2d',
                    'zoomIn2d',
                    'zoomOut2d',
                    'autoScale2d',
                    'resetScale2d'
                ],
                'modeBarButtonsToRemove': ['lasso2d', 'select2d'],
                'doubleClick': 'reset+autosize',
                'scrollZoom': True,
                'toImageButtonOptions': {
                    'format': 'png',
                    'filename': 'interactive_time_windows',
                    'height': 800,
                    'width': 1400,
                    'scale': 2
                }
            }
        ),

        # Enhanced window control section with better usability
        html.Div([
            # Help text for better user guidance
            html.Div([
                html.H4("🎮 Horizontal Time-Series Navigation Guide", style={
                    'color': '#2E86AB',
                    'margin-bottom': '15px',
                    'text-align': 'center'
                }),
                html.Div([
                    html.Div([
                        html.P("⬅️➡️ Horizontal Movement:", style={
                               'font-weight': 'bold', 'margin-bottom': '5px', 'color': '#495057'}),
                        html.P("• Range slider: Navigate time series", style={
                               'margin': '2px 0', 'font-size': '14px'}),
                        html.P("• Pan: Drag background horizontally", style={
                               'margin': '2px 0', 'font-size': '14px'}),
                        html.P("• Zoom: Mouse wheel for precision",
                               style={'margin': '2px 0', 'font-size': '14px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'vertical-align': 'top'}),

                    html.Div([
                        html.P("🎯 Window Control:", style={
                               'font-weight': 'bold', 'margin-bottom': '5px', 'color': '#495057'}),
                        html.P("• Drag: Move windows horizontally only", style={
                               'margin': '2px 0', 'font-size': '14px'}),
                        html.P("• Constrained: Vertical position fixed", style={
                               'margin': '2px 0', 'font-size': '14px'}),
                        html.P("• Time-focused: Align with data features", style={
                               'margin': '2px 0', 'font-size': '14px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '2%'}),

                    html.Div([
                        html.P("⚡ Time-Series Tips:", style={
                               'font-weight': 'bold', 'margin-bottom': '5px', 'color': '#495057'}),
                        html.P("• Scroll to find patterns", style={
                               'margin': '2px 0', 'font-size': '14px'}),
                        html.P("• Zoom for microsecond precision", style={
                               'margin': '2px 0', 'font-size': '14px'}),
                        html.P("• Use grid lines for alignment",
                               style={'margin': '2px 0', 'font-size': '14px'})
                    ], style={'width': '32%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '2%'})
                ])
            ], style={
                'background-color': '#f8f9fa',
                'padding': '15px',
                'border-radius': '8px',
                'border': '1px solid #dee2e6',
                'margin-bottom': '20px'
            }),

            html.Div([
                html.Div([
                    html.H5("🎛️ Window Management", style={
                            'color': '#495057', 'margin-bottom': '15px'}),
                    html.Div([
                        html.Button(
                            "➕ Add Window",
                            id='add-window-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#28a745',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin-right': '10px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        ),
                        html.Button(
                            "➖ Remove Last",
                            id='remove-window-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#dc3545',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin-right': '10px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        ),
                        html.Button(
                            "🔄 Reset Windows",
                            id='reset-windows-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#6c757d',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'margin-right': '10px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        ),
                        html.Button(
                            "📂 Load Previous",
                            id='load-previous-windows-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#17a2b8',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        )])
                ])
            ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
                html.Button(
                    "✂️ Split Selected Windows",
                    id='split-selected-windows-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#ff6b35',
                        'color': 'white',
                        'border': 'none',
                        'padding': '15px 30px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '18px',
                        'width': '100%',
                        'box-shadow': '0 4px 8px rgba(0,0,0,0.2)'
                    }
                )
            ], style={'width': '35%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'})
        ], style={'margin': '20px 0'}),

        # Sliding Window Generation Section
        html.Div([
            html.H5("🔄 Automatic Sliding Window Generation", style={
                'color': '#495057', 
                'margin-bottom': '15px',
                'margin-top': '25px'
            }),
            html.P("Generate overlapping windows from selected regions to increase training samples", 
                   style={'color': '#666', 'margin-bottom': '15px', 'font-size': '14px'}),
            
            html.Div([
                html.Div([
                    html.Label("Overlap Percentage:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Slider(
                        id='overlap-percentage-slider',
                        min=0,
                        max=90,
                        step=5,
                        value=50,
                        marks={0: '0%', 25: '25%', 50: '50%', 75: '75%', 90: '90%'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div(id='overlap-info', style={'margin-top': '10px', 'font-size': '13px', 'color': '#666'})
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                html.Div([
                    html.Label("Quality Threshold:", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
                    dcc.Slider(
                        id='quality-threshold-slider',
                        min=0.3,
                        max=1.0,
                        step=0.05,
                        value=0.7,
                        marks={0.3: '0.3', 0.5: '0.5', 0.7: '0.7', 0.9: '0.9', 1.0: '1.0'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                    html.Div("Higher = stricter quality filtering", 
                             style={'margin-top': '10px', 'font-size': '12px', 'color': '#999', 'font-style': 'italic'})
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
            ], style={'margin-bottom': '20px'}),
            
            html.Div([
                html.Button(
                    "🚀 Generate Sliding Windows",
                    id='generate-sliding-windows-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#007bff',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 30px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '16px',
                        'box-shadow': '0 3px 6px rgba(0,0,0,0.15)',
                        'margin-right': '15px'
                    }
                ),
                html.Button(
                    "💾 Save Generated Windows",
                    id='save-sliding-windows-btn',
                    n_clicks=0,
                    disabled=True,
                    style={
                        'background-color': '#28a745',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 30px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'font-size': '16px',
                        'box-shadow': '0 3px 6px rgba(0,0,0,0.15)'
                    }
                )
            ], style={'margin-bottom': '20px'}),
            
            # Preview area
            html.Div(id='sliding-windows-preview'),
            
            # Store for generated windows
            dcc.Store(id='sliding-windows-data')
        ], style={
            'background-color': '#f8f9fa',
            'padding': '20px',
            'border-radius': '8px',
            'border': '2px dashed #007bff',
            'margin': '20px 0'
        }),

        # Split results graph
        html.Div([
            html.H5("🔍 Window Split Results", style={
                'color': '#495057', 
                'margin-bottom': '15px',
                'margin-top': '25px'
            }),
            dcc.Graph(
                id='split-samples-graph',
                style={'height': '500px'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'split_windows',
                        'height': 500,
                        'width': 1200,
                        'scale': 1
                    }
                }
            )
        ], style={'margin-top': '30px', 'margin-bottom': '30px'})

    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Split Dataset Management Section
    html.Div([
        html.H3("📊 Split Window Management", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("View, analyze, and manage individual split windows from the time-series data",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Select Split Window:", style={
                           'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='split-dataset-selector',
                    options=[],
                    placeholder="No split windows available - split some windows first",
                    clearable=False,
                    style={'margin-bottom': '15px'}
                )
            ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
                html.Button(
                    "🗑️ Delete Selected Window",
                    id='delete-split-window-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#dc3545',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 20px',
                        'border-radius': '4px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%',
                        'margin-bottom': '10px'
                    }
                ),
                html.Button(
                    "🧹 Clean All Windows",
                    id='clean-generated-data-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#ffc107',
                        'color': '#212529',
                        'border': 'none',
                        'padding': '12px 20px',
                        'border-radius': '4px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%'
                    }
                )
            ], style={'width': '35%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'})
        ]),

        # Info panel for selected split window
        html.Div(id='split-window-info', style={
            'background-color': '#f8f9fa',
            'padding': '15px',
            'border-radius': '5px',
            'border-left': '4px solid #2E86AB',
            'margin': '20px 0'
        }),

        # Graph to show selected split window
        dcc.Graph(id='selected-split-graph', style={'margin-bottom': '20px'}),

        # Statistics table for the selected window
        html.Div([
            html.H5("📈 Window Statistics", style={
                    'color': '#495057', 'margin-bottom': '15px'}),
            dash_table.DataTable(
                id='split-window-stats-table',
                columns=[],
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
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # ML Model Training Preprocessing Section
    html.Div([
        html.H3("🤖 Feature Engineering & Model Preparation", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Advanced feature engineering pipeline for machine learning model training with normalization and data splitting",
               style={'color': '#666', 'margin-bottom': '15px'}),
        
        # Workflow instructions
        html.Div([
            html.H5("🚀 Feature Engineering Workflow", style={
                'color': '#28a745', 'margin-bottom': '10px'}),
            html.Ol([
                html.Li("📋 Select training windows (use 'Select All' for all available windows)"),
                html.Li("🛠️ Click 'Engineer Features' to prepare your data with normalization"),
                html.Li("📊 Click 'Perform Train-Test Split' to divide engineered data"),
                html.Li("💾 Click 'Save Training Data' to store ML-ready datasets")
            ], style={'margin': '0', 'padding-left': '20px', 'color': '#495057'})
        ], style={
            'background-color': '#f8f9fa',
            'padding': '15px',
            'border-radius': '6px',
            'margin-bottom': '25px',
            'border-left': '4px solid #28a745'
        }),

        # Dataset selection for training
        html.Div([
            html.H5("📁 Training Dataset Selection", style={
                    'color': '#495057', 'margin-bottom': '15px'}),
            html.Div([
                html.Div([
                    html.Label("Select Training Windows:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='training-dataset-selector',
                        options=[],
                        placeholder="Choose split windows to use for training",
                        multi=True,
                        style={'margin-bottom': '10px'}
                    )
                ], style={'width': '75%', 'display': 'inline-block', 'vertical-align': 'top'}),
                
                html.Div([
                    html.Button(
                        "📋 Select All",
                        id='select-all-training-windows-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#17a2b8',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 15px',
                            'border-radius': '4px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'width': '100%',
                            'margin-bottom': '5px'
                        }
                    ),
                    html.Button(
                        "🗑️ Clear All",
                        id='clear-all-training-windows-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#6c757d',
                            'color': 'white',
                            'border': 'none',
                            'padding': '10px 15px',
                            'border-radius': '4px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'width': '100%'
                        }
                    )
                ], style={'width': '22%', 'display': 'inline-block', 'margin-left': '3%', 'vertical-align': 'top'}),
                
                html.Div(id='training-dataset-info', style={
                    'background-color': '#e3f2fd',
                    'padding': '15px',
                    'border-radius': '6px',
                    'margin': '15px 0 20px 0',
                    'border-left': '4px solid #2196f3'
                })
            ])
        ]),

        # Feature engineering section
        html.Div([
            html.H5("🔧 Feature Engineering Configuration", style={
                    'color': '#495057', 'margin-bottom': '15px'}),

            html.Div([
                html.Div([
                    html.Label("Normalization Method:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='normalization-method',
                        options=[
                            {'label': '📏 Min-Max Scaling (0-1)',
                             'value': 'minmax'},
                            {'label': '📊 Standard Scaling (Z-score)',
                             'value': 'standard'},
                            {'label': '🔄 Robust Scaling', 'value': 'robust'},
                            {'label': '❌ No Normalization', 'value': 'none'}
                        ],
                        value='standard',
                        placeholder="Select normalization method",
                        style={'margin-bottom': '15px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("Feature Selection:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='feature-selection-method',
                        options=[
                            {'label': '🎯 All Features (138: Time + Frequency)',
                             'value': 'all'},
                            {'label': '📈 Raw Axes Only (6 sensors)',
                                'value': 'statistical'},
                            {'label': '🌊 Time-Domain Only (90 features)',
                                'value': 'time_domain'},
                            {'label': '📊 Custom Selection', 'value': 'custom'}
                        ],
                        value='all',
                        placeholder="Select features to include",
                        style={'margin-bottom': '15px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
            ])
        ], style={'margin-bottom': '25px'}),

        # Train-validation-test split configuration
        html.Div([
            html.H5("🎲 Train-Validation-Test Split Configuration",
                    style={'color': '#495057', 'margin-bottom': '15px'}),
            
            html.Div([
                html.P("Configure data split ratios. Set validation to 0% to use cross-validation instead.",
                       style={'color': '#666', 'font-size': '13px', 'margin-bottom': '15px', 'font-style': 'italic'}),
            ]),

            html.Div([
                html.Div([
                    html.Label("🎓 Training Set:",
                               style={'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id='train-split',
                        min=0.4, max=0.8, step=0.05,
                        marks={i/10: f"{i*10}%" for i in range(4, 9)},
                        value=0.6,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'width': '32%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("🔍 Validation Set:",
                               style={'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Slider(
                        id='val-split',
                        min=0.0, max=0.3, step=0.05,
                        marks={0: '0% (CV)', **{i/10: f"{i*10}%" for i in range(1, 4)}},
                        value=0.2,
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),
                ], style={'width': '32%', 'display': 'inline-block', 'margin-left': '2%', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("🧪 Test Set:",
                               style={'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    html.Div(
                        id='test-split-display',
                        style={
                            'padding': '10px',
                            'background-color': '#f8f9fa',
                            'border-radius': '4px',
                            'border': '1px solid #dee2e6',
                            'text-align': 'center',
                            'font-size': '18px',
                            'font-weight': 'bold',
                            'color': '#495057'
                        }
                    ),
                ], style={'width': '32%', 'display': 'inline-block', 'margin-left': '2%', 'vertical-align': 'top'})
            ]),
            
            html.Div(
                id='split-ratio-info', 
                style={'margin-top': '15px', 'font-size': '14px', 'color': '#666', 'padding': '10px', 'background-color': '#e3f2fd', 'border-radius': '4px'}
            ),
            
            html.Div([

                html.Div([
                    html.Label("Random State:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Input(
                        id='random-state-input',
                        type='number',
                        value=42,
                        min=0,
                        max=9999,
                        placeholder="Seed for reproducibility",
                        style={
                            'width': '100%',
                            'padding': '8px',
                            'border': '1px solid #ddd',
                            'border-radius': '4px'
                        }
                    )
                ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'})
            ])
        ], style={'margin-bottom': '25px'}),

        # Action buttons
        html.Div([
            html.Button(
                "⚙️ Engineer Features",
                id='preprocess-for-training-btn',
                n_clicks=0,
                style={
                    'background-color': '#4CAF50',
                    'color': 'white',
                    'border': 'none',
                    'padding': '15px 25px',
                    'border-radius': '6px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '16px',
                    'margin-right': '15px'
                }
            ),
            html.Button(
                "📊 Perform Train-Val-Test Split",
                id='train-test-split-btn',
                n_clicks=0,
                style={
                    'background-color': '#2196F3',
                    'color': 'white',
                    'border': 'none',
                    'padding': '15px 25px',
                    'border-radius': '6px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '16px',
                    'margin-right': '15px'
                }
            ),
            html.Button(
                "💾 Save Training Data",
                id='save-preprocessed-training-btn',
                n_clicks=0,
                style={
                    'background-color': '#FF9800',
                    'color': 'white',
                    'border': 'none',
                    'padding': '15px 25px',
                    'border-radius': '6px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '16px',
                    'margin-right': '15px'
                }
            ),
            html.Button(
                "🧹 Clear Training Data",
                id='clear-training-data-btn',
                n_clicks=0,
                style={
                    'background-color': '#dc3545',
                    'color': 'white',
                    'border': 'none',
                    'padding': '15px 25px',
                    'border-radius': '6px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '16px'
                }
            )
        ], style={'margin': '20px 0', 'text-align': 'center'}),

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
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    })

], style={
    'font-family': 'Arial, sans-serif',
    'margin': '0 auto',
    'max-width': '1400px',
    'padding': '20px',
    'background-color': '#f8f9fa'
})
