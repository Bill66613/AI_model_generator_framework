from dash import dcc, html

layout = html.Div([
    # Header Section
    html.Div([
        html.H2("🤖 Model Training & Edge Deployment", style={
            'color': '#2E86AB',
            'text-align': 'center',
            'margin-bottom': '10px',
            'font-weight': 'bold'
        }),
        html.P("Advanced machine learning pipeline for Human Activity Recognition with edge deployment optimization", style={
            'text-align': 'center',
            'color': '#666',
            'font-style': 'italic',
            'margin-bottom': '30px'
        })
    ], style={'margin-bottom': '30px'}),

    # Model Configuration Section
    html.Div([
        html.H3("⚙️ Model Configuration", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Choose the optimal machine learning algorithm for your deployment requirements",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Select Model Type:", style={
                           'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='model-type-selector',
                    options=[
                        {'label': '🌲 Random Forest (Recommended for Edge)',
                         'value': 'random_forest'},
                        {'label': '🧠 Neural Network (High Accuracy)',
                         'value': 'neural_network'},
                        {'label': '⚡ Support Vector Machine (Fast Training)',
                         'value': 'svm'}
                    ],
                    placeholder="Select a machine learning model",
                    value='random_forest',
                    disabled=True,
                    style={'margin-bottom': '15px'}
                )
            ], style={'width': '100%', 'margin-bottom': '20px'}),

            # Model comparison cards
            html.Div([
                html.Div([
                    html.H5("🌲 Random Forest", style={
                            'color': '#2E8B57', 'margin-bottom': '10px'}),
                    html.P("Best balance of accuracy and edge deployment efficiency", style={
                           'margin-bottom': '8px'}),
                    html.Ul([
                        html.Li("Excellent for embedded systems"),
                        html.Li("Low memory footprint"),
                        html.Li("Fast inference speed"),
                        html.Li("Good interpretability")
                    ], style={'font-size': '14px', 'color': '#666'})
                ], style={
                    'background-color': '#f8fff8',
                    'padding': '15px',
                    'border-radius': '8px',
                    'border-left': '4px solid #2E8B57',
                    'width': '32%',
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin-right': '2%'
                }),

                html.Div([
                    html.H5("🧠 Neural Network", style={
                            'color': '#4682B4', 'margin-bottom': '10px'}),
                    html.P("Highest accuracy but requires more computational resources", style={
                           'margin-bottom': '8px'}),
                    html.Ul([
                        html.Li("Superior pattern recognition"),
                        html.Li("Handles complex features"),
                        html.Li("Requires more memory"),
                        html.Li("Longer training time")
                    ], style={'font-size': '14px', 'color': '#666'})
                ], style={
                    'background-color': '#f8f9ff',
                    'padding': '15px',
                    'border-radius': '8px',
                    'border-left': '4px solid #4682B4',
                    'width': '32%',
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin-right': '2%'
                }),

                html.Div([
                    html.H5("⚡ Support Vector Machine", style={
                            'color': '#FF6347', 'margin-bottom': '10px'}),
                    html.P("Fast training and good performance on small datasets", style={
                           'margin-bottom': '8px'}),
                    html.Ul([
                        html.Li("Quick training process"),
                        html.Li("Effective on small data"),
                        html.Li("Good generalization"),
                        html.Li("Moderate resource usage")
                    ], style={'font-size': '14px', 'color': '#666'})
                ], style={
                    'background-color': '#fff8f8',
                    'padding': '15px',
                    'border-radius': '8px',
                    'border-left': '4px solid #FF6347',
                    'width': '32%',
                    'display': 'inline-block',
                    'vertical-align': 'top'
                })
            ])
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Training Controls Section
    html.Div([
        html.H3("🚀 Training Controls", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Execute training pipeline with advanced optimization techniques",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div([
            html.Div([
                html.Button(
                    "🚀 Start Training",
                    id='start-training-btn',
                    disabled=True,
                    style={
                        'background-color': '#28a745',
                        'border': 'none',
                        'padding': '15px 25px',
                        'font-size': '16px',
                        'border-radius': '6px',
                        'color': 'white',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%',
                        'margin-bottom': '10px'
                    }
                ),
                html.P("Begin model training with selected configuration",
                       style={'font-size': '12px', 'color': '#666', 'text-align': 'center'})
            ], style={'width': '32%', 'display': 'inline-block', 'margin-right': '2%'}),

            html.Div([
                html.Button(
                    "📊 Hyperparameter Optimization",
                    id='optimize-hyperparams-btn',
                    disabled=True,
                    style={
                        'background-color': '#17a2b8',
                        'border': 'none',
                        'padding': '15px 25px',
                        'font-size': '16px',
                        'border-radius': '6px',
                        'color': 'white',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%',
                        'margin-bottom': '10px'
                    }
                ),
                html.P("Automatic parameter tuning for optimal performance",
                       style={'font-size': '12px', 'color': '#666', 'text-align': 'center'})
            ], style={'width': '32%', 'display': 'inline-block', 'margin-right': '2%'}),

            html.Div([
                html.Button(
                    "🔄 Cross-Validation",
                    id='cross-validate-btn',
                    disabled=True,
                    style={
                        'background-color': '#ffc107',
                        'border': 'none',
                        'padding': '15px 25px',
                        'font-size': '16px',
                        'border-radius': '6px',
                        'color': '#212529',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%',
                        'margin-bottom': '10px'
                    }
                ),
                html.P("Validate model robustness with cross-validation",
                       style={'font-size': '12px', 'color': '#666', 'text-align': 'center'})
            ], style={'width': '32%', 'display': 'inline-block'})
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Data Summary Section
    html.Div([
        html.H3("📁 Training Data Summary", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.Div(id='training-data-summary', style={
            'min-height': '80px',
            'background-color': '#f8f9fa',
            'padding': '20px',
            'border-radius': '8px',
            'border': '1px solid #dee2e6'
        }, children=[
            html.P("No training data loaded. Please complete the train-validation-test split in the Preprocessing tab.",
                   style={'text-align': 'center', 'color': '#6c757d', 'font-style': 'italic', 'margin': '0'})
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Training Output Section
    html.Div([
        html.H3("📊 Training Results", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.Div(id='training-output', style={
            'min-height': '100px',
            'background-color': '#f8f9fa',
            'padding': '20px',
            'border-radius': '8px',
            'border': '1px solid #dee2e6'
        }, children=[
            html.P("No training session active. Configure your model and start training to see results.",
                   style={'text-align': 'center', 'color': '#6c757d', 'font-style': 'italic', 'margin': '0'})
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Model Evaluation Section
    html.Div([
        html.H3("📈 Model Evaluation & Analysis", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Analyze trained models with comprehensive performance metrics and visualizations",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div([
            html.Div([
                html.Label("Select Trained Model:", style={
                           'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='trained-model-selector',
                    options=[],  # Populated dynamically
                    placeholder="Choose a trained model for evaluation",
                    style={'margin-bottom': '15px'}
                )
            ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),

            html.Div([
                html.Button(
                    "📈 Detailed Evaluation",
                    id='evaluate-model-btn',
                    style={
                        'background-color': '#6c757d',
                        'border': 'none',
                        'padding': '12px 20px',
                        'font-size': '14px',
                        'border-radius': '5px',
                        'color': 'white',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%',
                        'margin-bottom': '10px'
                    }
                ),
                html.Button(
                    "🔍 Feature Importance",
                    id='feature-importance-btn',
                    style={
                        'background-color': '#6f42c1',
                        'border': 'none',
                        'padding': '12px 20px',
                        'font-size': '14px',
                        'border-radius': '5px',
                        'color': 'white',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%',
                        'margin-bottom': '10px'
                    }
                ),
                html.Button(
                    "🗑️ Remove Model",
                    id='remove-model-btn',
                    style={
                        'background-color': '#dc3545',
                        'border': 'none',
                        'padding': '12px 20px',
                        'font-size': '14px',
                        'border-radius': '5px',
                        'color': 'white',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '100%'
                    }
                )
            ], style={'width': '35%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'})
        ]),

        # Model Performance Visualization
        html.Div([
            dcc.Graph(
                id='model-performance-graph',
                style={'margin-top': '20px'},
                config={
                    'displayModeBar': True,
                    'displaylogo': False,
                    'toImageButtonOptions': {
                        'format': 'png',
                        'filename': 'model_performance',
                        'height': 600,
                        'width': 1200,
                        'scale': 1
                    }
                }
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

    # Confirmation Modal for Model Removal
    dcc.ConfirmDialog(
        id='confirm-remove-model',
        message='',
    ),

    # Alert for removal status
    html.Div(id='remove-model-alert', style={'margin-bottom': '20px', 'display': 'none'}),
    # Edge Deployment Section
    html.Div([
        html.H3("📱 Edge Deployment & Code Generation", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Generate optimized code for edge devices and embedded systems deployment",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div([
            html.H5("🎯 Deployment Configuration", style={
                    'color': '#495057', 'margin-bottom': '15px'}),

            html.Div([
                html.Div([
                    html.Label("Target Platform:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='deployment-platform',
                        options=[
                            {'label': '🔧 Arduino (C++)', 'value': 'arduino'},
                            {'label': '🖥️ ARM Cortex-M (C)',
                             'value': 'arm_cortex_m'},
                            {'label': '🔬 Seeed XIAO nRF52840',
                                'value': 'seeed_xiao'},
                            {'label': '🐍 Python (Edge Testing)',
                             'value': 'python_edge'},
                            {'label': '📱 TensorFlow Lite Micro',
                                'value': 'tflite_micro'}
                        ],
                        placeholder="Select deployment target",
                        value='seeed_xiao',
                        style={'margin-bottom': '15px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("Optimization Level:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='optimization-level',
                        options=[
                            {'label': '🎯 Accuracy Priority', 'value': 'accuracy'},
                            {'label': '⚖️ Balanced', 'value': 'balanced'},
                            {'label': '⚡ Speed Priority', 'value': 'speed'},
                            {'label': '🔋 Power Efficiency', 'value': 'power'}
                        ],
                        placeholder="Select optimization strategy",
                        value='balanced',
                        style={'margin-bottom': '15px'}
                    )
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
            ]),

            # Platform information cards
            html.Div([
                html.Div([
                    html.H6("🔬 Seeed XIAO nRF52840", style={
                            'color': '#2E86AB', 'margin-bottom': '8px'}),
                    html.P("ARM Cortex-M4F @ 64MHz",
                           style={'margin': '2px 0', 'font-size': '12px'}),
                    html.P("256KB RAM, 1MB Flash", style={
                           'margin': '2px 0', 'font-size': '12px'}),
                    html.P("Bluetooth 5.0, Ultra-low power",
                           style={'margin': '2px 0', 'font-size': '12px'})
                ], style={
                    'background-color': '#e3f2fd',
                    'padding': '12px',
                    'border-radius': '6px',
                    'border-left': '3px solid #2196f3',
                    'width': '32%',
                    'display': 'inline-block',
                    'margin-right': '2%'
                }),

                html.Div([
                    html.H6("🔧 Arduino Platform", style={
                            'color': '#2E86AB', 'margin-bottom': '8px'}),
                    html.P("Compatible with Arduino IDE", style={
                           'margin': '2px 0', 'font-size': '12px'}),
                    html.P("C++ code generation",
                           style={'margin': '2px 0', 'font-size': '12px'}),
                    html.P("Wide hardware support", style={
                           'margin': '2px 0', 'font-size': '12px'})
                ], style={
                    'background-color': '#f3e5f5',
                    'padding': '12px',
                    'border-radius': '6px',
                    'border-left': '3px solid #9c27b0',
                    'width': '32%',
                    'display': 'inline-block',
                    'margin-right': '2%'
                }),

                html.Div([
                    html.H6("📱 TensorFlow Lite Micro", style={
                            'color': '#2E86AB', 'margin-bottom': '8px'}),
                    html.P("Optimized for microcontrollers", style={
                           'margin': '2px 0', 'font-size': '12px'}),
                    html.P("Minimal memory footprint", style={
                           'margin': '2px 0', 'font-size': '12px'}),
                    html.P("Hardware acceleration", style={
                           'margin': '2px 0', 'font-size': '12px'})
                ], style={
                    'background-color': '#e8f5e8',
                    'padding': '12px',
                    'border-radius': '6px',
                    'border-left': '3px solid #4caf50',
                    'width': '32%',
                    'display': 'inline-block'
                })
            ], style={'margin': '20px 0'})
        ]),

        html.Div([
            html.H5("🚀 Deployment Actions", style={
                    'color': '#495057', 'margin-bottom': '15px'}),

            html.Div([
                html.Button(
                    "🚀 Generate Deployment Code",
                    id='generate-code-btn',
                    style={
                        'background-color': '#e83e8c',
                        'border': 'none',
                        'padding': '15px 25px',
                        'font-size': '16px',
                        'border-radius': '6px',
                        'color': 'white',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '48%',
                        'margin-right': '4%'
                    }
                ),
                html.Button(
                    "📊 Resource Analysis",
                    id='resource-analysis-btn',
                    style={
                        'background-color': '#fd7e14',
                        'border': 'none',
                        'padding': '15px 25px',
                        'font-size': '16px',
                        'border-radius': '6px',
                        'color': 'white',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'width': '48%'
                    }
                )
            ])
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Deployment Output Section
    html.Div([
        html.H3("💻 Generated Code & Resources", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.Div(id='deployment-output', style={
            'min-height': '120px',
            'background-color': '#f8f9fa',
            'padding': '20px',
            'border-radius': '8px',
            'border': '1px solid #dee2e6',
            'margin-bottom': '20px'
        }, children=[
            html.P("No deployment code generated yet. Select a model and target platform to generate optimized code.",
                   style={'text-align': 'center', 'color': '#6c757d', 'font-style': 'italic', 'margin': '0'})
        ])
    ], style={
        'background-color': '#ffffff',
        'padding': '25px',
        'border-radius': '10px',
        'box-shadow': '0 2px 10px rgba(0,0,0,0.1)',
        'margin-bottom': '25px',
        'border': '1px solid #e9ecef'
    }),

    # Training Statistics Section
    html.Div([
        html.H3("📈 Training Statistics & Progress", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
        html.P("Monitor training progress and session statistics in real-time",
               style={'color': '#666', 'margin-bottom': '20px'}),

        html.Div(id='training-stats', children=[
            html.Div([
                html.H5("📊 Current Session Statistics", style={
                        'color': '#495057', 'margin-bottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H6("Training Status", style={
                                'color': '#666', 'margin-bottom': '5px'}),
                        html.P("No active session", style={
                               'color': '#6c757d', 'font-size': '18px', 'font-weight': 'bold'})
                    ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'}),

                    html.Div([
                        html.H6("Models Trained", style={
                                'color': '#666', 'margin-bottom': '5px'}),
                        html.P("0", style={
                               'color': '#28a745', 'font-size': '18px', 'font-weight': 'bold'})
                    ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'}),

                    html.Div([
                        html.H6("Best Accuracy", style={
                                'color': '#666', 'margin-bottom': '5px'}),
                        html.P(
                            "N/A", style={'color': '#007bff', 'font-size': '18px', 'font-weight': 'bold'})
                    ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'}),

                    html.Div([
                        html.H6("Total Time", style={
                                'color': '#666', 'margin-bottom': '5px'}),
                        html.P("00:00:00", style={
                               'color': '#6f42c1', 'font-size': '18px', 'font-weight': 'bold'})
                    ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'})
                ], id='session-stats', style={'padding': '20px'})
            ])
        ])
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
