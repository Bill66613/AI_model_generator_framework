from dash import dcc, html

layout = html.Div([
    html.H3("🤖 Model Training & Evaluation"),
    
    # Model Selection Section
    html.Div([
        html.H4("1. Model Configuration"),
        html.Div([
            html.Label("Select Model Type:", style={'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
            dcc.Dropdown(
                id='model-type-selector',
                options=[
                    {'label': '🌲 Random Forest (Recommended for Edge)', 'value': 'random_forest'},
                    {'label': '🧠 Neural Network (High Accuracy)', 'value': 'neural_network'},
                    {'label': '⚡ Support Vector Machine (Fast Training)', 'value': 'svm'}
                ],
                placeholder="Select a machine learning model",
                value='random_forest',
                disabled=True,
                style={'margin-bottom': '10px'}
            ),
            html.Div([
                html.P([
                    "📊 ", html.Strong("Random Forest"), ": Best balance of accuracy and edge deployment efficiency"
                ], style={'margin': '5px 0', 'color': '#2E8B57'}),
                html.P([
                    "🧠 ", html.Strong("Neural Network"), ": Highest accuracy but requires more computational resources"
                ], style={'margin': '5px 0', 'color': '#4682B4'}),
                html.P([
                    "⚡ ", html.Strong("SVM"), ": Fast training and good performance on small datasets"
                ], style={'margin': '5px 0', 'color': '#FF6347'})
            ], style={'background-color': '#f8f9fa', 'padding': '15px', 'border-radius': '5px', 'margin': '10px 0'})
        ], style={'margin-bottom': '20px'})
    ]),
    
    # Training Controls
    html.Div([
        html.H4("2. Training Controls"),
        html.Div([
            html.Button(
                "🚀 Start Training", 
                id='start-training-btn', 
                disabled=True,
                className='btn btn-success',
                style={
                    'background-color': '#28a745',
                    'border': 'none',
                    'padding': '12px 24px',
                    'font-size': '16px',
                    'border-radius': '5px',
                    'color': 'white',
                    'cursor': 'pointer',
                    'margin-right': '10px'
                }
            ),
            html.Button(
                "📊 Hyperparameter Optimization", 
                id='optimize-hyperparams-btn', 
                disabled=True,
                className='btn btn-info',
                style={
                    'background-color': '#17a2b8',
                    'border': 'none',
                    'padding': '12px 24px',
                    'font-size': '16px',
                    'border-radius': '5px',
                    'color': 'white',
                    'cursor': 'pointer',
                    'margin-right': '10px'
                }
            ),
            html.Button(
                "🔄 Cross-Validation", 
                id='cross-validate-btn', 
                disabled=True,
                className='btn btn-warning',
                style={
                    'background-color': '#ffc107',
                    'border': 'none',
                    'padding': '12px 24px',
                    'font-size': '16px',
                    'border-radius': '5px',
                    'color': 'black',
                    'cursor': 'pointer'
                }
            )
        ], style={'margin-bottom': '20px'})
    ]),
    
    # Training Output
    html.Div(id='training-output', style={'margin': '20px 0'}),
    
    html.Hr(),
    
    # Model Evaluation Section
    html.Div([
        html.H4("3. Model Evaluation & Analysis"),
        html.Div([
            html.Label("Select Trained Model:", style={'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
            dcc.Dropdown(
                id='trained-model-selector',
                options=[],  # Populated dynamically
                placeholder="Select a trained model for evaluation",
                style={'margin-bottom': '10px'}
            ),
            html.Button(
                "📈 Detailed Evaluation", 
                id='evaluate-model-btn',
                style={
                    'background-color': '#6c757d',
                    'border': 'none',
                    'padding': '10px 20px',
                    'font-size': '14px',
                    'border-radius': '5px',
                    'color': 'white',
                    'cursor': 'pointer',
                    'margin-right': '10px'
                }
            ),
            html.Button(
                "🔍 Feature Importance", 
                id='feature-importance-btn',
                style={
                    'background-color': '#6f42c1',
                    'border': 'none',
                    'padding': '10px 20px',
                    'font-size': '14px',
                    'border-radius': '5px',
                    'color': 'white',
                    'cursor': 'pointer'
                }
            )
        ], style={'margin-bottom': '20px'})
    ]),
    
    # Model Performance Visualization
    dcc.Graph(id='model-performance-graph'),
    
    html.Hr(),
    
    # Deployment Section
    html.Div([
        html.H4("4. 📱 Edge Deployment"),
        html.Div([
            html.Div([
                html.Label("Target Platform:", style={'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
                dcc.Dropdown(
                    id='deployment-platform',
                    options=[
                        {'label': '🔧 Arduino (C++)', 'value': 'arduino'},
                        {'label': '🖥️ ARM Cortex-M (C)', 'value': 'arm_cortex_m'},
                        {'label': '🔬 Seeed XIAO nRF52840', 'value': 'seeed_xiao'},
                        {'label': '🐍 Python (Edge Testing)', 'value': 'python_edge'},
                        {'label': '📱 TensorFlow Lite Micro', 'value': 'tflite_micro'}
                    ],
                    placeholder="Select deployment target",
                    value='seeed_xiao',
                    style={'margin-bottom': '10px'}
                )
            ], style={'width': '48%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Optimization Level:", style={'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block'}),
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
                    style={'margin-bottom': '10px'}
                )
            ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),
        ], style={'overflow': 'hidden', 'margin-bottom': '15px'}),
        
        html.Div([
            html.Button(
                "🚀 Generate Deployment Code", 
                id='generate-code-btn',
                style={
                    'background-color': '#e83e8c',
                    'border': 'none',
                    'padding': '12px 24px',
                    'font-size': '16px',
                    'border-radius': '5px',
                    'color': 'white',
                    'cursor': 'pointer',
                    'margin-right': '10px'
                }
            ),
            html.Button(
                "📊 Resource Analysis", 
                id='resource-analysis-btn',
                style={
                    'background-color': '#fd7e14',
                    'border': 'none',
                    'padding': '12px 24px',
                    'font-size': '16px',
                    'border-radius': '5px',
                    'color': 'white',
                    'cursor': 'pointer'
                }
            )
        ], style={'margin-bottom': '20px'})
    ]),
    
    # Deployment Output
    html.Div(id='deployment-output', style={'margin': '20px 0'}),
    
    # Code Generation Results
    html.Div(id='generated-code-display', style={'margin': '20px 0'}),
    
    html.Hr(),
    
    # Training Progress and Statistics
    html.Div([
        html.H4("5. 📈 Training Statistics & Progress"),
        html.Div(id='training-stats', children=[
            html.Div([
                html.H5("📊 Current Session Statistics"),
                html.Div([
                    html.P("No training session active. Start training to see statistics.", 
                           style={'text-align': 'center', 'color': '#6c757d', 'font-style': 'italic'})
                ], id='session-stats')
            ], style={
                'background-color': '#f8f9fa', 
                'padding': '20px', 
                'border-radius': '5px',
                'border': '1px solid #dee2e6'
            })
        ])
    ])
])