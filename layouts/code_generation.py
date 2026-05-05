from dash import html, dcc
import plotly.graph_objects as go

layout = html.Div([
    html.Div([
        html.H2("🔧 Code Generation & Deployment", style={
            'color': '#2E86AB', 'margin-bottom': '10px', 'text-align': 'center'}),
        html.P("Generate embedded C/C++ code from trained models and deploy to target devices",
               style={'text-align': 'center', 'color': '#666', 'margin-bottom': '30px'}),

        # Workflow steps indicator
        html.Div([
            html.Div([
                html.Span("1️⃣ Select Model", style={
                          'margin-right': '20px', 'color': '#28a745'}),
                html.Span(
                    "→", style={'margin-right': '20px', 'color': '#ccc'}),
                html.Span("2️⃣ Configure Platform", style={
                          'margin-right': '20px', 'color': '#28a745'}),
                html.Span(
                    "→", style={'margin-right': '20px', 'color': '#ccc'}),
                html.Span("3️⃣ Generate Code", style={
                          'margin-right': '20px', 'color': '#28a745'}),
                html.Span(
                    "→", style={'margin-right': '20px', 'color': '#ccc'}),
                html.Span("4️⃣ Compile & Flash", style={'color': '#28a745'})
            ], style={'text-align': 'center', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '8px'})
        ], style={'margin-bottom': '30px'}),

        # Step 1: Model Selection
        html.Div([
            html.H4("📊 Step 1: Select Trained Model", style={
                'color': '#2E86AB', 'margin-bottom': '20px', 'border-bottom': '2px solid #2E86AB', 'padding-bottom': '10px'}),

            html.Div([
                html.Div([
                    html.Label("Select Model:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='deployment-model-selector',
                        options=[],
                        placeholder="Select a trained model from Training tab",
                        style={'margin-bottom': '15px'}
                    ),
                    html.Div(id='model-info-display', style={
                        'background': '#e3f2fd',
                        'padding': '15px',
                        'border-radius': '6px',
                        'border-left': '4px solid #2196f3',
                        'margin-bottom': '15px'
                    })
                ], style={'width': '100%'})
            ])
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '25px'
        }),

        # Step 2: Platform Configuration
        html.Div([
            html.H4("⚙️ Step 2: Configure Target Platform", style={
                'color': '#2E86AB', 'margin-bottom': '20px', 'border-bottom': '2px solid #2E86AB', 'padding-bottom': '10px'}),

            # --- Sub-section A: Platform & Model Configuration ---
            html.Div([
                # Left column: Framework + Board
                html.Div([
                    html.Label("Output Language / Framework:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='output-framework-selector',
                        options=[
                            {'label': '🔷 Arduino C++ (.ino) — Arduino framework, widest board support',
                             'value': 'arduino_cpp'},
                            {'label': '🇨 Generic C (C99) — Portable bare-metal C, no framework dependency',
                             'value': 'generic_c'},
                            {'label': '🅒+ Generic C++ (C++11) — Portable C++, standard library only',
                             'value': 'generic_cpp'},
                            {'label': '📡 ESP-IDF C — Native Espressif IoT Development Framework',
                             'value': 'esp_idf_c'},
                            {'label': '🐍 MicroPython — Python for microcontrollers (ESP32, RP2040)',
                             'value': 'micropython'},
                            {'label': '🌀 Zephyr RTOS C — Zephyr real-time OS (nRF, STM32, ESP32)',
                             'value': 'zephyr_c'},
                        ],
                        value='arduino_cpp',
                        placeholder='Select output language / framework',
                        style={'margin-bottom': '15px'}
                    ),
                    html.Div(id='framework-description', style={
                        'font-size': '12px', 'color': '#666', 'padding': '8px',
                        'background': '#f0f7ff', 'border-radius': '4px', 'border-left': '3px solid #2E86AB',
                        'margin-bottom': '15px'
                    }),

                    html.Label("Target Board:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='target-board-selector',
                        options=[],
                        value='generic',
                        placeholder='Select target board (or Generic)',
                        style={'margin-bottom': '15px'}
                    ),
                    html.Div(id='board-specs-display', style={
                        'font-size': '12px', 'color': '#555', 'padding': '8px',
                        'background': '#f8f9fa', 'border-radius': '4px',
                        'margin-bottom': '10px'
                    }),
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                # Right column: Model params + Deployment approach
                html.Div([
                    html.Div([
                        html.H5("⚙️ Model Parameters (from training)", style={
                            'font-size': '14px', 'color': '#495057', 'margin-bottom': '10px', 'margin-top': '0px'
                        }),
                        html.Div(id='model-parameters-display', style={
                            'background': '#f8f9fa',
                            'padding': '12px',
                            'border-radius': '6px',
                            'border-left': '4px solid #6c757d',
                            'font-size': '13px'
                        })
                    ]),

                    html.Label("Deployment Approach:", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'margin-top': '15px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='deployment-approach',
                        options=[
                            {'label': '🔧 Direct C/C++ Code Generation — Standalone, no runtime dependency',
                             'value': 'direct'},
                            {'label': '🧠 TFLite Micro — TensorFlow Lite for Microcontrollers runtime',
                             'value': 'tflite_micro'},
                            {'label': '📦 ONNX Runtime — Open Neural Network Exchange runtime',
                             'value': 'onnx_runtime'}
                        ],
                        value='direct',
                        placeholder="Select deployment approach",
                        style={'margin-bottom': '5px'}
                    ),
                    html.Div([
                        html.Div("• Direct: All C/C++ code generated — no external runtime needed", style={
                                 'margin-bottom': '3px'}),
                        html.Div("• TFLite Micro: .tflite model + interpreter (requires tensorflow)", style={
                                 'margin-bottom': '3px'}),
                        html.Div("• ONNX Runtime: .onnx model + ONNX C++ API (requires onnx, skl2onnx)", style={
                                 'font-size': '11px', 'color': '#999'})
                    ], style={'font-size': '12px', 'color': '#666', 'margin-top': '5px', 'font-style': 'italic',
                              'padding': '8px', 'background': '#f8f9fa', 'border-radius': '4px'}),
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
            ], style={'margin-bottom': '20px'}),

            html.Hr(style={'border': '1px solid #e9ecef', 'margin': '15px 0'}),

            # --- Sub-section B: Optimization & Quantization ---
            html.Div([
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
                        value='balanced',
                        placeholder="Select optimization strategy",
                        style={'margin-bottom': '15px'}
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("Weight Quantization:", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='quantization-mode',
                        options=[
                            {'label': '🔢 None (Float32) — Full precision', 'value': 'none'},
                            {'label': '⚡ INT8 — 75% smaller, ~1-2% accuracy loss', 'value': 'int8'},
                            {'label': '📊 INT16 — 50% smaller, negligible loss', 'value': 'int16'},
                            {'label': '🔀 Float16 — Reduced precision floats', 'value': 'float16'}
                        ],
                        value='none',
                        placeholder="Select weight quantization",
                        style={'margin-bottom': '15px'}
                    ),
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
            ], style={'margin-bottom': '15px'}),

            # --- Sub-section C: Inference Settings ---
            html.Div([
                html.Div([
                    html.Label("Window Overlap (%):", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Input(
                        id='deployment-stride',
                        type='number',
                        placeholder='50 (50% overlap)',
                        value=50,
                        min=0,
                        max=99,
                        style={'width': '100%', 'padding': '8px',
                               'border': '1px solid #ddd', 'border-radius': '4px',
                               'margin-bottom': '5px'}
                    ),
                    html.Div([
                        html.Div("0% = no overlap (fastest) · 50% = 2× updates · 75% = smoothest",
                                 style={'margin-bottom': '3px'}),
                    ], style={'font-size': '11px', 'color': '#666', 'font-style': 'italic'}),
                ], style={'width': '30%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("Confidence Threshold:", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Input(
                        id='deployment-confidence-threshold',
                        type='number',
                        placeholder='0.6',
                        value=0.6,
                        min=0.0,
                        max=1.0,
                        step=0.05,
                        style={'width': '100%', 'padding': '8px',
                               'border': '1px solid #ddd', 'border-radius': '4px',
                               'margin-bottom': '5px'}
                    ),
                    html.Div([
                        html.Div("Below threshold → 'unknown'. Higher = fewer false positives",
                                 style={'margin-bottom': '3px'}),
                    ], style={'font-size': '11px', 'color': '#666', 'font-style': 'italic'}),
                ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("Prediction Smoothing:", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Input(
                        id='deployment-smoothing-window',
                        type='number',
                        placeholder='3',
                        value=3,
                        min=1,
                        max=9,
                        step=2,
                        style={'width': '100%', 'padding': '8px',
                               'border': '1px solid #ddd', 'border-radius': '4px',
                               'margin-bottom': '5px'}
                    ),
                    html.Div([
                        html.Div("Majority vote over N predictions. 1 = off, 3 = recommended",
                                 style={'margin-bottom': '3px'}),
                    ], style={'font-size': '11px', 'color': '#666', 'font-style': 'italic'}),
                ], style={'width': '30%', 'display': 'inline-block', 'margin-left': '5%', 'vertical-align': 'top'})
            ], style={'margin-bottom': '15px'}),

            html.Hr(style={'border': '1px solid #e9ecef', 'margin': '15px 0'}),

            # --- Sub-section D: On-device Signal Filters ---
            html.Div([
                html.H5("📡 On-device Signal Preprocessing", style={
                    'font-size': '14px', 'color': '#495057', 'margin-bottom': '12px'}),

                html.Div([
                    # IIR Filter
                    html.Div([
                        dcc.Checklist(
                            id='deployment-iir-filter-enabled',
                            options=[{'label': ' IIR low-pass filter (Butterworth)', 'value': 'enabled'}],
                            value=[],
                            style={'margin-bottom': '5px'}
                        ),
                        html.Div([
                            html.Div("Per-sample Butterworth filter before buffering",
                                     style={'margin-bottom': '3px'}),
                            html.Div("⚠️ Training uses filtfilt (zero-phase); device uses causal lfilter — slight parity gap",
                                     style={'color': '#ff9800'}),
                        ], style={'font-size': '11px', 'color': '#666', 'margin-left': '25px', 'margin-bottom': '10px'}),
                    ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                    # Kalman Filter
                    html.Div([
                        dcc.Checklist(
                            id='deployment-kalman-filter-enabled',
                            options=[{'label': ' Kalman filter (recommended)', 'value': 'enabled'}],
                            value=[],
                            style={'margin-bottom': '5px'}
                        ),
                        html.Div([
                            html.Div("Constant-velocity Kalman filter per channel",
                                     style={'margin-bottom': '3px'}),
                            html.Div("✓ Causal — identical in training & deployment (no parity gap)",
                                     style={'color': '#28a745'}),
                        ], style={'font-size': '11px', 'color': '#666', 'margin-left': '25px', 'margin-bottom': '10px'}),
                    ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'}),
                ]),
            ]),
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '25px'
        }),

        # Step 3: Code Generation
        html.Div([
            html.H4("💻 Step 3: Generate Embedded Code", style={
                'color': '#2E86AB', 'margin-bottom': '20px', 'border-bottom': '2px solid #2E86AB', 'padding-bottom': '10px'}),

            html.Div([
                html.Div([
                    html.Button(
                        "⚡ Generate C/C++ Code",
                        id='generate-code-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#28a745',
                            'color': 'white',
                            'border': 'none',
                            'padding': '15px 30px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'font-size': '16px',
                            'display': 'inline-block',
                            'margin-right': '15px'
                        }
                    ),
                    html.Button(
                        "📊 Resource Analysis",
                        id='resource-analysis-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#fd7e14',
                            'color': 'white',
                            'border': 'none',
                            'padding': '15px 30px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'font-size': '16px',
                            'display': 'inline-block'
                        }
                    )
                ], style={'margin-bottom': '20px'}),

                html.Div(id='resource-analysis-output',
                         style={'margin-bottom': '20px'}),

                html.Div(id='code-generation-status',
                         style={'margin-bottom': '20px'}),

                # Code preview
                html.Div([
                    html.H5("Generated Code Preview:", style={
                            'color': '#495057', 'margin-bottom': '10px'}),
                    html.Div([
                        dcc.Textarea(
                            id='code-preview',
                            value='// Generated code will appear here...',
                            style={
                                'width': '100%',
                                'height': '400px',
                                'font-family': 'monospace',
                                'font-size': '12px',
                                'background': '#282c34',
                                'color': '#abb2bf',
                                'border': '1px solid #444',
                                'border-radius': '6px',
                                'padding': '15px',
                                'resize': 'vertical'
                            },
                            disabled=False
                        )
                    ]),

                    html.Div([
                        html.Button(
                            "💾 Download Code",
                            id='download-code-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#17a2b8',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold',
                                'margin-top': '15px',
                                'margin-right': '10px'
                            }
                        ),
                        html.Button(
                            "📋 Copy to Clipboard",
                            id='copy-code-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#6c757d',
                                'color': 'white',
                                'border': 'none',
                                'padding': '10px 20px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'font-weight': 'bold',
                                'margin-top': '15px'
                            }
                        )
                    ]),
                    dcc.Download(id='download-generated-code')
                ], id='code-preview-section', style={'display': 'none'})
            ])
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '25px'
        }),

        # Step 4: Compilation & Flashing
        html.Div([
            html.H4("🔥 Step 4: Compile & Flash to Device", style={
                'color': '#2E86AB', 'margin-bottom': '20px', 'border-bottom': '2px solid #2E86AB', 'padding-bottom': '10px'}),

            html.Div([
                # Toolchain selection
                html.Div([
                    html.Label("Build Toolchain:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    html.P("Select how to compile the Arduino C++ code generated in Step 3. "
                           "Both toolchains use the same generated code.",
                           style={'font-size': '12px', 'color': '#6c757d', 'margin-bottom': '8px'}),
                    dcc.RadioItems(
                        id='toolchain-selector',
                        options=[
                            {'label': ' Arduino CLI (Official Arduino toolchain)',
                             'value': 'arduino'},
                            {'label': ' PlatformIO (Advanced multi-platform IDE)',
                             'value': 'platformio'}
                        ],
                        value='arduino',
                        labelStyle={'display': 'block',
                                    'margin-bottom': '8px'},
                        style={'margin-bottom': '10px'}
                    ),
                    html.Div(id='toolchain-status', style={
                        'padding': '10px',
                        'background': '#e3f2fd',
                        'border-radius': '4px',
                        'font-size': '12px',
                        'margin-bottom': '15px'
                    })
                ], style={'margin-bottom': '20px'}),

                # Serial port selection
                html.Div([
                    html.Label("Serial Port:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    html.Div([
                        dcc.Dropdown(
                            id='serial-port-selector-deploy',
                            options=[],
                            placeholder="Select COM port",
                            style={'width': '70%', 'display': 'inline-block'}
                        ),
                        html.Button(
                            "🔄 Refresh",
                            id='refresh-ports-btn-deploy',
                            n_clicks=0,
                            style={
                                'background-color': '#6c757d',
                                'color': 'white',
                                'border': 'none',
                                'padding': '8px 15px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'margin-left': '10px',
                                'display': 'inline-block'
                            }
                        )
                    ]),

                    html.Div(id='port-info-display', style={
                        'margin-top': '10px',
                        'padding': '10px',
                        'background': '#fff3cd',
                        'border-radius': '4px',
                        'font-size': '13px',
                        'color': '#856404'
                    })
                ], style={'margin-bottom': '20px'}),

                # Compilation options
                html.Div([
                    dcc.Checklist(
                        id='compilation-options',
                        options=[
                            {'label': ' Verbose output', 'value': 'verbose'}
                        ],
                        value=[],
                        style={'margin-bottom': '15px'},
                        labelStyle={'display': 'block', 'margin-bottom': '8px'}
                    )
                ]),

                # Load previously generated code
                html.Div([
                    html.Label("Or load previously generated code:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    html.Div([
                        dcc.Dropdown(
                            id='load-generated-code-selector',
                            options=[],
                            placeholder="Select a previously generated project...",
                            style={'width': '75%', 'display': 'inline-block'}
                        ),
                        html.Button(
                            "📂 Load",
                            id='load-generated-code-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#6c757d',
                                'color': 'white',
                                'border': 'none',
                                'padding': '8px 15px',
                                'border-radius': '4px',
                                'cursor': 'pointer',
                                'margin-left': '10px',
                                'display': 'inline-block',
                                'vertical-align': 'top'
                            }
                        )
                    ])
                ], style={'margin-bottom': '15px'}),

                # Code readiness status
                html.Div(id='compile-code-status', style={
                    'margin-bottom': '15px',
                    'padding': '10px',
                    'border-radius': '4px',
                    'font-size': '13px'
                }),

                # Action buttons
                html.Div([
                    html.Button(
                        "⚙️ Compile",
                        id='compile-btn',
                        n_clicks=0,
                        disabled=True,
                        style={
                            'background-color': '#007bff',
                            'color': 'white',
                            'border': 'none',
                            'padding': '12px 25px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'margin-right': '10px',
                            'opacity': '0.6'
                        }
                    ),
                    html.Button(
                        "🔥 Compile & Flash",
                        id='compile-flash-btn',
                        n_clicks=0,
                        disabled=True,
                        style={
                            'background-color': '#dc3545',
                            'color': 'white',
                            'border': 'none',
                            'padding': '12px 25px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'opacity': '0.6'
                        }
                    )
                ], style={'margin-bottom': '20px'}),

                # Progress and output
                html.Div([
                    html.H5("Build Output:", style={
                            'color': '#495057', 'margin-bottom': '10px'}),
                    dcc.Loading(
                        id='compilation-loading',
                        type='default',
                        children=[
                            html.Div(id='compilation-output', style={
                                'background': '#1e1e1e',
                                'color': '#d4d4d4',
                                'padding': '15px',
                                'border-radius': '6px',
                                'font-family': 'monospace',
                                'font-size': '12px',
                                'height': '300px',
                                'overflow-y': 'auto',
                                'white-space': 'pre-wrap'
                            })
                        ]
                    ),

                    html.Div(id='flash-progress', style={'margin-top': '15px'})
                ], id='compilation-section', style={'display': 'none'})
            ])
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '25px'
        }),

        # Data stores
        dcc.Store(id='generated-code-store', storage_type='session'),
        dcc.Store(id='selected-model-store', storage_type='session'),
        dcc.Store(id='selected-framework-store', storage_type='session'),
        dcc.Interval(id='port-refresh-interval', interval=5000, disabled=True)

    ], style={
        'max-width': '1400px',
        'margin': '0 auto',
        'padding': '20px'
    })
])
