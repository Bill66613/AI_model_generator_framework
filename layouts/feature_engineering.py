from dash import html, dcc, dash_table

layout = html.Div([
    html.Div([
        html.H2("🔬 Feature Engineering & Dataset Preparation", style={
            'color': '#2E86AB', 'margin-bottom': '10px', 'text-align': 'center'}),
        html.P("Unified feature extraction and dataset preparation across all activity labels",
               style={'text-align': 'center', 'color': '#666', 'margin-bottom': '30px'}),

        # Global configuration notice
        html.Div([
            html.H5("⚙️ Unified Settings", style={
                    'color': '#495057', 'margin-bottom': '15px'}),
            html.P([
                "Configure feature engineering settings ",
                html.Strong("once"),
                " and apply them ",
                html.Strong("consistently"),
                " across all activity labels. This ensures fair model comparison and prevents data leakage."
            ], style={'color': '#666', 'background': '#e7f3ff', 'padding': '15px', 'border-radius': '8px', 'border-left': '4px solid #007bff'})
        ], style={'margin-bottom': '30px'}),

        # Step 1: Dataset Selection
        html.Div([
            html.H3("📊 Step 1: Select Training Datasets", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
            html.P("Select windows from all activity labels to include in training",
                   style={'color': '#666', 'margin-bottom': '20px'}),

            # Activity label selector
            html.Div([
                html.Label("Available Activity Labels:", style={
                    'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='activity-labels-selector',
                    options=[],
                    placeholder="Loading available labels...",
                    multi=True,
                    style={'margin-bottom': '15px'}
                ),
                html.Div(id='activity-labels-info', style={
                    'color': '#666', 'font-size': '14px', 'margin-bottom': '20px'
                })
            ]),

            # Windows per label display
            html.Div(id='windows-per-label-display', style={
                'background': '#f8f9fa',
                'padding': '20px',
                'border-radius': '8px',
                'margin-bottom': '20px'
            }),

            # Action buttons
            html.Div([
                html.Button(
                    "📋 Select All Labels",
                    id='select-all-labels-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#17a2b8',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 30px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'margin-right': '15px'
                    }
                ),
                html.Button(
                    "🗑️ Clear Selection",
                    id='clear-labels-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#6c757d',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 30px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold'
                    }
                )
            ], style={'margin-bottom': '30px'})
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '30px'
        }),

        # Step 2: Feature Engineering Configuration
        html.Div([
            html.H3("🔧 Step 2: Configure Feature Engineering", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
            html.P("Settings will be applied uniformly across all selected datasets",
                   style={'color': '#666', 'margin-bottom': '20px'}),

            html.Div([
                # Feature selection
                html.Div([
                    html.Label("Feature Extraction Method:", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='global-feature-selection',
                        options=[
                            {'label': '🧭 Orientation-Invariant Time-Domain ONLY - 41 features',
                             'value': 'orientation_invariant_time_only'},
                            {'label': '🧭 Orientation-Invariant + DFT - 63 features',
                             'value': 'orientation_invariant'},
                            {'label': '🎯 Per-Axis Time-Domain - 90 features',
                             'value': 'time_domain'},
                            {'label': '🎯 Per-Axis All + FFT - 156 features',
                             'value': 'all'},
                            {'label': '🌊 Per-Axis Frequency Only - 66 features',
                             'value': 'frequency_domain'},
                            {'label': '📊 Raw Sensor Axes - 6 features',
                             'value': 'raw'}
                        ],
                        value='orientation_invariant_time_only',
                        placeholder="Select feature extraction method",
                        clearable=False,
                        style={'margin-bottom': '20px'}
                    ),
                    html.Div([
                        html.Strong("⚠️ Note: "),
                        "Orientation-robust DFT (63 features) ",
                        html.Strong("is fully deployable"),
                        " — code generators emit a lightweight sin/cos DFT. ",
                        "Only ",
                        html.Strong("per-axis"),
                        " frequency features (66 / 156) cannot be deployed."
                    ], style={
                        'padding': '12px',
                        'backgroundColor': '#fff3cd',
                        'borderLeft': '4px solid #ffc107',
                        'borderRadius': '6px',
                        'fontSize': '13px',
                        'color': '#856404',
                        'marginBottom': '15px'
                    }),
                    html.Div(id='feature-count-display', style={
                        'color': '#666', 'font-size': '14px', 'margin-bottom': '20px'
                    })
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                # Normalization method
                html.Div([
                    html.Label("Normalization Method (applied during training):", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='global-normalization-method',
                        options=[
                            {'label': '📏 Standard Scaler (Z-score)',
                             'value': 'standard'},
                            {'label': '📐 MinMax Scaler (0-1)',
                             'value': 'minmax'},
                            {'label': '🎯 Robust Scaler (median/IQR)',
                             'value': 'robust'},
                            {'label': '❌ No Normalization', 'value': 'none'}
                        ],
                        value='standard',
                        placeholder="Select normalization method",
                        clearable=False,
                        style={'margin-bottom': '20px'}
                    ),
                    html.Div([
                        html.P([
                            "ℹ️ Normalization is ",
                            html.Strong("not applied here"),
                            " — it is saved to metadata and applied during ",
                            html.Strong("model training"),
                            " so the scaler is bundled with the model for deployment."
                        ], style={
                            'font-size': '12px', 'color': '#856404', 'margin': '0',
                            'padding': '8px', 'background': '#fff3cd',
                            'border-radius': '4px', 'border-left': '3px solid #ffc107'
                        })
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
            ], style={'margin-bottom': '30px'}),

            # Window Configuration (for zero-padding)
            html.Div([
                html.H5("⚙️ Window Configuration", style={
                        'color': '#495057', 'margin-bottom': '15px'}),

                html.Div([
                    html.Label("Target Window Size (ms):", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Input(
                        id='global-window-size',
                        type='number',
                        value=1500,
                        min=1,
                        placeholder="Window duration in milliseconds",
                        style={'width': '100%', 'padding': '8px',
                               'border': '1px solid #ddd', 'border-radius': '4px'}
                    ),
                    html.Div("💡 Windows smaller than this will be zero-padded (1500ms = 150 samples @ 100Hz)",
                             style={'font-size': '12px', 'color': '#666', 'margin-top': '5px', 'font-style': 'italic'})
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top'}),

                html.Div([
                    html.Label("Sampling Rate (Hz):", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Input(
                        id='global-sampling-rate',
                        type='number',
                        value=100,
                        min=1,
                        placeholder="Sampling rate in Hz",
                        style={'width': '100%', 'padding': '8px',
                               'border': '1px solid #ddd', 'border-radius': '4px'}
                    ),
                    html.Div("ℹ️ Used to calculate time duration from samples",
                             style={'font-size': '12px', 'color': '#666', 'margin-top': '5px', 'font-style': 'italic'})
                ], style={'width': '48%', 'display': 'inline-block', 'margin-left': '4%', 'vertical-align': 'top'})
            ], style={'margin-bottom': '30px', 'padding': '15px', 'background': '#f8f9fa', 'border-radius': '6px'})
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '30px'
        }),

        # Step 3: Data Augmentation (optional)
        html.Div([
            html.H3("🔁 Step 3: Data Augmentation (Optional)", style={
                'color': '#2E86AB', 'margin-bottom': '10px'}),
            html.P(
                "Generate synthetic training windows to improve model robustness. "
                "Augmentation is applied to raw sensor signals before feature extraction, "
                "so new features are computed naturally from augmented data.",
                style={'color': '#666', 'margin-bottom': '20px'}),

            # Enable toggle
            html.Div([
                dcc.Checklist(
                    id='augmentation-enable',
                    options=[
                        {'label': '  Enable Data Augmentation', 'value': 'enabled'}],
                    value=[],
                    style={'font-weight': 'bold',
                           'font-size': '16px', 'margin-bottom': '15px'}
                ),
            ]),

            # Augmentation options (collapsible)
            html.Div(id='augmentation-options-container', children=[
                # Method selection
                html.Div([
                    html.Label("Augmentation Methods:", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Checklist(
                        id='augmentation-methods',
                        options=[
                            {'label': '  📊 Jittering — Add Gaussian noise (simulates sensor noise variation)',
                             'value': 'jitter'},
                            {'label': '  📏 Scaling — Vary amplitude ±10% (simulates different movement intensities)',
                             'value': 'scaling'},
                            {'label': '  🔄 Rotation — Random 3D rotation ±20° (simulates device orientation changes) ⭐ RECOMMENDED',
                             'value': 'rotation'},
                            {'label': '  ⏱️ Time Warping — Smooth speed variation (simulates pace changes)',
                             'value': 'time_warp'},
                            {'label': '  🔀 Permutation — Shuffle temporal segments (order-invariant features)',
                             'value': 'permutation'},
                        ],
                        value=['jitter', 'rotation'],
                        style={'line-height': '2.2'}
                    ),
                ], style={'margin-bottom': '20px'}),

                # Augmentation factor
                html.Div([
                    html.Div([
                        html.Label("Augmentation Factor (copies per window):", style={
                            'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                        dcc.Slider(
                            id='augmentation-factor',
                            min=1,
                            max=10,
                            step=1,
                            value=2,
                            marks={1: '1×', 2: '2×',
                                   3: '3×', 5: '5×', 10: '10×'},
                            tooltip={"placement": "bottom",
                                     "always_visible": True}
                        ),
                        html.Div(id='augmentation-factor-display', style={
                            'color': '#666', 'font-size': '13px', 'margin-top': '8px'
                        })
                    ], style={'width': '60%', 'display': 'inline-block', 'vertical-align': 'top'}),

                    html.Div([
                        html.Div([
                            html.Strong("💡 Recommendation:"),
                            html.Br(),
                            "• 2-3× for small datasets (<200 windows)",
                            html.Br(),
                            "• 1-2× for balanced datasets (>500 windows)",
                            html.Br(),
                            "• Rotation + Jitter is the best combination for HAR",
                        ], style={
                            'padding': '12px',
                            'backgroundColor': '#e8f5e9',
                            'borderLeft': '4px solid #4caf50',
                            'borderRadius': '6px',
                            'fontSize': '12px',
                            'color': '#2e7d32',
                        })
                    ], style={'width': '35%', 'display': 'inline-block',
                              'margin-left': '5%', 'vertical-align': 'top'})
                ], style={'margin-bottom': '15px'}),

                # Static / stationary class protection
                html.Div([
                    html.Div([
                        html.Strong("🛡️ Class-Aware Protection"),
                        html.Span(" — Static activities get micro-jitter only",
                                  style={'color': '#666', 'fontSize': '13px'}),
                    ], style={'margin-bottom': '8px'}),
                    html.P(
                        "Activities like 'still' or 'standing' are defined by the absence of motion. "
                        "Full augmentation (rotation, scaling) would make them resemble low-intensity "
                        "movement and cause class confusion. These labels receive only σ=0.01 sensor "
                        "noise instead.",
                        style={'color': '#666', 'fontSize': '12px', 'margin-bottom': '10px'}),
                    html.Label("Static activity labels (auto-detected, editable):", style={
                        'font-weight': 'bold', 'margin-bottom': '5px', 'display': 'block',
                        'fontSize': '13px'}),
                    dcc.Input(
                        id='augmentation-static-labels',
                        type='text',
                        placeholder='e.g. still, standing, sitting  (comma-separated)',
                        value='',
                        style={'width': '100%', 'padding': '8px', 'borderRadius': '5px',
                               'border': '1px solid #ced4da', 'fontSize': '13px'}
                    ),
                    html.Div(
                        "Leave blank for auto-detection (keywords: still, stand, sit, lying, idle)",
                        style={'color': '#999', 'fontSize': '11px', 'margin-top': '4px'}),
                ], style={
                    'padding': '15px',
                    'backgroundColor': '#fff3e0',
                    'borderLeft': '4px solid #ff9800',
                    'borderRadius': '6px',
                    'margin-bottom': '15px',
                }),
                # hidden by default, shown when enabled
            ], style={'display': 'none'}),
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '30px'
        }),

        # Step 4: Train/Val/Test Split Configuration
        html.Div([
            html.H3("📊 Step 4: Configure Dataset Split", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),
            html.P("Define how to split the combined dataset into train/validation/test sets",
                   style={'color': '#666', 'margin-bottom': '20px'}),

            # Split mode selector
            html.Div([
                html.Label("Split Mode:", style={
                    'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.RadioItems(
                    id='split-mode-selector',
                    options=[
                        {'label': ' 🎲 Auto (ratio-based random split)', 'value': 'auto'},
                        {'label': ' ✋ Manual (assign each window by hand)', 'value': 'manual'},
                    ],
                    value='auto',
                    inline=True,
                    style={'font-size': '15px', 'margin-bottom': '20px'}
                ),
            ]),

            # --- Auto split panel ---
            html.Div(id='auto-split-panel', children=[
                html.Div([
                    # Train split
                    html.Div([
                        html.Label("Training Set Ratio:", style={
                            'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                        dcc.Slider(
                            id='global-train-split',
                            min=0.5,
                            max=0.9,
                            step=0.05,
                            value=0.7,
                            marks={0.5: '50%', 0.6: '60%',
                                   0.7: '70%', 0.8: '80%', 0.9: '90%'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(id='global-train-split-display', style={
                            'color': '#666', 'font-size': '14px', 'margin-top': '10px'
                        })
                    ], style={'width': '31%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '3%'}),

                    # Validation split
                    html.Div([
                        html.Label("Validation Set Ratio:", style={
                            'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                        dcc.Slider(
                            id='global-val-split',
                            min=0.05,
                            max=0.3,
                            step=0.05,
                            value=0.15,
                            marks={0.05: '5%', 0.1: '10%',
                                   0.15: '15%', 0.2: '20%', 0.3: '30%'},
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Div(id='global-val-split-display', style={
                            'color': '#666', 'font-size': '14px', 'margin-top': '10px'
                        })
                    ], style={'width': '31%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-right': '3%'}),

                    # Test split (calculated)
                    html.Div([
                        html.Label("Test Set Ratio:", style={
                            'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                        html.Div(id='global-test-split-display', style={
                            'font-size': '28px',
                            'font-weight': 'bold',
                            'color': '#28a745',
                            'margin-top': '20px'
                        })
                    ], style={'width': '31%', 'display': 'inline-block', 'vertical-align': 'top'})
                ], style={'margin-bottom': '20px'}),

                # Random state
                html.Div([
                    html.Label("Random Seed (for reproducibility):", style={
                        'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Input(
                        id='global-random-state',
                        type='number',
                        value=42,
                        min=0,
                        max=9999,
                        style={'width': '200px', 'padding': '8px',
                               'border-radius': '4px', 'border': '1px solid #ccc'}
                    ),
                    html.P("ℹ️ Same seed = reproducible splits across runs", style={
                        'font-size': '12px', 'color': '#666', 'font-style': 'italic', 'margin-top': '5px'
                    })
                ])
            ]),

            # --- Manual split panel ---
            html.Div(id='manual-split-panel', style={'display': 'none'}, children=[
                html.Div([
                    html.P([
                        "Select activity labels above (Step 1), then click ",
                        html.Strong("Load Windows"),
                        " to list all windows. Assign each window to ",
                        html.Strong("Train"), ", ", html.Strong("Val"), ", or ",
                        html.Strong("Test"),
                        " using the dropdown in the table. "
                        "Windows left as 'Unassigned' are excluded from training."
                    ], style={'color': '#555', 'margin-bottom': '15px',
                              'background': '#e7f3ff', 'padding': '12px',
                              'border-radius': '6px', 'border-left': '4px solid #2196f3'}),

                    html.Div([
                        html.Button(
                            "🔄 Load / Refresh Windows",
                            id='load-manual-split-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#17a2b8', 'color': 'white',
                                'border': 'none', 'padding': '10px 24px',
                                'border-radius': '6px', 'cursor': 'pointer',
                                'font-weight': 'bold', 'margin-right': '10px'
                            }
                        ),
                        html.Button(
                            "📋 Auto-assign remaining (70/15/15)",
                            id='auto-assign-remaining-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#6c757d', 'color': 'white',
                                'border': 'none', 'padding': '10px 24px',
                                'border-radius': '6px', 'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        ),
                    ], style={'margin-bottom': '15px'}),

                    # Summary badges
                    html.Div(id='manual-split-summary', style={'margin-bottom': '12px'}),

                    # Assignment table
                    html.Div(id='manual-split-table-container', children=[
                        html.P("Click 'Load / Refresh Windows' to populate the table.",
                               style={'color': '#999', 'font-style': 'italic'})
                    ])
                ])
            ]),

        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '30px'
        }),

        # Step 5: Execute Feature Engineering
        html.Div([
            html.H3("🚀 Step 5: Execute Feature Engineering", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),

            html.Button(
                "⚙️ Engineer Features for All Selected Datasets",
                id='execute-feature-engineering-btn',
                n_clicks=0,
                style={
                    'background-color': '#28a745',
                    'color': 'white',
                    'border': 'none',
                    'padding': '15px 40px',
                    'border-radius': '8px',
                    'cursor': 'pointer',
                    'font-weight': 'bold',
                    'font-size': '18px',
                    'width': '100%',
                    'box-shadow': '0 4px 8px rgba(0,0,0,0.2)',
                    'margin-bottom': '20px'
                }
            ),

            # Results display
            html.Div(id='feature-engineering-results', style={
                'margin-top': '20px'
            }),

            # Dataset statistics table
            html.Div(id='engineered-dataset-stats', style={
                'margin-top': '20px'
            })
        ], style={
            'background': 'white',
            'padding': '25px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '30px'
        }),

        # Hidden stores
        dcc.Store(id='engineered-dataset-store'),
        dcc.Store(id='manual-split-store'),   # {file_path: 'train'|'val'|'test'|'unassigned'}

    ], style={
        'max-width': '1400px',
        'margin': '0 auto',
        'padding': '20px'
    })
])
