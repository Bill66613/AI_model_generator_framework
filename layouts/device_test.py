from dash import html, dcc
import plotly.graph_objects as go

layout = html.Div([
    html.Div([
        html.H2("📡 Device Testing & Real-Time Monitoring", style={
            'color': '#2E86AB', 'margin-bottom': '10px', 'text-align': 'center'}),
        html.P("Connect to Arduino/ESP32 via UART and visualize sensor data in real-time",
               style={'text-align': 'center', 'color': '#666', 'margin-bottom': '30px'}),

        # Serial port configuration
        # Connection controls
        html.Div([
            html.H4("Serial Port Configuration", style={
                'color': '#2E86AB', 'margin-bottom': '20px'}),

            html.Div([
                html.Div([
                    html.Label("Serial Port:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    html.Div([
                        dcc.Dropdown(
                            id='serial-port-selector',
                            options=[
                                {'label': 'No ports detected', 'value': 'none'}],
                            placeholder="Select serial port",
                            style={
                                'width': '400px', 'display': 'inline-block', 'margin-right': '10px'}
                        ),
                        html.Button(
                            "🔄 Refresh",
                            id='refresh-ports-btn',
                            n_clicks=0,
                            style={
                                'background-color': '#6c757d',
                                'color': 'white',
                                'border': 'none',
                                'padding': '8px 20px',
                                'border-radius': '6px',
                                'cursor': 'pointer',
                                'font-weight': 'bold'
                            }
                        )
                    ], style={'margin-bottom': '15px'}),
                ], style={'margin-bottom': '15px'}),

                html.Div([
                    html.Label("Baud Rate:", style={
                               'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                    dcc.Dropdown(
                        id='baud-rate-selector',
                        options=[
                            {'label': '9600', 'value': 9600},
                            {'label': '115200', 'value': 115200},
                            {'label': '921600', 'value': 921600}
                        ],
                        value=115200,
                        style={'width': '200px', 'margin-bottom': '15px'}
                    ),
                ], style={'margin-bottom': '15px'}),

                html.Button(
                    "🔌 Connect Device",
                    id='connect-device-btn',
                    n_clicks=0,
                    style={
                        'background-color': '#28a745',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 30px',
                        'border-radius': '6px',
                        'cursor': 'pointer',
                        'font-weight': 'bold',
                        'margin-top': '10px'
                    }
                ),

                html.Div(id='connection-status', style={'margin-top': '10px'})
            ]),
        ], style={
            'background': 'white',
            'padding': '30px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '20px'
        }),

        # Real-time visualization and controls
        html.Div([
            # Top row: Plot and Live Prediction side by side
            html.Div([
                # Left: Real-time plot
                html.Div([
                    html.H4("Real-Time Sensor Data", style={
                        'color': '#2E86AB', 'margin-bottom': '15px'}),
                    dcc.Graph(
                        id='realtime-sensor-plot',
                        figure=go.Figure().update_layout(
                            title='Real-Time IMU Data',
                            xaxis_title='Time (s)',
                            yaxis_title='Sensor Values',
                            height=500,
                            annotations=[{
                                'text': 'Device connection required',
                                'xref': 'paper',
                                'yref': 'paper',
                                'x': 0.5,
                                'y': 0.5,
                                'showarrow': False,
                                'font': {'size': 20, 'color': '#ccc'}
                            }]
                        ),
                        config={'displayModeBar': True}
                    )
                ], style={'width': '70%', 'display': 'inline-block', 'vertical-align': 'top'}),

                # Right: Live prediction and stats
                html.Div([
                    html.H4("Live Activity Classification", style={
                        'color': '#2E86AB', 'margin-bottom': '15px', 'text-align': 'center'}),

                    html.Div([
                        html.Div(id='activity-prediction',
                                 style={'font-size': '28px', 'font-weight': 'bold',
                                        'text-align': 'center', 'padding': '20px',
                                        'background': '#f8f9fa', 'border-radius': '8px',
                                        'margin-bottom': '10px'}),
                        html.Div(id='prediction-confidence',
                                 style={'text-align': 'center', 'color': '#666',
                                        'margin-bottom': '20px'}),
                    ]),

                    html.Hr(),

                    html.H5("Data Statistics", style={
                            'margin-top': '20px', 'margin-bottom': '10px'}),
                    html.Div(id='data-stats', style={'font-size': '14px'}),

                    html.Hr(style={'margin-top': '20px'}),

                    html.Button(
                        "🗑️ Clear Buffer",
                        id='clear-buffer-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#ffc107',
                            'color': 'black',
                            'border': 'none',
                            'padding': '10px 20px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'width': '100%',
                            'margin-top': '10px'
                        }
                    ),
                    html.Div(id='clear-buffer-feedback')

                ], style={
                    'width': '28%',
                    'display': 'inline-block',
                    'vertical-align': 'top',
                    'margin-left': '2%',
                    'background': '#f8f9fa',
                    'padding': '20px',
                    'border-radius': '10px'
                })
            ], style={'margin-bottom': '20px'}),

            # Debug Console
            html.Div([
                html.Div([
                    html.H5("🔍 Debug Console", style={
                            'margin-bottom': '10px', 'color': '#2E86AB', 'display': 'inline-block'}),
                    html.Button(
                        "🗑️ Clear Console",
                        id='clear-console-btn',
                        n_clicks=0,
                        style={
                            'background-color': '#6c757d',
                            'color': 'white',
                            'border': 'none',
                            'padding': '6px 15px',
                            'border-radius': '6px',
                            'cursor': 'pointer',
                            'font-weight': 'bold',
                            'font-size': '12px',
                            'float': 'right'
                        }
                    )
                ], style={'margin-bottom': '10px', 'overflow': 'auto'}),
                html.Div(
                    id='debug-console',
                    style={
                        'background': '#1e1e1e',
                        'color': '#00ff00',
                        'padding': '15px',
                        'border-radius': '6px',
                        'font-family': 'monospace',
                        'font-size': '12px',
                        'height': '250px',
                        'overflow-y': 'auto',
                        'white-space': 'pre-wrap',
                        'word-wrap': 'break-word'
                    },
                    children="Waiting for connection..."
                )
            ], style={
                'background': 'white',
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
                'margin-bottom': '20px'
            }),

            # Instructions
            html.Div([
                html.H5("📋 Instructions", style={'margin-bottom': '10px'}),
                html.Ol([
                    html.Li(
                        "Upload your Arduino/ESP32 with the generated code from the Code Generation tab"),
                    html.Li("Connect the device to your computer via USB"),
                    html.Li("Click 'Refresh' to detect available serial ports"),
                    html.Li(
                        "Select the correct port and baud rate (default: 115200)"),
                    html.Li("Click 'Connect Device' to start real-time monitoring"),
                    html.Li(
                        "Watch the Debug Console for connection status and data parsing info"),
                    html.Li(
                        "The plot will show live sensor data, and activity predictions will appear on the right"),
                    html.Li(
                        "Expected data format from device: aX,aY,aZ,gX,gY,gZ (comma-separated)")
                ], style={'font-size': '14px', 'color': '#666'})
            ], style={
                'background': '#fff3cd',
                'padding': '20px',
                'border-radius': '8px',
                'border-left': '4px solid #ffc107'
            })

        ], style={
            'background': 'white',
            'padding': '30px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin-bottom': '20px'
        }),

        # Update intervals
        dcc.Interval(
            id='plot-update-interval',
            interval=200,  # Update plot every 200ms (5 Hz)
            n_intervals=0
        ),

        dcc.Interval(
            id='inference-interval',
            interval=750,  # Run inference every 750ms
            n_intervals=0
        )

    ], style={
        'max-width': '1400px',
        'margin': '0 auto',
        'padding': '20px'
    })
])
