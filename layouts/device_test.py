from dash import html, dcc
import plotly.graph_objects as go

layout = html.Div([
    html.Div([
        html.H2("📡 Device Testing & Real-Time Monitoring", style={
            'color': '#2E86AB', 'margin-bottom': '10px', 'text-align': 'center'}),
        html.P("Connect to Arduino/ESP32 via UART and visualize sensor data in real-time",
               style={'text-align': 'center', 'color': '#666', 'margin-bottom': '30px'}),

        # Coming soon notice
        html.Div([
            html.H3("🚧 Feature Under Development", style={
                'color': '#ff9800', 'text-align': 'center', 'margin-bottom': '20px'}),
            html.P([
                "This tab will enable real-time device testing with the following features:"
            ], style={'text-align': 'center', 'color': '#666', 'margin-bottom': '20px'}),
            
            html.Ul([
                html.Li("📌 Serial port connection (UART/USB)", style={'margin-bottom': '10px'}),
                html.Li("📊 Real-time sensor data plotting (aX, aY, aZ, gX, gY, gZ)", style={'margin-bottom': '10px'}),
                html.Li("🎯 Live activity classification with deployed model", style={'margin-bottom': '10px'}),
                html.Li("💾 Data recording for validation", style={'margin-bottom': '10px'}),
                html.Li("⚙️ Configurable sampling rate and buffer size", style={'margin-bottom': '10px'}),
                html.Li("📈 Performance metrics (latency, accuracy)", style={'margin-bottom': '10px'})
            ], style={'list-style-type': 'none', 'padding-left': '0'}),

            html.Div([
                html.H4("Planned Architecture:", style={'margin-top': '30px', 'margin-bottom': '15px'}),
                html.Pre("""
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│   Arduino/  │  UART   │   Python     │  Plot   │    Dash     │
│   ESP32     │────────▶│   Serial     │────────▶│   Graph     │
│   (Sensor)  │         │   Reader     │         │   (UI)      │
└─────────────┘         └──────────────┘         └─────────────┘
      │                        │                         │
      │ Collect IMU data       │ Parse & buffer          │ Display
      │ @ 100Hz                │ Run inference           │ Real-time
      └────────────────────────┴─────────────────────────┘
                """, style={
                    'background': '#f5f5f5',
                    'padding': '20px',
                    'border-radius': '8px',
                    'font-family': 'monospace',
                    'font-size': '12px',
                    'overflow-x': 'auto'
                })
            ], style={'margin-top': '30px'})

        ], style={
            'background': 'white',
            'padding': '40px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin': '40px auto',
            'max-width': '900px'
        }),

        # Placeholder UI elements (disabled)
        html.Div([
            html.H4("Serial Port Configuration (Coming Soon)", style={
                'color': '#6c757d', 'margin-bottom': '20px'}),
            
            html.Div([
                html.Label("Serial Port:", style={'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='serial-port-selector',
                    options=[{'label': 'No ports detected', 'value': 'none'}],
                    placeholder="Select serial port",
                    disabled=True,
                    style={'margin-bottom': '15px'}
                ),
                
                html.Label("Baud Rate:", style={'font-weight': 'bold', 'margin-bottom': '8px', 'display': 'block'}),
                dcc.Dropdown(
                    id='baud-rate-selector',
                    options=[
                        {'label': '9600', 'value': 9600},
                        {'label': '115200', 'value': 115200},
                        {'label': '921600', 'value': 921600}
                    ],
                    value=115200,
                    disabled=True,
                    style={'margin-bottom': '15px'}
                ),

                html.Button(
                    "🔌 Connect Device",
                    id='connect-device-btn',
                    disabled=True,
                    style={
                        'background-color': '#6c757d',
                        'color': 'white',
                        'border': 'none',
                        'padding': '12px 30px',
                        'border-radius': '6px',
                        'cursor': 'not-allowed',
                        'font-weight': 'bold',
                        'margin-top': '10px'
                    }
                )
            ], style={'width': '400px'}),

            # Placeholder graph
            html.Div([
                html.H4("Real-Time Sensor Plot (Coming Soon)", style={
                    'color': '#6c757d', 'margin-top': '40px', 'margin-bottom': '20px'}),
                dcc.Graph(
                    id='realtime-sensor-plot',
                    figure=go.Figure().update_layout(
                        title='Real-Time IMU Data',
                        xaxis_title='Time (s)',
                        yaxis_title='Sensor Values',
                        height=400,
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
                    config={'displayModeBar': False}
                )
            ])

        ], style={
            'background': 'white',
            'padding': '30px',
            'border-radius': '10px',
            'box-shadow': '0 2px 8px rgba(0,0,0,0.1)',
            'margin': '40px auto',
            'max-width': '1200px',
            'opacity': '0.6'
        })

    ], style={
        'max-width': '1400px',
        'margin': '0 auto',
        'padding': '20px'
    })
])
