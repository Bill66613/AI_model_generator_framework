from dash import Input, Output, State, html, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import numpy as np
import pandas as pd
from utils.device_reader import device_reader
from utils.model_training import create_feature_vector

# Cache for loaded model and scaler
_model_cache = {
    'model': None,
    'scaler': None,
    'model_info': None,
    'last_modified': None
}


def register_callbacks(app):

    @app.callback(
        [Output('serial-port-selector', 'options'),
         Output('serial-port-selector', 'disabled')],
        Input('refresh-ports-btn', 'n_clicks')
    )
    def refresh_serial_ports(n_clicks):
        """Refresh available serial ports"""
        ports = device_reader.get_available_ports()
        if not ports:
            ports = [{'label': 'No ports detected', 'value': 'none'}]
            return ports, True
        return ports, False

    @app.callback(
        [Output('connect-device-btn', 'children'),
         Output('connect-device-btn', 'style'),
         Output('connection-status', 'children'),
         Output('connection-status', 'style')],
        Input('connect-device-btn', 'n_clicks'),
        [State('serial-port-selector', 'value'),
         State('baud-rate-selector', 'value'),
         State('connect-device-btn', 'children')]
    )
    def toggle_connection(n_clicks, port, baudrate, button_text):
        """Connect or disconnect from device"""
        if not n_clicks:
            return (
                "🔌 Connect Device",
                {'background-color': '#28a745', 'color': 'white', 'border': 'none',
                 'padding': '12px 30px', 'border-radius': '6px', 'cursor': 'pointer',
                 'font-weight': 'bold', 'margin-top': '10px'},
                "",
                {'display': 'none'}
            )

        if device_reader.is_connected:
            # Disconnect
            device_reader.disconnect()
            return (
                "🔌 Connect Device",
                {'background-color': '#28a745', 'color': 'white', 'border': 'none',
                 'padding': '12px 30px', 'border-radius': '6px', 'cursor': 'pointer',
                 'font-weight': 'bold', 'margin-top': '10px'},
                "",
                {'display': 'none'}
            )
        else:
            # Connect
            if not port or port == 'none':
                return (
                    "🔌 Connect Device",
                    {'background-color': '#28a745', 'color': 'white', 'border': 'none',
                     'padding': '12px 30px', 'border-radius': '6px', 'cursor': 'pointer',
                     'font-weight': 'bold', 'margin-top': '10px'},
                    html.Div("⚠️ Please select a serial port", style={
                             'color': '#dc3545', 'margin-top': '10px'}),
                    {'display': 'block'}
                )

            success, message = device_reader.connect(port, baudrate)

            if success:
                return (
                    "🔌 Disconnect",
                    {'background-color': '#dc3545', 'color': 'white', 'border': 'none',
                     'padding': '12px 30px', 'border-radius': '6px', 'cursor': 'pointer',
                     'font-weight': 'bold', 'margin-top': '10px'},
                    html.Div(f"✅ {message}", style={
                             'color': '#28a745', 'margin-top': '10px'}),
                    {'display': 'block'}
                )
            else:
                return (
                    "🔌 Connect Device",
                    {'background-color': '#28a745', 'color': 'white', 'border': 'none',
                     'padding': '12px 30px', 'border-radius': '6px', 'cursor': 'pointer',
                     'font-weight': 'bold', 'margin-top': '10px'},
                    html.Div(f"❌ {message}", style={
                             'color': '#dc3545', 'margin-top': '10px'}),
                    {'display': 'block'}
                )

    @app.callback(
        Output('realtime-sensor-plot', 'figure'),
        Input('plot-update-interval', 'n_intervals'),
        prevent_initial_call=True
    )
    def update_realtime_plot(n_intervals):
        """Update real-time sensor plot"""
        if not device_reader.is_connected:
            # Show placeholder when disconnected
            fig = go.Figure()
            fig.update_layout(
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
            )
            return fig

        # Get current data
        data = device_reader.get_data()

        if len(data['time']) == 0:
            fig = go.Figure()
            fig.update_layout(
                title='Real-Time IMU Data',
                xaxis_title='Time (s)',
                yaxis_title='Sensor Values',
                height=500,
                annotations=[{
                    'text': 'Waiting for data...',
                    'xref': 'paper',
                    'yref': 'paper',
                    'x': 0.5,
                    'y': 0.5,
                    'showarrow': False,
                    'font': {'size': 20, 'color': '#ccc'}
                }]
            )
            return fig

        # Create subplots for accelerometer and gyroscope
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Accelerometer (m/s²)', 'Gyroscope (deg/s)'),
            vertical_spacing=0.12,
            row_heights=[0.5, 0.5]
        )

        # Accelerometer data
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['aX'], name='aX',
                       line=dict(color='#FF6B6B', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['aY'], name='aY',
                       line=dict(color='#4ECDC4', width=2)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['aZ'], name='aZ',
                       line=dict(color='#45B7D1', width=2)),
            row=1, col=1
        )

        # Gyroscope data
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['gX'], name='gX',
                       line=dict(color='#FFA07A', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['gY'], name='gY',
                       line=dict(color='#98D8C8', width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data['time'], y=data['gZ'], name='gZ',
                       line=dict(color='#6C88C4', width=2)),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="Acceleration (m/s²)", row=1, col=1)
        fig.update_yaxes(title_text="Angular Velocity (deg/s)", row=2, col=1)

        fig.update_layout(
            height=500,
            showlegend=True,
            hovermode='x unified',
            margin=dict(l=50, r=30, t=50, b=50)
        )

        return fig

    @app.callback(
        [Output('activity-prediction', 'children'),
         Output('prediction-confidence', 'children')],
        Input('inference-interval', 'n_intervals'),
        prevent_initial_call=True
    )
    def run_inference(n_intervals):
        """Run model inference on latest data"""
        try:
            if not device_reader.is_connected:
                return "No device connected", ""

            # Check if model exists
            model_path = 'persistent_data/trained_model.pkl'
            scaler_path = 'persistent_data/scaler.pkl'

            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                return "No trained model available", "Train a model in the Training tab first"

            # Get latest window
            window_data = device_reader.get_latest_window(150)

            if window_data is None:
                return "Collecting data...", f"Need 150 samples (current: {len(device_reader.data_buffer['time'])})"

            # Load or use cached model and scaler
            model_modified = os.path.getmtime(model_path)

            if (_model_cache['model'] is None or
                _model_cache['last_modified'] is None or
                    _model_cache['last_modified'] < model_modified):

                # Load model and scaler (only when needed)
                with open(model_path, 'rb') as f:
                    model_info = pickle.load(f)
                _model_cache['model'] = model_info['model']
                _model_cache['model_info'] = model_info

                with open(scaler_path, 'rb') as f:
                    _model_cache['scaler'] = pickle.load(f)

                _model_cache['last_modified'] = model_modified
                print(f"Loaded model from disk (modified: {model_modified})")

            # Use cached model and scaler
            model = _model_cache['model']
            scaler = _model_cache['scaler']
            model_info = _model_cache['model_info']

            # Convert window data to DataFrame
            window_df = pd.DataFrame({
                'aX': window_data['aX'],
                'aY': window_data['aY'],
                'aZ': window_data['aZ'],
                'gX': window_data['gX'],
                'gY': window_data['gY'],
                'gZ': window_data['gZ']
            })

            # Extract features using the same method as training
            sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
            include_freq = model_info.get('include_frequency', False)
            features_df = create_feature_vector(
                window_df, sensor_cols, sampling_rate=100, include_frequency=include_freq)

            # Scale features
            features_scaled = scaler.transform(features_df)

            # Predict
            prediction = model.predict(features_scaled)[0]

            # Get confidence if available
            confidence_text = ""
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_scaled)[0]
                confidence = max(probabilities) * 100
                confidence_text = f"Confidence: {confidence:.1f}%"

            return f"🎯 {prediction}", confidence_text

        except Exception as e:
            print(f"Inference error: {e}")
            import traceback
            traceback.print_exc()
            return "Error during inference", str(e)

    @app.callback(
        Output('data-stats', 'children'),
        Input('plot-update-interval', 'n_intervals'),
        prevent_initial_call=True
    )
    def update_stats(n_intervals):
        """Update data statistics"""
        if not device_reader.is_connected:
            return html.Div("Not connected", style={'color': '#999'})

        data = device_reader.get_data()
        num_samples = len(data['time'])

        if num_samples == 0:
            return html.Div("Waiting for data...", style={'color': '#999'})

        duration = data['time'][-1] if len(data['time']) > 0 else 0
        sampling_rate = num_samples / duration if duration > 0 else 0

        return html.Div([
            html.Div(f"📊 Samples: {num_samples}",
                     style={'margin-bottom': '5px'}),
            html.Div(f"⏱️ Duration: {duration:.1f}s",
                     style={'margin-bottom': '5px'}),
            html.Div(f"📈 Rate: {sampling_rate:.1f} Hz",
                     style={'margin-bottom': '5px'})
        ])

    @app.callback(
        Output('clear-buffer-feedback', 'children'),
        Input('clear-buffer-btn', 'n_clicks')
    )
    def clear_data_buffer(n_clicks):
        """Clear the data buffer"""
        if not n_clicks:
            return ""

        device_reader.clear_buffer()
        return html.Div("✅ Buffer cleared", style={'color': '#28a745', 'margin-top': '10px'})

    @app.callback(
        Output('debug-console', 'children'),
        [Input('plot-update-interval', 'n_intervals'),
         Input('clear-console-btn', 'n_clicks')],
        prevent_initial_call=True
    )
    def update_debug_console(n_intervals, clear_clicks):
        """Update debug console with latest logs"""
        # Check which input triggered the callback
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'clear-console-btn.n_clicks':
            device_reader.clear_debug_log()

        logs = device_reader.get_debug_log()

        if not logs:
            return "Waiting for connection..."

        # Show latest logs (most recent at bottom)
        return "\n".join(logs)
