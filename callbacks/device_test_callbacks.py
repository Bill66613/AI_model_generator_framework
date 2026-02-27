from dash import Input, Output, State, html, callback_context
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
import json
import glob
import numpy as np
import pandas as pd
from utils.device_reader import device_reader
from utils.model_training import create_feature_vector

from config.config import (
    PERSISTENT_DIR, SENSOR_COLUMNS, ACCEL_COLUMNS, GYRO_COLUMNS,
    DEFAULT_SAMPLING_RATE
)

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

        # Colour palette for traces
        accel_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFD93D', '#6BCB77', '#E8A87C']
        gyro_colors  = ['#FFA07A', '#98D8C8', '#6C88C4', '#C9B1FF', '#FF9CEE', '#A0D2DB']

        # Accelerometer data (dynamic — uses ACCEL_COLUMNS)
        for i, col in enumerate(ACCEL_COLUMNS):
            if col in data:
                fig.add_trace(
                    go.Scatter(x=data['time'], y=data[col], name=col,
                               line=dict(color=accel_colors[i % len(accel_colors)], width=2)),
                    row=1, col=1
                )

        # Gyroscope data (dynamic — uses GYRO_COLUMNS)
        for i, col in enumerate(GYRO_COLUMNS):
            if col in data:
                fig.add_trace(
                    go.Scatter(x=data['time'], y=data[col], name=col,
                               line=dict(color=gyro_colors[i % len(gyro_colors)], width=2)),
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
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def run_inference(n_intervals, base_dir):
        """Run model inference on latest data.

        Loads the most recent trained model from the working directory,
        reads the feature-engineering metadata so that inference uses the
        **same** feature method, window size and sampling rate as training.
        """
        try:
            if not device_reader.is_connected:
                return "No device connected", ""

            # ---- Resolve working directory ----
            if not base_dir:
                base_dir = PERSISTENT_DIR

            models_dir = os.path.join(base_dir, 'models')
            training_dir = os.path.join(base_dir, 'training')

            # ---- Locate trained model file (most recent .joblib) ----
            model_files = sorted(
                glob.glob(os.path.join(models_dir, '*.joblib')),
                key=os.path.getmtime, reverse=True
            )
            if not model_files:
                return "No trained model available", "Train a model in the Training tab first"

            model_path = model_files[0]
            model_modified = os.path.getmtime(model_path)

            # ---- Locate FE metadata (most recent) ----
            fe_meta_files = sorted(
                glob.glob(os.path.join(training_dir, '*_fe_metadata.json')),
                key=os.path.getmtime, reverse=True
            )
            fe_meta = {}
            if fe_meta_files:
                with open(fe_meta_files[0], 'r') as f:
                    fe_meta = json.load(f)

            # Read parameters from metadata, with sensible fallbacks
            window_size_samples = fe_meta.get(
                'window_size_samples',
                int(fe_meta.get('window_size_ms', 1500) / 1000 * fe_meta.get('sampling_rate', DEFAULT_SAMPLING_RATE))
            )
            sampling_rate = fe_meta.get('sampling_rate', DEFAULT_SAMPLING_RATE)
            feature_method = fe_meta.get('feature_method', 'orientation_invariant_time_only')
            include_freq = fe_meta.get('include_frequency', False)
            orientation_robust = fe_meta.get('orientation_robust', True)
            include_per_axis = fe_meta.get('include_per_axis', False)
            sensor_cols = fe_meta.get('sensor_columns', SENSOR_COLUMNS)

            # ---- Get latest window from device ----
            window_data = device_reader.get_latest_window(window_size_samples)

            if window_data is None:
                current = len(device_reader.data_buffer['time'])
                return "Collecting data...", f"Need {window_size_samples} samples (current: {current})"

            # ---- Load / cache model ----
            if (_model_cache['model'] is None or
                _model_cache['last_modified'] is None or
                    _model_cache['last_modified'] < model_modified):
                import joblib as _jl
                model_bundle = _jl.load(model_path)
                _model_cache['model'] = model_bundle.get('model', model_bundle)
                _model_cache['model_info'] = model_bundle
                _model_cache['scaler'] = model_bundle.get('scaler')
                _model_cache['last_modified'] = model_modified
                print(f"Loaded model from {model_path}")

            model = _model_cache['model']
            scaler = _model_cache['scaler']

            # ---- Build feature vector using the SAME method as training ----
            window_df = pd.DataFrame(window_data)

            features_df = create_feature_vector(
                window_df, sensor_cols, sampling_rate=sampling_rate,
                include_frequency=include_freq,
                orientation_robust=orientation_robust,
                include_per_axis=include_per_axis
            )

            # Scale features (if a scaler was saved with the model)
            if scaler is not None:
                features_scaled = scaler.transform(features_df)
            else:
                features_scaled = features_df.values

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
