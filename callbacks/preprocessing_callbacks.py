import os
import json
import pandas as pd
from dash import Input, Output, State, no_update, html, dcc
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

from config.config import (
    PERSISTENT_DIR, METADATA_FILE, DATASETS_DIR, WINDOWS_DIR, TRAINING_DIR, MODELS_DIR,
    get_dataset_path, get_window_path, get_window_pattern, get_training_data_path
)
from utils.data_processing import clean_data, low_pass_filter
from utils.model_training import extract_time_domain_features, extract_frequency_domain_features


def _get_feature_method_label(feature_method):
    """Convert feature method code to descriptive label."""
    labels = {
        'all': 'Time + Frequency Domain',
        'statistical': 'Raw Sensor Axes Only',
        'time_domain': 'Time-Domain Only',
        'custom': 'Custom Selection'
    }
    return labels.get(feature_method, feature_method)


def compute_window_quality(window_data, sensor_cols=['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']):
    """
    Calculate quality score for a window.

    Args:
        window_data: DataFrame with sensor readings
        sensor_cols: List of sensor column names

    Returns:
        quality_score: 0-1 (1 = high quality)
        reasons: List of quality issues
    """
    quality_score = 1.0
    reasons = []

    # Check 1: Sufficient variance (not stationary/flat line)
    available_accel = [col for col in [
        'aX', 'aY', 'aZ'] if col in window_data.columns]
    if available_accel:
        variance = window_data[available_accel].var().mean()
        if variance < 0.01:
            quality_score -= 0.3
            reasons.append("Low variance (possibly stationary)")

    # Check 2: No extreme outliers
    for col in sensor_cols:
        if col in window_data.columns:
            # Accelerometer: check for > 4g (unusual for human activities)
            if col.startswith('a') and (window_data[col].abs() > 40).any():
                quality_score -= 0.2
                reasons.append(f"Extreme outliers in {col}")
                break
            # Gyroscope: check for > 2000 deg/s
            elif col.startswith('g') and (window_data[col].abs() > 2000).any():
                quality_score -= 0.2
                reasons.append(f"Extreme outliers in {col}")
                break

    # Check 3: No missing data
    if window_data[sensor_cols].isnull().any().any():
        quality_score -= 0.5
        reasons.append("Missing data")

    # Check 4: Sufficient data points
    if len(window_data) < 100:  # Less than 1 second @ 100Hz
        quality_score -= 0.3
        reasons.append("Insufficient data points")

    return max(0, quality_score), reasons


def generate_sliding_windows_from_current(current_windows, df, window_size_samples,
                                          overlap_percent, quality_threshold):
    """
    Generate overlapping windows from current manually-selected windows.
    Merges nearby manual windows into continuous regions to maximize window generation.

    Args:
        current_windows: List of current window configurations
        df: DataFrame with full dataset
        window_size_samples: Number of samples per window
        overlap_percent: Overlap percentage (0-90)
        quality_threshold: Minimum quality score (0-1)

    Returns:
        good_windows: List of high-quality window data
        flagged_windows: List of low-quality windows
        stats: Dictionary with statistics
    """
    stride = int(window_size_samples * (1 - overlap_percent / 100))
    good_windows = []
    flagged_windows = []

    sensor_cols = [col for col in df.columns if col in [
        'aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']]

    # Sort windows by start time (ensure numeric values)
    sorted_windows = sorted(
        current_windows, key=lambda w: float(w['start_time']))

    # Merge overlapping or adjacent windows into continuous regions
    # This allows sliding windows to span across multiple manual selections
    merged_regions = []
    if sorted_windows:
        current_region = {
            'start_time': float(sorted_windows[0]['start_time']),
            'end_time': float(sorted_windows[0]['end_time']),
            'window_ids': [sorted_windows[0]['window_id']]
        }

        for window in sorted_windows[1:]:
            # If windows overlap or are within 1 window size of each other, merge them
            gap = float(window['start_time']) - \
                float(current_region['end_time'])
            # Approximate samples
            gap_samples = gap * (1000 / (window_size_samples * 10))

            if gap <= 0 or gap_samples < window_size_samples:
                # Merge: extend current region
                current_region['end_time'] = max(
                    float(current_region['end_time']), float(window['end_time']))
                current_region['window_ids'].append(window['window_id'])
            else:
                # New separate region
                merged_regions.append(current_region)
                current_region = {
                    'start_time': float(window['start_time']),
                    'end_time': float(window['end_time']),
                    'window_ids': [window['window_id']]
                }

        merged_regions.append(current_region)

    # Generate sliding windows from each merged region
    for region in merged_regions:
        start_time = region['start_time']
        end_time = region['end_time']

        # Get the time indices for this region
        region_mask = (df['Time_seconds'] >= start_time) & (
            df['Time_seconds'] <= end_time)
        region_data = df[region_mask].copy()

        if len(region_data) < window_size_samples:
            continue

        # Generate sliding windows within this region
        region_start_idx = region_data.index[0]

        for start_idx in range(0, len(region_data) - window_size_samples + 1, stride):
            end_idx = start_idx + window_size_samples
            window_df = region_data.iloc[start_idx:end_idx].copy()

            # Compute quality
            quality, reasons = compute_window_quality(window_df, sensor_cols)

            window_info = {
                'data': window_df,
                'start_idx': region_start_idx + start_idx,
                'end_idx': region_start_idx + end_idx,
                'start_time': window_df['Time_seconds'].iloc[0],
                'end_time': window_df['Time_seconds'].iloc[-1],
                'quality': quality,
                'reasons': reasons,
                'parent_window': '-'.join(map(str, region['window_ids']))
            }

            if quality >= quality_threshold:
                good_windows.append(window_info)
            else:
                flagged_windows.append(window_info)

    stats = {
        'total_generated': len(good_windows) + len(flagged_windows),
        'good_windows': len(good_windows),
        'flagged_windows': len(flagged_windows),
        'avg_quality': np.mean([w['quality'] for w in good_windows]) if good_windows else 0,
        'overlap_percent': overlap_percent,
        'stride_samples': stride
    }

    return good_windows, flagged_windows, stats


def register_callbacks(app):
    """Register all callbacks with the app."""
    @app.callback(
        Output('dataset-selector_', 'options'),
        Input('tabs', 'value'),
        Input('working-directory-store', 'data')
    )
    def populate_dataset_selector(tab, base_dir):
        """Populate the dataset selector with available datasets."""
        if tab == 'tab-2':
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return [{'label': filename, 'value': filename} for filename in metadata.keys()]
            return []
        return []


    @app.callback(
        Output('preprocessed-graph', 'figure'),
        Output('dataset-status-display', 'children'),
        Input('dataset-selector_', 'value'),
        State('working-directory-store', 'data')
    )
    def display_dataset_and_status(dataset_name, base_dir):
        """Displays selected dataset as a chart and comprehensive status."""
        if not dataset_name:
            return {}, html.Div("Select a dataset to view status", style={'color': '#6c757d', 'font-style': 'italic'})

        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        dataset_info = metadata.get(dataset_name, {})

        # Determine processing stages
        stages = {
            'raw': True,  # Always true if dataset exists
            'preprocessed': 'cleaned_data_path' in dataset_info,
            'split': 'dragged_samples' in dataset_info and len(dataset_info.get('dragged_samples', [])) > 0,
            'training_ready': False  # Check for training data files
        }

        # Check if training data exists
        train_file = get_training_data_path(dataset_name, 'train', base_dir)
        test_file = get_training_data_path(dataset_name, 'test', base_dir)
        stages['training_ready'] = os.path.exists(
            train_file) and os.path.exists(test_file)

        # Create status display
        status_badges = []

        # Raw Data Status
        status_badges.append(
            html.Span("📁 Raw Data", className="badge", style={
                'background-color': '#6c757d',
                'color': 'white',
                'padding': '6px 12px',
                'border-radius': '12px',
                'margin-right': '8px',
                'margin-bottom': '8px',
                'font-size': '12px',
                'display': 'inline-block'
            })
        )

        # Signal Preprocessed Status
        if stages['preprocessed']:
            status_badges.append(
                html.Span("🔧 Signal Processed", className="badge", style={
                    'background-color': '#17a2b8',
                    'color': 'white',
                    'padding': '6px 12px',
                    'border-radius': '12px',
                    'margin-right': '8px',
                    'margin-bottom': '8px',
                    'font-size': '12px',
                    'display': 'inline-block'
                })
            )
        else:
            status_badges.append(
                html.Span("⏳ Signal Processing Pending", className="badge", style={
                    'background-color': '#ffc107',
                    'color': '#212529',
                    'padding': '6px 12px',
                    'border-radius': '12px',
                    'margin-right': '8px',
                    'margin-bottom': '8px',
                    'font-size': '12px',
                    'display': 'inline-block'
                })
            )

        # Split Status
        if stages['split']:
            # Count only files that actually exist on disk
            dragged_samples = dataset_info.get('dragged_samples', [])
            actual_split_count = sum(
                1 for file_path in dragged_samples if os.path.exists(file_path))

            status_badges.append(
                html.Span(f"✂️ Split ({actual_split_count} windows)", className="badge", style={
                    'background-color': '#fd7e14',
                    'color': 'white',
                    'padding': '6px 12px',
                    'border-radius': '12px',
                    'margin-right': '8px',
                    'margin-bottom': '8px',
                    'font-size': '12px',
                    'display': 'inline-block'
                })
            )
        else:
            status_badges.append(
                html.Span("⏳ Split Pending", className="badge", style={
                    'background-color': '#6c757d',
                    'color': 'white',
                    'padding': '6px 12px',
                    'border-radius': '12px',
                    'margin-right': '8px',
                    'margin-bottom': '8px',
                    'font-size': '12px',
                    'display': 'inline-block'
                })
            )

        # Training Data Status
        if stages['training_ready']:
            status_badges.append(
                html.Span("🚀 Training Ready", className="badge", style={
                    'background-color': '#28a745',
                    'color': 'white',
                    'padding': '6px 12px',
                    'border-radius': '12px',
                    'margin-right': '8px',
                    'margin-bottom': '8px',
                    'font-size': '12px',
                    'display': 'inline-block'
                })
            )
        else:
            status_badges.append(
                html.Span("⏳ Training Prep Pending", className="badge", style={
                    'background-color': '#6c757d',
                    'color': 'white',
                    'padding': '6px 12px',
                    'border-radius': '12px',
                    'margin-right': '8px',
                    'margin-bottom': '8px',
                    'font-size': '12px',
                    'display': 'inline-block'
                })
            )

        # Create comprehensive status display
        status_display = html.Div([
            html.H6("📊 Dataset Processing Status", style={
                'margin-bottom': '10px',
                'color': '#495057',
                'font-weight': 'bold'
            }),
            html.Div(status_badges, style={'line-height': '2.5'}),
            html.Hr(style={'margin': '15px 0'}),
            html.Div([
                html.Small(f"📁 Dataset: {dataset_name}", style={
                    'display': 'block', 'color': '#6c757d', 'margin-bottom': '5px'}),
                html.Small(f"📡 Sampling Rate: {dataset_info.get('sampling_rate', 'Unknown')} Hz", style={
                    'display': 'block', 'color': '#6c757d', 'margin-bottom': '5px'}),
                html.Small(f"🏷️ Activity: {dataset_info.get('label', 'Unknown')}", style={
                    'display': 'block', 'color': '#6c757d'})
            ])
        ], style={
            'background-color': '#f8f9fa',
            'padding': '15px',
            'border-radius': '8px',
            'border': '1px solid #dee2e6'
        })

        # Load and display the graph
        if stages['preprocessed']:
            file_path = dataset_info["cleaned_data_path"]
        else:
            file_path = dataset_info["path"]

        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Create time axis for proper labeling
            sampling_rate = dataset_info.get('sampling_rate', 100)
            df['Time_seconds'] = df.index / sampling_rate

            # Get sensor columns for plotting
            sensor_cols = [
                col for col in df.columns if col not in ['Time_seconds']]

            # Create figure with proper time axis
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                      '#d62728', '#9467bd', '#8c564b']

            # Limit to 6 sensors for clarity
            for i, col in enumerate(sensor_cols[:6]):
                fig.add_trace(
                    go.Scatter(
                        x=df['Time_seconds'],
                        y=df[col],
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=2),
                        hovertemplate=f'<b>{col}</b><br>Time: %{{x:.3f}}s<br>Value: %{{y:.3f}}<extra></extra>'
                    )
                )

            fig.update_layout(
                title=f"Preview of {dataset_name} ({'Signal Processed' if stages['preprocessed'] else 'Raw Data'})",
                xaxis_title="Time (seconds)",
                yaxis_title="Sensor Values",
                height=400,
                showlegend=True,
                hovermode='x unified'
            )

            return fig, status_display

        return {}, status_display


    @app.callback(
        Output('preprocessed-graph', 'figure', allow_duplicate=True),
        Output('stored-datasets', 'data', allow_duplicate=True),
        Input('clean-smooth-btn', 'n_clicks'),
        State('dataset-selector_', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def clean_and_smooth_data(n_clicks, dataset_name, base_dir):
        """Clean and smooth the selected dataset and display it in a graph."""
        if not dataset_name:
            return no_update, no_update

        if not base_dir:
            base_dir = PERSISTENT_DIR
        datasets_dir = os.path.join(base_dir, 'datasets')

        file_path = os.path.join(datasets_dir, dataset_name)
        if not os.path.exists(file_path):
            return no_update, no_update

        df = pd.read_csv(file_path)

        # Clean the data
        df = clean_data(df, method='remove_missing')
        df = clean_data(df, method='filter_outliers')

        # Apply low-pass filter
        df = low_pass_filter(df, cutoff=5, fs=50, order=2)

        # Apply Savitzky-Golay filter
        for col in df.select_dtypes(include=['float64', 'int64']).columns:
            df[col] = savgol_filter(df[col], window_length=5, polyorder=2)

        # Create time axis for proper labeling
        sampling_rate = 100  # Default sampling rate
        df['Time_seconds'] = df.index / sampling_rate

        # Get sensor columns for plotting
        sensor_cols = [col for col in df.columns if col not in ['Time_seconds']]

        # Create figure with proper time axis
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        for i, col in enumerate(sensor_cols[:6]):  # Limit to 6 sensors for clarity
            fig.add_trace(
                go.Scatter(
                    x=df['Time_seconds'],
                    y=df[col],
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{col}</b><br>Time: %{{x:.3f}}s<br>Value: %{{y:.3f}}<extra></extra>'
                )
            )

        fig.update_layout(
            title=f"Cleaned & Smoothed Data: {dataset_name}",
            xaxis_title="Time (seconds)",
            yaxis_title="Sensor Values",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )

        stored_datasets = df.to_dict(orient='records')

        return fig, stored_datasets


    @app.callback(
        Output('preprocessed-graph', 'figure', True),
        Input('save-cleaned-smoothed-btn', 'n_clicks'),
        State('dataset-selector_', 'value'),
        State('stored-datasets', 'data'),
        State('preprocessed-graph', 'figure'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def save_cleaned_smoothed_data(n_clicks, dataset_name, cleaned_smoothed, processed_figure, base_dir):
        """Save the cleaned and smoothed dataset to a new file and update metadata."""
        if not (dataset_name and processed_figure):
            return {}

        if not base_dir:
            base_dir = PERSISTENT_DIR
        datasets_dir = os.path.join(base_dir, 'datasets')

        df = pd.DataFrame(cleaned_smoothed)
        cleaned_smoothed_file_path = os.path.join(
            datasets_dir, f"cleaned_smoothed_{dataset_name}")
        # Use 4 decimal places precision for sensor data readability
        df.to_csv(cleaned_smoothed_file_path, index=False, float_format='%.4f')

        # Update metadata
        metadata_file = os.path.join(base_dir, "metadata.json")
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        metadata[dataset_name]["cleaned_data_path"] = cleaned_smoothed_file_path

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        return processed_figure


    @app.callback(
        Output('interactive-sample-graph', 'figure'),
        Output('current-windows', 'data', allow_duplicate=True),
        Input('apply-time-window-btn', 'n_clicks'),
        State('dataset-selector_', 'value'),
        State('time-window-span-input', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def apply_time_window(n_clicks, dataset_name, time_window_span, base_dir):
        """Apply the time window span and display the dataset with draggable windows."""
        if not (dataset_name and time_window_span):
            return {}, []

        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Get the correct file path
        if "cleaned_data_path" in metadata[dataset_name]:
            file_path = metadata[dataset_name]["cleaned_data_path"]
        else:
            file_path = metadata[dataset_name]["path"]

        if not os.path.exists(file_path):
            return {}

        df = pd.read_csv(file_path)
        sampling_rate = metadata.get(dataset_name, {}).get('sampling_rate', 100)
        rows_per_window = int((time_window_span / 1000) * sampling_rate)

        # Create time axis
        df['Time_seconds'] = df.index / sampling_rate

        # Get sensor columns
        sensor_cols = [col for col in df.columns if col not in [
            'Time_seconds', 'Window']]
        priority_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        available_cols = [col for col in priority_cols if col in sensor_cols]

        if not available_cols:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            available_cols = [
                col for col in numerical_cols if col != 'Time_seconds'][:6]

        if not available_cols:
            return {}

        # Create the master combined plot
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        # Fix: Use solid lines instead of dashed lines for better visibility
        line_styles = ['solid', 'solid', 'solid', 'solid', 'solid', 'solid']

        for i, col in enumerate(available_cols[:6]):
            fig.add_trace(
                go.Scatter(
                    x=df['Time_seconds'],
                    y=df[col],
                    name=col,
                    line=dict(
                        color=colors[i % len(colors)],
                        width=2,
                        dash=line_styles[i % len(line_styles)]
                    ),
                    hovertemplate=f'<b>{col}</b><br>Time: %{{x:.3f}}s<br>Value: %{{y:.3f}}<extra></extra>'
                )
            )

        # Calculate window parameters
        total_time = df['Time_seconds'].max()
        window_duration = time_window_span / 1000
        max_windows = min(int(total_time / window_duration) + 1, 8)

        # Get data range for rectangle sizing
        all_values = df[available_cols].values.flatten()
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        rect_y_min = y_min - 0.05 * y_range
        rect_y_max = y_max + 0.05 * y_range

        # Create initial windows data for storage
        initial_windows = []
        for window_idx in range(max_windows):
            window_start = window_idx * window_duration
            window_end = min(window_start + window_duration, total_time)
            initial_windows.append({
                'window_id': window_idx + 1,
                'start_time': window_start,
                'end_time': window_end,
                'y_min': rect_y_min,
                'y_max': rect_y_max
            })

        # Add draggable window rectangles with improved styling
        shapes = []
        for window_idx in range(max_windows):
            window_start = window_idx * window_duration
            window_end = window_start + window_duration

            # Create enhanced shape with superior visibility and draggability
            shape = dict(
                type="rect",
                x0=window_start, y0=rect_y_min,
                x1=window_end, y1=rect_y_max,
                # Enhanced opacity for better visibility
                fillcolor="rgba(255, 80, 80, 0.35)",
                line=dict(
                    color="rgb(220, 20, 20)",
                    width=4,  # Thicker border for better grabbing
                    dash="solid"
                ),
                editable=True,
                name=f"window_{window_idx + 1}",
                layer="above",  # Ensure rectangles are above data lines
                # Enhanced label configuration for better visibility
                label=dict(
                    text=f"W{window_idx + 1}",
                    textposition="middle center",
                    font=dict(size=14, color="red", family="Arial Black")
                )
            )
            shapes.append(shape)

            # Add annotation for window label (revert to original style)
            # fig.add_annotation(
            #     x=window_start + window_duration/2,
            #     y=rect_y_max - 0.02 * y_range,
            #     text=f"<b>W{window_idx}</b>",
            #     showarrow=False,
            #     font=dict(size=12, color="red", family="Arial Black"),
            #     bgcolor="rgba(255,255,255,0.8)",
            #     bordercolor="red",
            #     borderwidth=1
            # )

        # Add all shapes to the figure
        fig.update_layout(shapes=shapes)

        # Update layout with enhanced configuration for horizontal scrolling and constrained movement
        fig.update_layout(
            title=dict(
                text=f"Interactive Time-Series Data with Horizontally Draggable Windows - {dataset_name}<br>"
                f"Window Size: {time_window_span}ms ({rows_per_window} samples), "
                f"Sampling Rate: {sampling_rate}Hz",
                font=dict(size=16, color='#2E86AB'),
                x=0.5
            ),
            xaxis=dict(
                title="Time (seconds)",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                showspikes=True,
                spikecolor="red",
                spikethickness=2,
                spikedash="dot",
                spikemode="across",
                # Enable horizontal scrolling by extending the range beyond visible area
                range=[0, min(total_time, 20)],  # Show first 20 seconds initially
                rangeslider=dict(
                    visible=True,
                    thickness=0.05,
                    bgcolor="rgba(240,248,255,0.8)",
                    bordercolor="#2E86AB",
                    borderwidth=1
                ),
                # Allow zooming and panning
                fixedrange=False
            ),
            yaxis=dict(
                title="Sensor Values",
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.2)',
                showspikes=True,
                spikecolor="red",
                spikethickness=2,
                spikedash="dot",
                spikemode="across",
                # Fix Y-axis range to prevent vertical movement of windows
                fixedrange=True,
                range=[rect_y_min - 0.1 * y_range, rect_y_max + 0.1 * y_range]
            ),
            height=800,  # Increased height to accommodate range slider
            width=1400,  # Full width for better horizontal scrolling
            showlegend=True,
            hovermode='x unified',
            dragmode='pan',  # Pan mode for easier navigation
            # Enhanced newshape configuration
            newshape=dict(
                fillcolor="rgba(255, 100, 100, 0.3)",
                line=dict(color="rgb(255, 0, 0)", width=3),
                opacity=0.9
            ),
            # Enhanced modebar with horizontal navigation tools
            modebar=dict(
                bgcolor='rgba(255,255,255,0.8)',
                color='#2E86AB',
                activecolor='#ff6b35',
                orientation='h'
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="rgba(128,128,128,0.5)",
                borderwidth=1
            ),
            # Enhanced margin for range slider
            margin=dict(l=80, r=80, t=140, b=120),
            # Configure for horizontal-only window movement
            selectdirection='h',  # 'h' for horizontal selection only
            uirevision=True,  # Preserve user interactions
            # Add plot background styling
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='white'
        )

        # Add comprehensive user instructions for horizontal scrolling interface
        instruction_text = (
            "🎯 <b>Horizontal Time-Series Navigation - Professional Mode:</b><br><br>"
            "📏 <b>Horizontal Navigation:</b><br>"
            "• <b>Time Scroll:</b> Use range slider at bottom to scroll through time<br>"
            "• <b>Zoom:</b> Mouse wheel or zoom tools for temporal precision<br>"
            "• <b>Pan:</b> Click and drag background horizontally<br>"
            "• <b>Reset View:</b> Double-click or use 'Reset axes' button<br><br>"
            "🎮 <b>Window Movement (Horizontal Only):</b><br>"
            "• <b>Drag Windows:</b> Click and drag red rectangles left/right only<br>"
            "• <b>Constrained:</b> Windows move horizontally along time axis<br>"
            "• <b>Precision:</b> Zoom in for fine temporal positioning<br>"
            "• <b>Labels:</b> Window IDs (W0, W1...) embedded in rectangles<br><br>"
            "🛠️ <b>Advanced Time Tools:</b><br>"
            "• <b>Range Slider:</b> Navigate entire time series at bottom<br>"
            "• <b>Grid Lines:</b> Temporal grid for precise alignment<br>"
            "• <b>Crosshairs:</b> Time alignment aids when hovering<br>"
            "• <b>Add/Remove:</b> Use control buttons for window management<br><br>"
            "💡 <b>Time-Series Pro Tips:</b><br>"
            "• Scroll to find interesting temporal patterns • Windows snap to time grid<br>"
            "• Use range slider for quick navigation • Zoom for microsecond precision<br>"
            "• Vertical position fixed - focus on temporal placement"
        )

        fig.add_annotation(
            text=instruction_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=11, color="#1a365d", family="Arial"),
            bgcolor="rgba(240,248,255,0.95)",
            bordercolor="#2E86AB",
            borderwidth=2,
            align="left",
            valign="top"
        )

        return fig, initial_windows


    @app.callback(
        Output('current-windows', 'data'),
        Output('interactive-sample-graph', 'figure', allow_duplicate=True),
        Input('add-window-btn', 'n_clicks'),
        Input('remove-window-btn', 'n_clicks'),
        Input('reset-windows-btn', 'n_clicks'),
        State('current-windows', 'data'),
        State('dataset-selector_', 'value'),
        State('time-window-span-input', 'value'),
        State('interactive-sample-graph', 'figure'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def manage_windows(add_clicks, remove_clicks, reset_clicks, current_windows, dataset_name, time_window_span, current_figure, base_dir):
        """Manage adding, removing, and resetting windows."""
        from dash import ctx

        if not (dataset_name and time_window_span and current_figure):
            return current_windows or [], current_figure or {}

        # Get which button was clicked
        button_id = ctx.triggered[0]['prop_id'].split(
            '.')[0] if ctx.triggered else None

        # Ensure current_windows is a list
        if not current_windows:
            current_windows = []

        # Load dataset info for window calculations
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        if "cleaned_data_path" in metadata[dataset_name]:
            file_path = metadata[dataset_name]["cleaned_data_path"]
        else:
            file_path = metadata[dataset_name]["path"]

        if not os.path.exists(file_path):
            return current_windows, current_figure

        df = pd.read_csv(file_path)
        sampling_rate = metadata.get(dataset_name, {}).get('sampling_rate', 100)
        df['Time_seconds'] = df.index / sampling_rate

        total_time = df['Time_seconds'].max()
        window_duration = time_window_span / 1000

        # Get sensor columns for y-axis range
        sensor_cols = [col for col in df.columns if col not in [
            'Time_seconds', 'Window']]
        priority_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        available_cols = [col for col in priority_cols if col in sensor_cols]

        if not available_cols:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            available_cols = [
                col for col in numerical_cols if col != 'Time_seconds'][:6]

        if not available_cols:
            return current_windows, current_figure

        all_values = df[available_cols].values.flatten()
        y_min, y_max = all_values.min(), all_values.max()
        y_range = y_max - y_min
        rect_y_min = y_min - 0.05 * y_range
        rect_y_max = y_max + 0.05 * y_range

        if button_id == 'add-window-btn':
            # If no current windows, initialize with existing figure shapes first
            if not current_windows and current_figure and 'layout' in current_figure and 'shapes' in current_figure['layout']:
                # Extract existing windows from figure
                shapes = current_figure['layout']['shapes']
                for idx, shape in enumerate(shapes):
                    if shape.get('type') == 'rect' and 'x0' in shape and 'x1' in shape:
                        current_windows.append({
                            'window_id': idx,
                            'start_time': shape['x0'],
                            'end_time': shape['x1'],
                            'y_min': rect_y_min,
                            'y_max': rect_y_max
                        })

            # Add a new window at the end or at a good position
            if current_windows:
                # Place new window after the last one
                last_window = max(current_windows, key=lambda w: w['start_time'])
                new_start = min(
                    last_window['start_time'] + window_duration, total_time - window_duration)
            else:
                # First window at the beginning
                new_start = 0

            new_end = min(new_start + window_duration, total_time)
            new_window = {
                'window_id': len(current_windows) + 1,
                'start_time': new_start,
                'end_time': new_end,
                'y_min': rect_y_min,
                'y_max': rect_y_max
            }
            current_windows.append(new_window)

        elif button_id == 'remove-window-btn':
            # Remove the last window
            if current_windows:
                current_windows.pop()

        elif button_id == 'reset-windows-btn':
            # Reset to initial windows
            max_windows = min(int(total_time / window_duration) + 1, 8)
            current_windows = []
            for window_idx in range(max_windows):
                window_start = window_idx * window_duration
                window_end = min(window_start + window_duration, total_time)
                current_windows.append({
                    'window_id': window_idx + 1,
                    'start_time': window_start,
                    'end_time': window_end,
                    'y_min': rect_y_min,
                    'y_max': rect_y_max
                })

        # Update the figure with new windows
        updated_figure = update_figure_windows(
            current_figure, current_windows, window_duration, y_range)

        return current_windows, updated_figure


    def update_figure_windows(figure, windows, window_duration, y_range):
        """Update the figure with the current windows configuration."""
        if not figure or 'layout' not in figure:
            return figure

        # Create new shapes for the windows
        shapes = []

        for window in windows:
            # Create enhanced rectangle shape with superior visibility
            shape = dict(
                type="rect",
                x0=window['start_time'],
                y0=window['y_min'],
                x1=window['end_time'],
                y1=window['y_max'],
                # Enhanced opacity for better visibility
                fillcolor="rgba(255, 80, 80, 0.35)",
                line=dict(
                    color="rgb(220, 20, 20)",
                    width=4,  # Thicker border for better grabbing
                    dash="solid"
                ),
                editable=True,
                name=f"window_{window['window_id']}",
                layer="above",  # Ensure rectangles are above data lines
                # Enhanced label configuration for better visibility
                label=dict(
                    text=f"W{window['window_id']}",
                    textposition="middle center",
                    font=dict(size=14, color="red", family="Arial Black")
                )
            )
            shapes.append(shape)

            # Add window label (revert to original style)
            # figure['data'].append(dict(
            #     x=[window['start_time'] + (window['end_time'] - window['start_time']) / 2],
            #     y=[window['y_max'] - 0.02 * y_range],
            #     text=[f"<b>W{window['window_id']}</b>"],
            #     mode='text',
            #     textfont=dict(size=12, color="red", family="Arial Black"),
            #     showlegend=False
            # ))

        # Update figure layout with new shapes
        figure['layout']['shapes'] = shapes

        return figure


    @app.callback(
        [Output('interactive-sample-graph', 'figure', allow_duplicate=True),
         Output('current-windows', 'data', allow_duplicate=True)],
        Input('load-previous-windows-btn', 'n_clicks'),
        [State('dataset-selector_', 'value'),
         State('interactive-sample-graph', 'figure'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def load_previous_windows(n_clicks, dataset_name, current_figure, base_dir):
        """Load previously saved window positions from metadata."""
        if not dataset_name:
            print("Load Previous: No dataset selected")
            return no_update, no_update

        try:
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if dataset_name not in metadata:
                print(
                    f"Load Previous: Dataset {dataset_name} not found in metadata")
                return no_update, no_update

            # Check for saved window positions (manual windows take priority)
            saved_positions = metadata[dataset_name].get(
                'manual_window_positions', [])

            if not saved_positions:
                # Fall back to sliding window positions if no manual windows
                saved_positions = metadata[dataset_name].get(
                    'sliding_window_positions', [])

            if not saved_positions:
                print(
                    f"Load Previous: No saved window positions for {dataset_name}")
                return no_update, no_update

            # Get saved window size
            window_size_ms = metadata[dataset_name].get('window_size_ms', 1500)

            # Load the dataset to create the graph
            if "cleaned_data_path" in metadata[dataset_name]:
                file_path = metadata[dataset_name]["cleaned_data_path"]
            else:
                file_path = metadata[dataset_name]["path"]

            if not os.path.exists(file_path):
                print(f"Load Previous: Dataset file not found: {file_path}")
                return no_update, no_update

            df = pd.read_csv(file_path)
            sampling_rate = metadata.get(
                dataset_name, {}).get('sampling_rate', 100)

            # Create time axis
            df['Time_seconds'] = df.index / sampling_rate

            # If we don't have a figure yet, create one
            if not current_figure or 'data' not in current_figure:
                # Get sensor columns
                sensor_cols = [col for col in df.columns if col not in [
                    'Time_seconds', 'Window']]
                priority_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
                available_cols = [
                    col for col in priority_cols if col in sensor_cols]

                if not available_cols:
                    numerical_cols = df.select_dtypes(
                        include=['float64', 'int64']).columns
                    available_cols = [
                        col for col in numerical_cols if col != 'Time_seconds'][:6]

                # Create the figure
                fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                          '#d62728', '#9467bd', '#8c564b']

                for i, col in enumerate(available_cols[:6]):
                    fig.add_trace(go.Scatter(
                        x=df['Time_seconds'],
                        y=df[col],
                        mode='lines',
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ))

                # Get data range
                all_values = df[available_cols].values.flatten()
                y_min, y_max = np.nanmin(all_values), np.nanmax(all_values)
                y_range = y_max - y_min
                rect_y_min = y_min - 0.05 * y_range
                rect_y_max = y_max + 0.05 * y_range
            else:
                # Use existing figure
                fig = current_figure.copy()

                # Get y-axis range from existing figure or calculate from data
                if 'layout' in fig and 'yaxis' in fig['layout'] and 'range' in fig['layout']['yaxis']:
                    y_range_data = fig['layout']['yaxis']['range']
                    rect_y_min, rect_y_max = y_range_data[0], y_range_data[1]
                else:
                    # Calculate from figure data
                    y_data = []
                    for trace in fig.get('data', []):
                        if 'y' in trace:
                            y_data.extend(trace['y'])

                    if y_data:
                        y_min, y_max = min(y_data), max(y_data)
                        y_range = y_max - y_min
                        rect_y_min = y_min - 0.1 * y_range
                        rect_y_max = y_max + 0.1 * y_range
                    else:
                        rect_y_min, rect_y_max = -10, 10

            # Create window data and shapes
            windows = []
            shapes = []

            for idx, pos in enumerate(saved_positions):
                window = {
                    'window_id': idx,  # Use sequential IDs
                    'start_time': float(pos['start_time']),
                    'end_time': float(pos['end_time']),
                    'y_min': rect_y_min,
                    'y_max': rect_y_max
                }
                windows.append(window)

                shape = dict(
                    type="rect",
                    x0=float(pos['start_time']),
                    y0=rect_y_min,
                    x1=float(pos['end_time']),
                    y1=rect_y_max,
                    fillcolor="rgba(255, 80, 80, 0.35)",
                    line=dict(color="rgb(220, 20, 20)", width=4, dash="solid"),
                    editable=True,
                    name=f"window_{idx}",
                    layer="above",
                    label=dict(
                        text=f"W{idx}",
                        textposition="middle center",
                        font=dict(size=14, color="red", family="Arial Black")
                    )
                )
                shapes.append(shape)

            # Update figure layout
            if 'layout' not in fig:
                fig['layout'] = {}
            fig['layout']['shapes'] = shapes

            print(
                f"Load Previous: Successfully loaded {len(windows)} windows for {dataset_name}")
            return fig, windows

        except Exception as e:
            import traceback
            print(f"Error loading previous windows: {e}")
            print(traceback.format_exc())
            return no_update, no_update


    @app.callback(
        Output('current-windows', 'data', allow_duplicate=True),
        Input('interactive-sample-graph', 'relayoutData'),
        State('current-windows', 'data'),
        prevent_initial_call=True
    )
    def update_window_positions(relayout_data, current_windows):
        """Update window positions when shapes are dragged by the user."""
        if not (relayout_data and current_windows):
            return no_update

        # Check if shapes were edited/moved
        shapes_edited = any(key.startswith('shapes[') and ('x0' in key or 'x1' in key)
                            for key in relayout_data.keys()) if relayout_data else False

        if shapes_edited:
            updated_windows = current_windows.copy()

            # Update window positions based on relayout data
            for key, value in relayout_data.items():
                if key.startswith('shapes[') and ('x0' in key or 'x1' in key):
                    # Extract shape index from key like "shapes[0].x0"
                    import re
                    match = re.search(r'shapes\[(\d+)\]\.([xy][01])', key)
                    if match:
                        shape_idx = int(match.group(1))
                        coord_type = match.group(2)

                        # Ensure we have this window in our data
                        if shape_idx < len(updated_windows):
                            if coord_type == 'x0':
                                updated_windows[shape_idx]['start_time'] = float(
                                    value)
                            elif coord_type == 'x1':
                                updated_windows[shape_idx]['end_time'] = float(
                                    value)

            print(f"Updated window positions: {updated_windows}")
            return updated_windows

        return no_update


    @app.callback(
        Output('interactive-sample-graph', 'figure', allow_duplicate=True),
        Input('interactive-sample-graph', 'relayoutData'),
        State('interactive-sample-graph', 'figure'),
        State('current-windows', 'data'),
        prevent_initial_call=True
    )
    def constrain_window_movement(relayout_data, current_figure, current_windows):
        """Constrain window movement to horizontal only by resetting Y coordinates."""
        if not (relayout_data and current_figure and current_windows):
            return no_update

        # Check if shapes were edited/moved
        shapes_edited = any(key.startswith(
            'shapes[') for key in relayout_data.keys()) if relayout_data else False

        if shapes_edited:
            updated_figure = current_figure.copy()
            shapes = updated_figure.get('layout', {}).get('shapes', [])

            # Reset Y coordinates for all shapes to maintain vertical constraint
            for i, shape in enumerate(shapes):
                if i < len(current_windows):
                    window = current_windows[i]
                    # Force Y coordinates to stay at original positions
                    shape['y0'] = window['y_min']
                    shape['y1'] = window['y_max']

            return updated_figure

        return no_update


    @app.callback(
        Output('overlap-info', 'children'),
        Input('overlap-percentage-slider', 'value'),
        State('time-window-span-input', 'value')
    )
    def update_overlap_info(overlap_percent, window_size_ms):
        """Display information about overlap settings."""
        if not window_size_ms:
            return ""

        # Calculate window parameters
        sampling_rate = 100  # Hz
        window_samples = int((window_size_ms / 1000) * sampling_rate)
        stride = int(window_samples * (1 - overlap_percent / 100))

        # Estimate multiplication factor
        if overlap_percent == 0:
            factor = "1x"
        else:
            factor = f"{1 / (1 - overlap_percent / 100):.1f}x"

        return html.Div([
            html.Span(f"📊 Stride: {stride} samples | ", style={
                      'color': '#007bff', 'font-weight': 'bold'}),
            html.Span(f"Sample increase: ~{factor}", style={
                      'color': '#28a745', 'font-weight': 'bold'}),
            html.Br(),
            html.Span(f"Example: 10 windows → ~{int(10 * float(factor[:-1]))} windows with {overlap_percent}% overlap",
                      style={'font-size': '12px', 'color': '#666', 'font-style': 'italic'})
        ])


    @app.callback(
        [Output('sliding-windows-preview', 'children'),
         Output('sliding-windows-data', 'data'),
         Output('save-sliding-windows-btn', 'disabled')],
        Input('generate-sliding-windows-btn', 'n_clicks'),
        [State('dataset-selector_', 'value'),
         State('current-windows', 'data'),
         State('time-window-span-input', 'value'),
         State('overlap-percentage-slider', 'value'),
         State('quality-threshold-slider', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def generate_sliding_windows(n_clicks, dataset_name, current_windows, window_size_ms,
                                 overlap_percent, quality_threshold, base_dir):
        """Generate sliding windows from manually selected regions."""
        if not (dataset_name and current_windows and window_size_ms):
            return html.Div("⚠️ Please select a dataset and define windows first.",
                            style={'color': '#FF9800', 'padding': '20px'}), None, True

        # Ensure all window times are floats
        current_windows = [
            {
                **w,
                'start_time': float(w['start_time']) if not isinstance(w['start_time'], (int, float)) else w['start_time'],
                'end_time': float(w['end_time']) if not isinstance(w['end_time'], (int, float)) else w['end_time']
            }
            for w in current_windows
        ]

        try:
            # Load data
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if "cleaned_data_path" in metadata[dataset_name]:
                file_path = metadata[dataset_name]["cleaned_data_path"]
            else:
                file_path = metadata[dataset_name]["path"]

            if not os.path.exists(file_path):
                return html.Div("❌ Dataset file not found.", style={'color': '#dc3545'}), None, True

            df = pd.read_csv(file_path)
            sampling_rate = metadata.get(
                dataset_name, {}).get('sampling_rate', 100)

            # Create time axis if not present
            if 'Time_seconds' not in df.columns:
                df['Time_seconds'] = df.index / sampling_rate

            window_samples = int((window_size_ms / 1000) * sampling_rate)

            # Generate sliding windows
            good_windows, flagged_windows, stats = generate_sliding_windows_from_current(
                current_windows, df, window_samples, overlap_percent, quality_threshold
            )

            if not good_windows and not flagged_windows:
                return html.Div("⚠️ No windows generated. Try reducing quality threshold.",
                                style={'color': '#FF9800', 'padding': '20px'}), None, True

            # Create quality distribution visualization
            quality_scores = [w['quality'] for w in good_windows + flagged_windows]
            quality_fig = go.Figure()

            quality_fig.add_trace(go.Histogram(
                x=quality_scores,
                nbinsx=20,
                marker_color='#007bff',
                name='Quality Distribution'
            ))

            quality_fig.add_vline(
                x=quality_threshold,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Threshold ({quality_threshold})",
                annotation_position="top right"
            )

            quality_fig.update_layout(
                title='Window Quality Score Distribution',
                xaxis_title='Quality Score',
                yaxis_title='Count',
                height=300,
                showlegend=False
            )

            # Create a preview visualization of the first few windows
            preview_sample_count = min(3, len(good_windows))
            if preview_sample_count > 0:
                preview_data = pd.concat([good_windows[i]['data'] for i in range(
                    preview_sample_count)], ignore_index=True)
                sensor_cols = [col for col in preview_data.columns if col in [
                    'aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']]

                preview_fig = go.Figure()
                colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                          '#d62728', '#9467bd', '#8c564b']

                for i, col in enumerate(sensor_cols[:6]):
                    preview_fig.add_trace(go.Scatter(
                        x=preview_data.index,
                        y=preview_data[col],
                        mode='lines',
                        name=col,
                        line=dict(color=colors[i % len(colors)], width=1.5)
                    ))

                preview_fig.update_layout(
                    title=f"Preview - First {preview_sample_count} Windows",
                    xaxis_title="Sample Index",
                    yaxis_title="Sensor Values",
                    height=250,
                    hovermode='x unified',
                    margin=dict(t=40, b=40)
                )
            else:
                preview_fig = None

            # Create preview content
            preview = html.Div([
                html.H5("✅ Sliding Windows Generated Successfully!", style={
                    'color': '#28a745', 'margin-bottom': '15px'
                }),

                # Statistics cards
                html.Div([
                    html.Div([
                        html.H2(str(stats['good_windows']), style={
                                'color': '#28a745', 'margin': '0'}),
                        html.P("High Quality Windows", style={
                               'color': '#666', 'margin': '5px 0'})
                    ], style={
                        'display': 'inline-block', 'width': '23%', 'text-align': 'center',
                        'background': 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)',
                        'padding': '15px', 'border-radius': '8px', 'margin-right': '2%'
                    }),

                    html.Div([
                        html.H2(str(stats['flagged_windows']), style={
                                'color': '#ffc107', 'margin': '0'}),
                        html.P("Flagged (Low Quality)", style={
                               'color': '#666', 'margin': '5px 0'})
                    ], style={
                        'display': 'inline-block', 'width': '23%', 'text-align': 'center',
                        'background': 'linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%)',
                        'padding': '15px', 'border-radius': '8px', 'margin-right': '2%'
                    }),

                    html.Div([
                        html.H2(f"{stats['avg_quality']:.2f}", style={
                                'color': '#007bff', 'margin': '0'}),
                        html.P("Average Quality", style={
                               'color': '#666', 'margin': '5px 0'})
                    ], style={
                        'display': 'inline-block', 'width': '23%', 'text-align': 'center',
                        'background': 'linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%)',
                        'padding': '15px', 'border-radius': '8px', 'margin-right': '2%'
                    }),

                    html.Div([
                        html.H2(f"{stats['overlap_percent']}%", style={
                                'color': '#6c757d', 'margin': '0'}),
                        html.P("Overlap Used", style={
                               'color': '#666', 'margin': '5px 0'})
                    ], style={
                        'display': 'inline-block', 'width': '23%', 'text-align': 'center',
                        'background': 'linear-gradient(135deg, #e2e3e5 0%, #d6d8db 100%)',
                        'padding': '15px', 'border-radius': '8px'
                    })
                ], style={'margin-bottom': '20px'}),

                # Quality distribution chart
                dcc.Graph(figure=quality_fig, config={'displayModeBar': False}),

                # Preview graph
                dcc.Graph(figure=preview_fig, config={
                          'displayModeBar': False}) if preview_fig else html.Div(),

                # Details
                html.Div([
                    html.P([
                        html.Strong("Configuration: "),
                        f"Window size: {window_size_ms}ms ({window_samples} samples) | ",
                        f"Stride: {stats['stride_samples']} samples | ",
                        f"Quality threshold: {quality_threshold}"
                    ], style={'margin-bottom': '10px'}),
                    html.P([
                        html.Strong("Improvement: "),
                        f"From {len(current_windows)} manual windows → {stats['good_windows']} high-quality windows ",
                        f"({stats['good_windows'] / len(current_windows):.1f}x increase)"
                    ], style={'color': '#28a745', 'font-weight': 'bold'}),
                ], style={
                    'background-color': '#f8f9fa',
                    'padding': '15px',
                    'border-radius': '6px',
                    'margin-top': '15px'
                }),

                # Action reminder
                html.Div([
                    html.P([
                        html.Strong("📌 Next Step: "),
                        "Click ",
                        html.Strong("'💾 Save Generated Windows'"),
                        " below to save these windows and view them in the Split Results graph."
                    ], style={'margin': '0'})
                ], style={
                    'background-color': '#d1ecf1',
                    'border-left': '4px solid #17a2b8',
                    'padding': '15px',
                    'border-radius': '6px',
                    'margin-top': '15px'
                }),

                # Warning for flagged windows
                html.Div([
                    html.P([
                        html.Strong("⚠️ Note: "),
                        f"{stats['flagged_windows']} windows were flagged for quality issues and excluded. ",
                        "Lower the quality threshold to include them."
                    ], style={'margin': '0'})
                ], style={
                    'background-color': '#fff3cd',
                    'border-left': '4px solid #ffc107',
                    'padding': '15px',
                    'border-radius': '6px',
                    'margin-top': '15px'
                }) if stats['flagged_windows'] > 0 else html.Div()
            ])

            # Prepare data for storage
            window_data = {
                'good_windows': [
                    {
                        'data': w['data'].to_dict('records'),
                        'start_time': w['start_time'],
                        'end_time': w['end_time'],
                        'quality': w['quality']
                    }
                    for w in good_windows
                ],
                'dataset_name': dataset_name,
                'window_size_ms': window_size_ms,
                'overlap_percent': overlap_percent,
                'stats': stats
            }

            return preview, window_data, False

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in generate_sliding_windows: {error_details}")
            return html.Div([
                html.H5("❌ Error generating windows", style={'color': '#dc3545'}),
                html.P(str(e))
            ]), None, True


    @app.callback(
        Output('split-samples-graph', 'figure'),
        Output('sliding-windows-preview', 'children', allow_duplicate=True),
        Output('sliding-windows-data', 'data', allow_duplicate=True),
        Input('save-sliding-windows-btn', 'n_clicks'),
        State('sliding-windows-data', 'data'),
        State('dataset-selector_', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def save_sliding_windows(n_clicks, window_data, dataset_name, base_dir):
        """Save generated sliding windows to disk."""
        if not window_data or not dataset_name:
            return no_update, no_update, no_update

        try:
            good_windows = window_data['good_windows']

            sample_files = []
            all_selected_data = []

            # Save each generated window
            for idx, window_info in enumerate(good_windows):
                window_df = pd.DataFrame(window_info['data'])

                # Extract sensor columns
                sensor_cols = [col for col in window_df.columns if col in [
                    'aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']]

                # Generate unique filename
                window_id = f"sliding_{idx}"
                sample_file_path = get_window_path(window_id, dataset_name)

                # Save window data
                window_df[sensor_cols].to_csv(
                    sample_file_path, index=False, float_format='%.4f')

                sample_files.append(sample_file_path)
                all_selected_data.append(window_df)

            # Update metadata
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Clean up old sliding window files first
            if 'dragged_samples' in metadata[dataset_name]:
                old_files = metadata[dataset_name]['dragged_samples']
                for old_file in old_files:
                    # Only remove sliding windows, not manual dragged windows
                    if 'sliding_' in old_file and os.path.exists(old_file):
                        try:
                            os.remove(old_file)
                            print(f"Removed old sliding window file: {old_file}")
                        except Exception as e:
                            print(f"Could not remove {old_file}: {e}")

                # Keep only manual dragged window files in metadata
                metadata[dataset_name]['dragged_samples'] = [
                    f for f in old_files if 'dragged_window_' in f
                ]
            else:
                metadata[dataset_name]['dragged_samples'] = []

            # Add new sliding window files
            metadata[dataset_name]['dragged_samples'].extend(sample_files)

            # Save sliding window positions for reload
            metadata[dataset_name]['sliding_window_positions'] = [
                {
                    'window_id': f"sliding_{idx}",
                    'start_time': w['start_time'],
                    'end_time': w['end_time']
                }
                for idx, w in enumerate(good_windows)
            ]

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create visualization of all windows
            combined_data = pd.concat(all_selected_data, ignore_index=True)
            sensor_cols = [col for col in combined_data.columns if col in [
                'aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']]
            available_cols = sensor_cols[:6]

            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                      '#d62728', '#9467bd', '#8c564b']

            for i, col in enumerate(available_cols):
                fig.add_trace(go.Scatter(
                    x=combined_data.index,
                    y=combined_data[col],
                    mode='lines',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=1.5)
                ))

            fig.update_layout(
                title=f"Saved Sliding Windows Preview - {len(good_windows)} Windows",
                xaxis_title="Sample Index",
                yaxis_title="Sensor Values",
                height=500,
                hovermode='x unified'
            )

            success_message = html.Div([
                html.H5("💾 Windows Saved Successfully!", style={
                        'color': '#28a745', 'margin-bottom': '15px'}),
                html.P([
                    html.Strong(f"{len(good_windows)} windows"),
                    f" saved to: persistent_data/"
                ]),
                html.P([
                    "These windows are now available in the ",
                    html.Strong("Training Data Preparation"),
                    " section below."
                ], style={'color': '#666', 'font-style': 'italic'})
            ], style={
                'background-color': '#d4edda',
                'border-left': '4px solid #28a745',
                'padding': '20px',
                'border-radius': '6px'
            })

            return fig, success_message, None

        except Exception as e:
            import traceback
            print(f"ERROR in save_sliding_windows: {traceback.format_exc()}")
            error_msg = html.Div([
                html.H5("❌ Error saving windows", style={'color': '#dc3545'}),
                html.P(str(e))
            ])
            return no_update, error_msg, window_data


    @app.callback(
        Output('split-samples-graph', 'figure', allow_duplicate=True),
        Input('split-selected-windows-btn', 'n_clicks'),
        State('dataset-selector_', 'value'),
        State('current-windows', 'data'),
        State('interactive-sample-graph', 'figure'),
        State('time-window-span-input', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def split_selected_windows(n_clicks, dataset_name, current_windows, current_figure, time_window_span, base_dir):
        """Split the dataset into samples based on current window positions."""
        if not (dataset_name and time_window_span):
            print("No dataset selected or time window span not specified.")
            return {}

        print("Current windows:", current_windows)

        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Get the correct file path
        if "cleaned_data_path" in metadata[dataset_name]:
            file_path = metadata[dataset_name]["cleaned_data_path"]
        else:
            file_path = metadata[dataset_name]["path"]

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return {}

        print("File path:", file_path)
        df = pd.read_csv(file_path)
        sampling_rate = metadata.get(dataset_name, {}).get('sampling_rate', 100)

        # Create time axis
        df['Time_seconds'] = df.index / sampling_rate

        # Use current windows data instead of relayout data
        window_ranges = []

        # If we have current windows data, use it
        if current_windows:
            for window in current_windows:
                window_ranges.append({
                    'window_id': window['window_id'],
                    'start_time': window['start_time'],
                    'end_time': window['end_time'],
                    'duration': window['end_time'] - window['start_time']
                })
        else:
            # Fallback: extract windows from figure shapes if current_windows is empty
            if current_figure and 'layout' in current_figure and 'shapes' in current_figure['layout']:
                shapes = current_figure['layout']['shapes']
                for idx, shape in enumerate(shapes):
                    if shape.get('type') == 'rect' and 'x0' in shape and 'x1' in shape:
                        x0, x1 = shape['x0'], shape['x1']
                        if x0 > x1:
                            x0, x1 = x1, x0
                        window_ranges.append({
                            'window_id': idx,
                            'start_time': x0,
                            'end_time': x1,
                            'duration': x1 - x0
                        })

        if not window_ranges:
            print(
                "No window positions found. Please add windows first using the window controls.")
            return {}

        # Get sensor columns
        sensor_cols = [col for col in df.columns if col not in ['Time_seconds']]
        priority_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        available_cols = [col for col in priority_cols if col in sensor_cols]

        if not available_cols:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            available_cols = [
                col for col in numerical_cols if col != 'Time_seconds'][:6]

        sample_files = []
        sample_info = []
        all_selected_data = []

        # Extract and save each dragged window
        for window_info in window_ranges:
            window_id = window_info['window_id']
            start_time = window_info['start_time']
            end_time = window_info['end_time']

            mask = (df['Time_seconds'] >= start_time) & (
                df['Time_seconds'] <= end_time)
            window_data = df[mask][available_cols].copy()

            if len(window_data) > 0:
                window_data['Window_ID'] = window_id
                window_data['Time_seconds'] = df[mask]['Time_seconds']
                all_selected_data.append(window_data)

                # Use working directory for window storage
                windows_dir = os.path.join(base_dir, 'windows')
                os.makedirs(windows_dir, exist_ok=True)
                sample_file_path = os.path.join(
                    windows_dir, f"dragged_window_{window_id}_{dataset_name}")
                # Use 4 decimal places precision for sensor data readability
                window_data[available_cols].to_csv(
                    sample_file_path, index=False, float_format='%.4f')
                sample_files.append(sample_file_path)

                sample_info.append({
                    'window_id': window_id,
                    'samples': len(window_data),
                    'time_start': start_time,
                    'time_end': end_time,
                    'duration': end_time - start_time
                })
                print(
                    f"Saved dragged window {window_id} with {len(window_data)} samples")

        if not all_selected_data:
            print("No data found in dragged windows.")
            return {}

        # Combine all selected data for visualization
        combined_data = pd.concat(all_selected_data, ignore_index=True)

        # Update metadata - clean up old manual windows first
        if 'dragged_samples' in metadata[dataset_name]:
            # Remove old dragged window files (but keep sliding windows)
            old_files = metadata[dataset_name]['dragged_samples']
            for old_file in old_files:
                # Only remove manual dragged windows, not sliding windows
                if 'dragged_window_' in old_file and os.path.exists(old_file):
                    try:
                        os.remove(old_file)
                        print(f"Removed old window file: {old_file}")
                    except Exception as e:
                        print(f"Could not remove {old_file}: {e}")

            # Keep only sliding window files in metadata
            metadata[dataset_name]['dragged_samples'] = [
                f for f in old_files if 'sliding_' in f
            ]
        else:
            metadata[dataset_name]['dragged_samples'] = []

        # Add new manual window files
        metadata[dataset_name]['dragged_samples'].extend(sample_files)

        # Save window positions for easy reload
        metadata[dataset_name]['manual_window_positions'] = [
            {
                'window_id': w['window_id'],
                'start_time': w['start_time'],
                'end_time': w['end_time']
            }
            for w in window_ranges
        ]
        # Save window size for easy reload
        metadata[dataset_name]['window_size_ms'] = time_window_span

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Create visualization of selected samples
        fig = go.Figure()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        line_styles = ['solid', 'solid', 'solid', 'dash', 'dash', 'dash']

        for i, col in enumerate(available_cols[:6]):
            for window_info in window_ranges:
                window_id = window_info['window_id']
                window_mask = combined_data['Window_ID'] == window_id
                window_subset = combined_data[window_mask]

                if len(window_subset) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=window_subset['Time_seconds'],
                            y=window_subset[col],
                            name=f'{col} (W{window_id})',
                            line=dict(
                                color=colors[i % len(colors)],
                                width=3,
                                dash=line_styles[i % len(line_styles)]
                            ),
                            hovertemplate=f'<b>{col} - Window {window_id}</b><br>Time: %{{x:.3f}}s<br>Value: %{{y:.3f}}<extra></extra>',
                            legendgroup=f'sensor_{i}',
                            showlegend=(window_id == window_ranges[0]['window_id'])
                        )
                    )

        # Add window boundary indicators
        for window_info in window_ranges:
            start_time = window_info['start_time']
            end_time = window_info['end_time']
            window_id = window_info['window_id']

            fig.add_vline(
                x=start_time,
                line=dict(color="green", width=3, dash="solid"),
                annotation_text=f"W{window_id} Start",
                annotation_position="top"
            )
            fig.add_vline(
                x=end_time,
                line=dict(color="red", width=3, dash="solid"),
                annotation_text=f"W{window_id} End",
                annotation_position="top"
            )

        # Update layout
        fig.update_layout(
            title=f"Extracted Dragged Windows from {dataset_name}<br>"
            f"Windows: {[w['window_id'] for w in window_ranges]}, Total Samples: {len(combined_data)}",
            xaxis_title="Time (seconds)",
            yaxis_title="Sensor Values",
            height=600,
            showlegend=True,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Add summary annotation
        summary_lines = [f"✅ Extracted {len(window_ranges)} dragged windows"]
        summary_lines.append(f"📊 Total samples: {len(combined_data)}")

        for window_info in sample_info:
            duration_ms = window_info['duration'] * 1000
            summary_lines.append(
                f"🔹 W{window_info['window_id']}: {window_info['samples']} samples, "
                f"{duration_ms:.0f}ms duration"
            )

        fig.add_annotation(
            text="<br>".join(summary_lines),
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10, color="green"),
            bgcolor="rgba(200,255,200,0.9)",
            bordercolor="green",
            borderwidth=2,
            align="left"
        )

        return fig


    @app.callback(
        Output('split-dataset-selector', 'options'),
        Output('split-dataset-selector', 'value'),
        Input('split-selected-windows-btn', 'n_clicks'),
        Input('dataset-selector_', 'value'),
        # Add this to trigger update when split completes
        Input('split-samples-graph', 'figure'),
        State('split-dataset-selector', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=False
    )
    def update_split_dataset_selector(split_clicks, dataset_name, split_graph, current_value, base_dir):
        """Update the dropdown options for split datasets."""
        if not dataset_name:
            return [], None

        try:
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Get all split window files for the current dataset
            split_files = []
            if 'dragged_samples' in metadata.get(dataset_name, {}):
                split_files = metadata[dataset_name]['dragged_samples']

            # Also check for any files in persistent_data that match the pattern
            import glob
            pattern = get_window_pattern(dataset_name)
            existing_files = glob.glob(pattern)

            # Combine and deduplicate
            all_files = list(set(split_files + existing_files))

            options = []
            for file_path in all_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path).replace(".csv", "")
                    dataset_base = dataset_name.replace(".csv", "")

                    # Extract window info from filename
                    # Handle nested prefix: dragged_window_sliding_X or dragged_window_X
                    temp = filename

                    # Remove dataset name suffix first
                    if temp.endswith(f"_{dataset_base}"):
                        temp = temp.replace(f"_{dataset_base}", "")

                    # Now check for window type prefixes
                    if temp.startswith("dragged_window_sliding_"):
                        # Nested case: dragged_window_sliding_0 -> 0
                        window_id = temp.replace("dragged_window_sliding_", "")
                        window_type = "Sliding"
                    elif temp.startswith("dragged_window_"):
                        # Manual case: dragged_window_0 -> 0
                        window_id = temp.replace("dragged_window_", "")
                        window_type = "Manual"
                    elif temp.startswith("sliding_"):
                        # Direct sliding case: sliding_0 -> 0
                        window_id = temp.replace("sliding_", "")
                        window_type = "Sliding"
                    else:
                        # Fallback
                        window_id = temp
                        window_type = "Window"

                    # Get file stats
                    df = pd.read_csv(file_path)
                    samples = len(df)
                    duration_est = samples / 100  # Assuming 100Hz sampling rate

                    display_name = f"{window_type} {window_id} ({samples} samples, ~{duration_est:.1f}s)"
                    options.append({'label': display_name, 'value': file_path})

            # Sort options by window ID
            options.sort(key=lambda x: x['label'])

            # Preserve current selection if it still exists
            if current_value and current_value in [opt['value'] for opt in options]:
                return options, current_value

            # Return first option as default if available
            return options, options[0]['value'] if options else None

        except Exception as e:
            print(f"Error updating split dataset selector: {e}")
            return [], None


    @app.callback(
        Output('split-window-info', 'children'),
        Output('selected-split-graph', 'figure'),
        Output('split-window-stats-table', 'columns'),
        Output('split-window-stats-table', 'data'),
        Input('split-dataset-selector', 'value'),
        prevent_initial_call=True
    )
    def display_selected_split_window(selected_file_path):
        """Display details and visualization of the selected split window."""
        if not selected_file_path or not os.path.exists(selected_file_path):
            return "No split window selected.", {}, [], []

        try:
            # Load the split window data
            df = pd.read_csv(selected_file_path)
            filename = os.path.basename(selected_file_path)

            # Extract window information from filename
            parts = filename.replace(".csv", "").split("_")
            window_id = parts[2] if len(parts) > 2 else "Unknown"
            dataset_name = "_".join(parts[3:]) if len(parts) > 3 else "Unknown"

            # Calculate statistics
            samples = len(df)
            duration_est = samples / 100  # Assuming 100Hz
            sensor_cols = [col for col in df.columns if col not in [
                'Time_seconds', 'Window_ID']]

            # Create info panel content
            info_content = html.Div([
                html.H5(f"Window {window_id} - {dataset_name}",
                        style={'margin': '0 0 10px 0', 'color': '#2E86AB'}),
                html.Div([
                    html.Span(f"📊 Samples: {samples}", style={
                              'margin-right': '20px', 'font-weight': 'bold'}),
                    html.Span(f"⏱️ Duration: ~{duration_est:.2f}s", style={
                              'margin-right': '20px', 'font-weight': 'bold'}),
                    html.Span(f"📡 Sensors: {len(sensor_cols)}", style={
                              'font-weight': 'bold'})
                ]),
                html.Div([
                    html.Span("📁 File: ", style={'font-weight': 'bold'}),
                    html.Span(filename, style={
                              'font-family': 'monospace', 'background-color': '#e9ecef', 'padding': '2px 6px', 'border-radius': '3px'})
                ], style={'margin-top': '10px'})
            ])

            # Create visualization
            fig = go.Figure()
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                      '#d62728', '#9467bd', '#8c564b']
            line_styles = ['solid', 'solid', 'solid', 'dash', 'dash', 'dash']

            # Add time axis if not present
            if 'Time_seconds' not in df.columns:
                df['Time_seconds'] = df.index / 100  # Assuming 100Hz

            # Limit to 6 sensors for clarity
            for i, col in enumerate(sensor_cols[:6]):
                fig.add_trace(
                    go.Scatter(
                        x=df['Time_seconds'] if 'Time_seconds' in df.columns else df.index,
                        y=df[col],
                        name=col,
                        line=dict(
                            color=colors[i % len(colors)],
                            width=2,
                            dash=line_styles[i % len(line_styles)]
                        ),
                        hovertemplate=f'<b>{col}</b><br>Time: %{{x:.3f}}s<br>Value: %{{y:.3f}}<extra></extra>'
                    )
                )

            # Update layout
            fig.update_layout(
                title=f"Split Window {window_id} - Detailed View",
                xaxis_title="Time (seconds)" if 'Time_seconds' in df.columns else "Sample Index",
                yaxis_title="Sensor Values",
                height=400,
                showlegend=True,
                hovermode='x unified',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Calculate statistics for table
            stats_data = []
            columns = [
                {"name": "Sensor", "id": "sensor"},
                {"name": "Mean", "id": "mean"},
                {"name": "Std Dev", "id": "std"},
                {"name": "Min", "id": "min"},
                {"name": "Max", "id": "max"},
                {"name": "Range", "id": "range"}
            ]

            for col in sensor_cols:
                stats_data.append({
                    'sensor': col,
                    'mean': f"{df[col].mean():.3f}",
                    'std': f"{df[col].std():.3f}",
                    'min': f"{df[col].min():.3f}",
                    'max': f"{df[col].max():.3f}",
                    'range': f"{df[col].max() - df[col].min():.3f}"
                })

            return info_content, fig, columns, stats_data

        except Exception as e:
            error_msg = f"Error loading split window: {str(e)}"
            return html.Div(error_msg, style={'color': 'red'}), {}, [], []


    @app.callback(
        Output('split-dataset-selector', 'options', allow_duplicate=True),
        Output('split-dataset-selector', 'value', allow_duplicate=True),
        Input('delete-split-window-btn', 'n_clicks'),
        State('split-dataset-selector', 'value'),
        State('dataset-selector_', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def delete_split_window(n_clicks, selected_file_path, dataset_name, base_dir):
        """Delete the selected split window file and update metadata."""
        if not selected_file_path or not os.path.exists(selected_file_path):
            return no_update, no_update

        try:
            # Extract window_id from filename - handle both manual and sliding windows
            filename = os.path.basename(selected_file_path)
            # Remove dataset name and base prefix
            temp = filename.replace(f"_{dataset_name}", "").replace(
                "dragged_window_", "").replace(".csv", "")

            # Determine if it's a sliding window or manual window
            if temp.startswith("sliding_"):
                # Sliding window: dragged_window_sliding_79_dataset.csv -> sliding_79 -> 79
                deleted_window_id = int(temp.replace("sliding_", ""))
                is_sliding = True
            else:
                # Manual window: dragged_window_5_dataset.csv -> 5
                deleted_window_id = int(temp)
                is_sliding = False

            # Delete the file
            os.remove(selected_file_path)
            print(
                f"Deleted {'sliding' if is_sliding else 'manual'} window file: {selected_file_path}")

            # Update metadata
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if dataset_name in metadata:
                # Remove from dragged_samples
                if 'dragged_samples' in metadata[dataset_name]:
                    if selected_file_path in metadata[dataset_name]['dragged_samples']:
                        metadata[dataset_name]['dragged_samples'].remove(
                            selected_file_path)

                # Only remove from manual_window_positions if it's a manual window
                # Sliding windows don't have entries in manual_window_positions
                if not is_sliding and 'manual_window_positions' in metadata[dataset_name]:
                    metadata[dataset_name]['manual_window_positions'] = [
                        pos for pos in metadata[dataset_name]['manual_window_positions']
                        if pos['window_id'] != deleted_window_id
                    ]
                    print(
                        f"Removed window position for manual window_id {deleted_window_id}")

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Get updated options using working directory
            import glob
            windows_dir = os.path.join(base_dir, 'windows')
            pattern = os.path.join(windows_dir, f"dragged_window_*_{dataset_name}")
            existing_files = glob.glob(pattern)

            options = []
            for file_path in existing_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    parts = filename.replace(f"_{dataset_name}", "").replace(
                        "dragged_window_", "")
                    window_id = parts.split(
                        "_")[0] if "_" in parts else parts.replace(".csv", "")

                    df = pd.read_csv(file_path)
                    samples = len(df)
                    duration_est = samples / 100

                    display_name = f"Window {window_id} ({samples} samples, ~{duration_est:.1f}s)"
                    options.append({'label': display_name, 'value': file_path})

            options.sort(key=lambda x: x['label'])

            # Return updated options and reset selection
            return options, options[0]['value'] if options else None

        except Exception as e:
            print(f"Error deleting split window: {e}")
            import traceback
            traceback.print_exc()
            return no_update, no_update


    @app.callback(
        Output('split-dataset-selector', 'options', allow_duplicate=True),
        Output('split-dataset-selector', 'value', allow_duplicate=True),
        Output('split-samples-graph', 'figure', allow_duplicate=True),
        Output('selected-split-graph', 'figure', allow_duplicate=True),
        Output('split-window-info', 'children', allow_duplicate=True),
        Output('split-window-stats-table', 'columns', allow_duplicate=True),
        Output('split-window-stats-table', 'data', allow_duplicate=True),
        Input('clean-generated-data-btn', 'n_clicks'),
        State('dataset-selector_', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def clean_all_generated_data(n_clicks, dataset_name, base_dir):
        """Clean all generated split window data for the current dataset."""
        if not dataset_name:
            return [], None, {}, {}, "No dataset selected.", [], []

        try:
            import glob

            # Get all split window files for the current dataset
            pattern = get_window_pattern(dataset_name)
            split_files = glob.glob(pattern)

            # Also get sample_window files that might have been generated
            sample_pattern = os.path.join(
                WINDOWS_DIR, f"sample_window_*_{dataset_name}")
            sample_files = glob.glob(sample_pattern)

            all_files_to_delete = split_files + sample_files
            deleted_count = 0

            # Delete all found files
            for file_path in all_files_to_delete:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    deleted_count += 1
                    print(f"Deleted: {file_path}")

            # Update metadata to remove references to deleted files
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if dataset_name in metadata:
                # Clear dragged_samples array if it exists
                if 'dragged_samples' in metadata[dataset_name]:
                    metadata[dataset_name]['dragged_samples'] = []

                # Remove any other generated data references
                keys_to_remove = [key for key in metadata[dataset_name].keys()
                                  if 'split' in key.lower() or 'window' in key.lower() or 'sample' in key.lower()]

                for key in keys_to_remove:
                    if key != 'cleaned_data_path':  # Don't remove cleaned data path
                        del metadata[dataset_name][key]

            # Save updated metadata
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f)

            print(
                f"Successfully cleaned {deleted_count} generated data files for {dataset_name}")

            # Clear all outputs and return empty states
            empty_info = html.Div([
                html.H5("All Generated Data Cleaned", style={
                        'margin': '0 0 10px 0', 'color': '#28a745'}),
                html.Div([
                    html.Span(f"✅ Deleted {deleted_count} files", style={
                              'font-weight': 'bold', 'color': '#28a745'}),
                    html.Br(),
                    html.Span("You can now re-split your data with new window configurations.",
                              style={'color': '#6c757d', 'font-style': 'italic'})
                ])
            ])

            return [], None, {}, {}, empty_info, [], []

        except Exception as e:
            error_msg = f"Error cleaning generated data: {str(e)}"
            print(error_msg)

            error_info = html.Div([
                html.H5("Error Cleaning Data", style={
                        'margin': '0 0 10px 0', 'color': '#dc3545'}),
                html.Div(error_msg, style={'color': '#dc3545'})
            ])

            return [], None, {}, {}, error_info, [], []


    @app.callback(
        Output('training-dataset-selector', 'options'),
        Output('training-dataset-info', 'children'),
        Input('dataset-selector_', 'value'),
        # Add this to trigger update when split completes
        Input('split-samples-graph', 'figure'),
        # Add this to trigger when split options change
        Input('split-dataset-selector', 'options'),
        prevent_initial_call=False
    )
    def populate_training_dataset_selector(dataset_name, split_graph, split_options):
        """Populate training dataset selector with available split windows."""
        if not dataset_name:
            return [], "No dataset selected."

        try:
            import glob

            # Get all split window files for the current dataset
            pattern = get_window_pattern(dataset_name)
            split_files = glob.glob(pattern)

            if not split_files:
                return [], html.Div([
                    html.P("⚠️ No split windows available for training.",
                           style={'color': '#FF9800', 'margin': '0'}),
                    html.P("Please split some windows first in the sections above.", style={
                           'font-size': '12px', 'margin': '5px 0 0 0'})
                ])

            options = []
            total_samples = 0

            for file_path in split_files:
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path).replace(".csv", "")
                    dataset_base = dataset_name.replace(".csv", "")

                    # Extract window info from filename
                    # Handle nested prefix: dragged_window_sliding_X or dragged_window_X
                    temp = filename

                    # Remove dataset name suffix first
                    if temp.endswith(f"_{dataset_base}"):
                        temp = temp.replace(f"_{dataset_base}", "")

                    # Now check for window type prefixes
                    if temp.startswith("dragged_window_sliding_"):
                        # Nested case: dragged_window_sliding_0 -> 0
                        window_id = temp.replace("dragged_window_sliding_", "")
                        window_type = "Sliding"
                    elif temp.startswith("dragged_window_"):
                        # Manual case: dragged_window_0 -> 0
                        window_id = temp.replace("dragged_window_", "")
                        window_type = "Manual"
                    elif temp.startswith("sliding_"):
                        # Direct sliding case: sliding_0 -> 0
                        window_id = temp.replace("sliding_", "")
                        window_type = "Sliding"
                    else:
                        # Fallback
                        window_id = temp
                        window_type = "Window"

                    # Get file stats
                    df = pd.read_csv(file_path)
                    samples = len(df)
                    duration_est = samples / 100  # Assuming 100Hz sampling rate
                    total_samples += samples

                    display_name = f"{window_type} {window_id} ({samples} samples, ~{duration_est:.1f}s)"
                    options.append({'label': display_name, 'value': file_path})

            # Sort options by window ID
            options.sort(key=lambda x: x['label'])

            # Create info message
            info_content = html.Div([
                html.P(f"📊 {len(options)} split windows available for training", style={
                       'margin': '0', 'font-weight': 'bold'}),
                html.P(f"💾 Total samples: {total_samples}", style={
                       'margin': '5px 0 0 0', 'font-size': '14px'})
            ])

            return options, info_content

        except Exception as e:
            error_msg = f"Error loading training datasets: {str(e)}"
            return [], html.Div(error_msg, style={'color': '#dc3545'})


    @app.callback(
        Output('training-dataset-selector', 'value'),
        Input('select-all-training-windows-btn', 'n_clicks'),
        Input('clear-all-training-windows-btn', 'n_clicks'),
        State('training-dataset-selector', 'options'),
        prevent_initial_call=True
    )
    def manage_training_window_selection(select_all_clicks, clear_all_clicks, available_options):
        """Handle Select All and Clear All buttons for training window selection."""
        from dash import ctx

        if not ctx.triggered or not available_options:
            return no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'select-all-training-windows-btn':
            # Select all available windows
            return [option['value'] for option in available_options]
        elif button_id == 'clear-all-training-windows-btn':
            # Clear all selections
            return []

        return no_update


    @app.callback(
        Output('test-split-display', 'children'),
        Output('split-ratio-info', 'children'),
        Input('train-split', 'value'),
        Input('val-split', 'value')
    )
    def update_split_displays(train_ratio, val_ratio):
        """Update test split display and split ratio information."""
        if train_ratio is None or val_ratio is None:
            return "20%", ""

        # Calculate test ratio
        test_ratio = 1.0 - train_ratio - val_ratio

        # Ensure valid split (test >= 0.1)
        if test_ratio < 0.1:
            test_ratio = 0.1
            train_ratio = 0.9 - val_ratio

        train_percent = int(train_ratio * 100)
        val_percent = int(val_ratio * 100)
        test_percent = int(test_ratio * 100)

        # Test display
        test_display = f"{test_percent}%"

        # Info message
        if val_percent == 0:
            info_msg = html.Div([
                html.Span(f"🎓 Training: {train_percent}%", style={
                    'margin-right': '15px', 'color': '#4CAF50', 'font-weight': 'bold'}),
                html.Span(f"🧪 Testing: {test_percent}%", style={
                    'margin-right': '15px', 'color': '#2196F3', 'font-weight': 'bold'}),
                html.Br(),
                html.Span("⚙️ Validation set is 0% - will use 5-fold cross-validation on training set",
                          style={'color': '#FF9800', 'font-style': 'italic', 'font-size': '12px'})
            ])
        else:
            info_msg = html.Div([
                html.Span(f"🎓 Training: {train_percent}%", style={
                    'margin-right': '15px', 'color': '#4CAF50', 'font-weight': 'bold'}),
                html.Span(f"🔍 Validation: {val_percent}%", style={
                    'margin-right': '15px', 'color': '#FF9800', 'font-weight': 'bold'}),
                html.Span(f"🧪 Testing: {test_percent}%", style={
                    'color': '#2196F3', 'font-weight': 'bold'})
            ])

        return test_display, info_msg


    @app.callback(
        Output('preprocessed-training-data', 'data'),
        Output('preprocessing-results', 'children'),
        Input('preprocess-for-training-btn', 'n_clicks'),
        State('training-dataset-selector', 'value'),
        State('normalization-method', 'value'),
        State('feature-selection-method', 'value'),
        State('dataset-selector_', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def preprocess_for_training(n_clicks, selected_files, norm_method, feature_method, dataset_name, base_dir):
        """Preprocess selected split windows for model training."""
        if not (selected_files and norm_method and feature_method):
            return {}, html.Div("⚠️ Please select datasets and preprocessing options.", style={'color': '#FF9800'})

        try:
            # Get the correct label from metadata.json
            activity_label = None
            try:
                if not base_dir:
                    base_dir = PERSISTENT_DIR
                metadata_file = os.path.join(base_dir, 'metadata.json')

                if os.path.exists(metadata_file):
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    dataset_info = metadata.get(dataset_name, {})
                    # Get the actual label from metadata
                    activity_label = dataset_info.get('label', None)

                    if activity_label:
                        print(
                            f"DEBUG: Dataset '{dataset_name}' has label '{activity_label}' from metadata")
                    else:
                        print(
                            f"WARNING: No label found in metadata for '{dataset_name}', using fallback")
                else:
                    print(f"WARNING: Metadata file not found at {METADATA_FILE}")
            except Exception as meta_error:
                print(f"WARNING: Error reading metadata: {meta_error}")

            # Fallback: extract from dataset name if metadata not available
            if not activity_label:
                activity_label = dataset_name.replace('.csv', '').rsplit('_', 1)[0]
                print(
                    f"DEBUG: Using fallback label '{activity_label}' from dataset name")

            # Load and combine all selected split windows
            all_data = []
            file_info = []

            for file_path in selected_files:
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)

                    # Extract window ID from filename
                    filename = os.path.basename(file_path)
                    parts = filename.replace(f"_{dataset_name}", "").replace(
                        "dragged_window_", "")
                    window_id = parts.split("_")[0] if "_" in parts else parts

                    # Add window ID and label for classification
                    df['Window_ID'] = window_id
                    # Use the label from metadata.json
                    df['Activity_Label'] = activity_label

                    all_data.append(df)
                    file_info.append({
                        'window_id': window_id,
                        'samples': len(df),
                        'file_path': file_path
                    })

            if not all_data:
                return {}, html.Div("❌ No valid data files found.", style={'color': '#dc3545'})

            # Extract features from each window
            sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
            extracted_features_list = []
            labels_list = []
            window_ids_list = []

            for df in all_data:
                # Get window-specific data
                window_id = df['Window_ID'].iloc[0]
                activity_label = df['Activity_Label'].iloc[0]

                # Remove metadata columns for feature extraction
                window_data = df[sensor_cols]

                # Extract features based on selection method
                if feature_method == 'statistical':
                    # Use only raw sensor values (no feature engineering)
                    # Average each sensor across the window
                    features_df = pd.DataFrame([{
                        col: window_data[col].mean() for col in sensor_cols
                    }])
                elif feature_method == 'time_domain':
                    # Extract 90 time-domain features (15 per axis)
                    features_df = extract_time_domain_features(
                        window_data, sensor_cols)
                elif feature_method == 'all':
                    # Extract both time-domain (90) and frequency-domain (48) features = 138 total
                    time_features = extract_time_domain_features(
                        window_data, sensor_cols)
                    freq_features = extract_frequency_domain_features(
                        window_data, sensor_cols)
                    features_df = pd.concat([time_features, freq_features], axis=1)
                else:  # custom
                    # Default to time-domain
                    features_df = extract_time_domain_features(
                        window_data, sensor_cols)

                extracted_features_list.append(features_df)
                labels_list.append(activity_label)
                window_ids_list.append(window_id)

            # Combine all extracted features
            features_df = pd.concat(extracted_features_list, ignore_index=True)
            feature_cols = features_df.columns.tolist()

            if not feature_cols:
                return {}, html.Div("❌ No features available for training.", style={'color': '#dc3545'})

            # Prepare arrays
            X = features_df.values
            y = np.array(labels_list)
            window_ids = np.array(window_ids_list)

            # Handle any missing values
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)

            # Apply normalization
            scaler = None
            if norm_method == 'minmax':
                scaler = MinMaxScaler()
                X_scaled = scaler.fit_transform(X)
            elif norm_method == 'standard':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
            elif norm_method == 'robust':
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)
            else:  # none
                X_scaled = X

            # Store preprocessed data
            preprocessed_data = {
                'features': X_scaled.tolist(),
                'labels': y.tolist(),
                'window_ids': window_ids.tolist(),
                'feature_names': feature_cols,
                'scaler_type': norm_method,
                'scaler_params': scaler.get_params() if scaler else None,
                'original_shape': X_scaled.shape
            }

            # Create results summary
            results_content = html.Div([
                html.H5("✅ Preprocessing Complete", style={
                        'color': '#4CAF50', 'margin': '0 0 15px 0'}),
                html.Div([
                    html.Div([
                        html.Span("📊 Total Samples: ", style={
                                  'font-weight': 'bold'}),
                        html.Span(f"{len(X_scaled)}")
                    ], style={'margin-bottom': '5px'}),
                    html.Div([
                        html.Span("🎯 Features Extracted: ",
                                  style={'font-weight': 'bold'}),
                        html.Span(
                            f"{len(feature_cols)} features ({_get_feature_method_label(feature_method)})")
                    ], style={'margin-bottom': '5px'}),
                    html.Div([
                        html.Span("📏 Normalization: ", style={
                                  'font-weight': 'bold'}),
                        html.Span(
                            f"{norm_method.title() if norm_method != 'none' else 'None'}")
                    ], style={'margin-bottom': '5px'}),
                    html.Div([
                        html.Span("🗂️ Windows Used: ", style={
                                  'font-weight': 'bold'}),
                        html.Span(f"{len(file_info)} windows")
                    ], style={'margin-bottom': '10px'}),
                    html.Details([
                        html.Summary("View Selected Features", style={
                                     'cursor': 'pointer', 'font-weight': 'bold'}),
                        html.Div([
                            html.Span(f"{i+1}. {col}") for i, col in enumerate(feature_cols[:10])
                        ] + ([html.Span(f"... and {len(feature_cols)-10} more")] if len(feature_cols) > 10 else []),
                            style={'margin-top': '10px', 'font-family': 'monospace', 'font-size': '12px'})
                    ])
                ])
            ])

            return preprocessed_data, results_content

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"ERROR in preprocess_for_training: {error_details}")
            error_msg = html.Div([
                html.H5("❌ Error during preprocessing", style={
                        'color': '#dc3545', 'margin-bottom': '10px'}),
                html.P(f"Error: {str(e)}", style={'margin-bottom': '5px'}),
                html.Details([
                    html.Summary("Show technical details", style={
                                 'cursor': 'pointer', 'color': '#6c757d'}),
                    html.Pre(error_details, style={'font-size': '11px', 'background-color': '#f8f9fa',
                             'padding': '10px', 'border-radius': '4px', 'overflow': 'auto'})
                ])
            ])
            return {}, error_msg


    @app.callback(
        Output('train-test-data', 'data'),
        Output('train-test-split-graph', 'figure'),
        Input('train-test-split-btn', 'n_clicks'),
        State('preprocessed-training-data', 'data'),
        State('train-split', 'value'),
        State('val-split', 'value'),
        State('random-state-input', 'value'),
        prevent_initial_call=True
    )
    def perform_enhanced_train_val_test_split(n_clicks, preprocessed_data, train_ratio, val_ratio, random_state):
        """Perform train-validation-test split on preprocessed data and visualize results."""
        if not (preprocessed_data and train_ratio is not None and val_ratio is not None):
            return {}, {}

        try:
            # Extract data
            X = np.array(preprocessed_data['features'])
            y = np.array(preprocessed_data['labels'])
            feature_names = preprocessed_data['feature_names']

            # Calculate test ratio
            test_ratio = 1.0 - train_ratio - val_ratio
            if test_ratio < 0.1:
                test_ratio = 0.1
                train_ratio = 0.9 - val_ratio

            # Perform split based on validation ratio
            if val_ratio == 0:
                # 2-way split: train and test only (use CV during training)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_ratio, random_state=random_state, stratify=y
                )
                X_val, y_val = None, None
            else:
                # 3-way split: train, validation, and test
                # First split: separate test set
                X_temp, X_test, y_temp, y_test = train_test_split(
                    X, y, test_size=test_ratio, random_state=random_state, stratify=y
                )

                # Second split: separate train and validation from remaining data
                val_size_adjusted = val_ratio / (train_ratio + val_ratio)
                X_train, X_val, y_train, y_val = train_test_split(
                    X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
                )

            # Store split data
            split_data = {
                'X_train': X_train.tolist(),
                'X_val': X_val.tolist() if X_val is not None else [],
                'X_test': X_test.tolist(),
                'y_train': y_train.tolist(),
                'y_val': y_val.tolist() if y_val is not None else [],
                'y_test': y_test.tolist(),
                'feature_names': feature_names,
                'train_ratio': train_ratio,
                'val_ratio': val_ratio,
                'test_ratio': test_ratio,
                'random_state': random_state,
                'has_validation': val_ratio > 0
            }

            # Create visualization
            fig = go.Figure()

            # Sample a few features for visualization
            max_features_to_plot = min(6, len(feature_names))
            selected_features = feature_names[:max_features_to_plot]

            colors = ['#1f77b4', '#ff7f0e', '#2ca02c',
                      '#d62728', '#9467bd', '#8c564b']

            for i, feature in enumerate(selected_features):
                feature_idx = feature_names.index(feature)
                current_idx = 0

                # Training data
                fig.add_trace(go.Scatter(
                    x=list(range(current_idx, current_idx + len(X_train))),
                    y=X_train[:, feature_idx],
                    mode='markers',
                    name=f'{feature} (Train)',
                    marker=dict(color=colors[i % len(colors)],
                                symbol='circle', size=4),
                    opacity=0.7
                ))
                current_idx += len(X_train)

                # Validation data (if present)
                if val_ratio > 0:
                    fig.add_trace(go.Scatter(
                        x=list(range(current_idx, current_idx + len(X_val))),
                        y=X_val[:, feature_idx],
                        mode='markers',
                        name=f'{feature} (Val)',
                        marker=dict(color=colors[i % len(colors)],
                                    symbol='square', size=4),
                        opacity=0.7
                    ))
                    current_idx += len(X_val)

                # Test data
                fig.add_trace(go.Scatter(
                    x=list(range(current_idx, current_idx + len(X_test))),
                    y=X_test[:, feature_idx],
                    mode='markers',
                    name=f'{feature} (Test)',
                    marker=dict(color=colors[i % len(colors)],
                                symbol='diamond', size=4),
                    opacity=0.7
                ))

            # Add vertical lines to separate sets
            fig.add_vline(
                x=len(X_train)-0.5,
                line=dict(color="green", width=2, dash="dash"),
                annotation_text="Train | Val" if val_ratio > 0 else "Train | Test",
                annotation_position="top"
            )

            if val_ratio > 0:
                fig.add_vline(
                    x=len(X_train) + len(X_val) - 0.5,
                    line=dict(color="red", width=2, dash="dash"),
                    annotation_text="Val | Test",
                    annotation_position="top"
                )

            # Update layout
            if val_ratio > 0:
                title_text = (f"Train-Validation-Test Split Visualization<br>"
                              f"Training: {len(X_train)} samples ({train_ratio:.1%}) | "
                              f"Validation: {len(X_val)} samples ({val_ratio:.1%}) | "
                              f"Testing: {len(X_test)} samples ({test_ratio:.1%})")
            else:
                title_text = (f"Train-Test Split Visualization (CV Mode)<br>"
                              f"Training: {len(X_train)} samples ({train_ratio:.1%}) | "
                              f"Testing: {len(X_test)} samples ({test_ratio:.1%})")

            fig.update_layout(
                title=title_text,
                xaxis_title="Sample Index",
                yaxis_title="Normalized Feature Values",
                height=500,
                showlegend=True,
                hovermode='closest',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            # Add summary annotation
            if val_ratio > 0:
                annotation_text = (f"📊 Features Extracted: {len(feature_names)}<br>"
                                   f"🎓 Train: {len(X_train)} samples<br>"
                                   f"📋 Validation: {len(X_val)} samples<br>"
                                   f"🧪 Test: {len(X_test)} samples<br>"
                                   f"🎲 Random State: {random_state}")
            else:
                annotation_text = (f"📊 Features Extracted: {len(feature_names)}<br>"
                                   f"🎓 Train: {len(X_train)} samples<br>"
                                   f"🧪 Test: {len(X_test)} samples<br>"
                                   f"⚙️ Using 5-fold CV<br>"
                                   f"🎲 Random State: {random_state}")

            fig.add_annotation(
                text=annotation_text,
                xref="paper", yref="paper",
                x=0.02, y=0.98,
                showarrow=False,
                font=dict(size=10, color="darkgreen"),
                bgcolor="rgba(200,255,200,0.9)",
                bordercolor="green",
                borderwidth=2,
                align="left"
            )

            return split_data, fig

        except Exception as e:
            error_msg = f"Error during train-test split: {str(e)}"
            print(error_msg)
            return {}, {}


    # Callback disabled - clear-training-data-btn button doesn't exist in layout
    # @callback(
    #     Output('preprocessed-training-data', 'data', allow_duplicate=True),
    #     Output('train-test-data', 'data', allow_duplicate=True),
    #     Output('train-test-split-graph', 'figure', allow_duplicate=True),
    #     Output('preprocessing-results', 'children', allow_duplicate=True),
    #     Input('clear-training-data-btn', 'n_clicks'),
    #     State('dataset-selector_', 'value'),
    #     State('working-directory-store', 'data'),
    #     prevent_initial_call=True
    # )
    # def clear_training_data(n_clicks, dataset_name, base_dir):
        """Clear all preprocessed training data and associated files."""
        if not dataset_name:
            return {}, {}, {}, html.Div("⚠️ No dataset selected.", style={'color': '#FF9800'})

        deleted_files = []
        try:
            # Clear training data directory for this dataset
            if not base_dir:
                base_dir = PERSISTENT_DIR
            training_dir = os.path.join(base_dir, 'training_data')

            if os.path.exists(training_dir):
                # Remove files related to current dataset
                patterns = [
                    f"{dataset_name}_train.csv",
                    f"{dataset_name}_test.csv",
                    f"{dataset_name}_metadata.json"
                ]

                for pattern in patterns:
                    file_path = os.path.join(training_dir, pattern)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        deleted_files.append(pattern)

            # Clear session data
            empty_data = {}
            empty_figure = {}

            # Success message
            success_content = html.Div([
                html.H5("🧹 Training Data Cleared Successfully", style={
                    'color': '#28a745', 'margin': '0 0 15px 0'}),
                html.Div([
                    html.P(f"✅ Cleared feature engineering data for: {dataset_name}", style={
                        'margin': '5px 0', 'font-weight': 'bold'}),
                    html.P(f"📁 Files removed: {len(deleted_files)}", style={
                        'margin': '5px 0'}),
                    html.P("🔧 Ready for new feature engineering process", style={
                        'margin': '5px 0', 'color': '#6c757d', 'font-style': 'italic'})
                ])
            ])

            return empty_data, empty_data, empty_figure, success_content

        except Exception as e:
            error_msg = f"❌ Error clearing training data: {str(e)}"
            return no_update, no_update, no_update, html.Div(error_msg, style={'color': '#dc3545'})


    @app.callback(
        Output('dataset-status-display', 'children', allow_duplicate=True),
        Input('save-cleaned-smoothed-btn', 'n_clicks'),
        Input('split-selected-windows-btn', 'n_clicks'),
        State('dataset-selector_', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def update_dataset_status_on_operations(clean_clicks, split_clicks, dataset_name, base_dir):
        """Update dataset status when any processing operation completes."""
        if not dataset_name:
            return no_update

        try:
            if not base_dir:
                base_dir = PERSISTENT_DIR
            metadata_file = os.path.join(base_dir, 'metadata.json')

            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            dataset_info = metadata.get(dataset_name, {})

            # Clean up metadata by removing non-existent files
            if 'dragged_samples' in dataset_info:
                existing_files = [
                    f for f in dataset_info['dragged_samples'] if os.path.exists(f)]
                if len(existing_files) != len(dataset_info['dragged_samples']):
                    # Update metadata to remove stale references
                    metadata[dataset_name]['dragged_samples'] = existing_files
                    with open(metadata_file, 'w') as f:
                        json.dump(metadata, f, indent=2)
                    print(
                        f"Cleaned metadata for {dataset_name}: removed {len(dataset_info['dragged_samples']) - len(existing_files)} stale references")
                    dataset_info['dragged_samples'] = existing_files

            # Determine processing stages
            stages = {
                'raw': True,
                'preprocessed': 'cleaned_data_path' in dataset_info,
                'split': 'dragged_samples' in dataset_info and len(dataset_info.get('dragged_samples', [])) > 0,
                'training_ready': False
            }

            # Check if training data exists
            training_dir = os.path.join(base_dir, 'training_data')
            train_file = os.path.join(training_dir, f"{dataset_name}_train.csv")
            test_file = os.path.join(training_dir, f"{dataset_name}_test.csv")
            stages['training_ready'] = os.path.exists(
                train_file) and os.path.exists(test_file)

            # Create status display (same logic as in display_dataset_and_status)
            status_badges = []

            # Raw Data Status
            status_badges.append(
                html.Span("📁 Raw Data", className="badge", style={
                    'background-color': '#6c757d',
                    'color': 'white',
                    'padding': '6px 12px',
                    'border-radius': '12px',
                    'margin-right': '8px',
                    'margin-bottom': '8px',
                    'font-size': '12px',
                    'display': 'inline-block'
                })
            )

            # Signal Preprocessed Status
            if stages['preprocessed']:
                status_badges.append(
                    html.Span("🔧 Signal Processed", className="badge", style={
                        'background-color': '#17a2b8',
                        'color': 'white',
                        'padding': '6px 12px',
                        'border-radius': '12px',
                        'margin-right': '8px',
                        'margin-bottom': '8px',
                        'font-size': '12px',
                        'display': 'inline-block'
                    })
                )
            else:
                status_badges.append(
                    html.Span("⏳ Signal Processing Pending", className="badge", style={
                        'background-color': '#ffc107',
                        'color': '#212529',
                        'padding': '6px 12px',
                        'border-radius': '12px',
                        'margin-right': '8px',
                        'margin-bottom': '8px',
                        'font-size': '12px',
                        'display': 'inline-block'
                    })
                )

            # Split Status
            if stages['split']:
                # Count only files that actually exist on disk
                dragged_samples = dataset_info.get('dragged_samples', [])
                actual_split_count = sum(
                    1 for file_path in dragged_samples if os.path.exists(file_path))

                status_badges.append(
                    html.Span(f"✂️ Split ({actual_split_count} windows)", className="badge", style={
                        'background-color': '#fd7e14',
                        'color': 'white',
                        'padding': '6px 12px',
                        'border-radius': '12px',
                        'margin-right': '8px',
                        'margin-bottom': '8px',
                        'font-size': '12px',
                        'display': 'inline-block'
                    })
                )
            else:
                status_badges.append(
                    html.Span("⏳ Split Pending", className="badge", style={
                        'background-color': '#6c757d',
                        'color': 'white',
                        'padding': '6px 12px',
                        'border-radius': '12px',
                        'margin-right': '8px',
                        'margin-bottom': '8px',
                        'font-size': '12px',
                        'display': 'inline-block'
                    })
                )

            # Training Data Status
            if stages['training_ready']:
                status_badges.append(
                    html.Span("🚀 Training Ready", className="badge", style={
                        'background-color': '#28a745',
                        'color': 'white',
                        'padding': '6px 12px',
                        'border-radius': '12px',
                        'margin-right': '8px',
                        'margin-bottom': '8px',
                        'font-size': '12px',
                        'display': 'inline-block'
                    })
                )
            else:
                status_badges.append(
                    html.Span("⏳ Training Prep Pending", className="badge", style={
                        'background-color': '#6c757d',
                        'color': 'white',
                        'padding': '6px 12px',
                        'border-radius': '12px',
                        'margin-right': '8px',
                        'margin-bottom': '8px',
                        'font-size': '12px',
                        'display': 'inline-block'
                    })
                )

            # Create comprehensive status display
            status_display = html.Div([
                html.H6("📊 Dataset Processing Status", style={
                    'margin-bottom': '10px',
                    'color': '#495057',
                    'font-weight': 'bold'
                }),
                html.Div(status_badges, style={'line-height': '2.5'}),
                html.Hr(style={'margin': '15px 0'}),
                html.Div([
                    html.Small(f"📁 Dataset: {dataset_name}", style={
                        'display': 'block', 'color': '#6c757d', 'margin-bottom': '5px'}),
                    html.Small(f"📡 Sampling Rate: {dataset_info.get('sampling_rate', 'Unknown')} Hz", style={
                        'display': 'block', 'color': '#6c757d', 'margin-bottom': '5px'}),
                    html.Small(f"🏷️ Activity: {dataset_info.get('label', 'Unknown')}", style={
                        'display': 'block', 'color': '#6c757d'})
                ])
            ], style={
                'background-color': '#f8f9fa',
                'padding': '15px',
                'border-radius': '8px',
                'border': '1px solid #dee2e6'
            })

            return status_display

        except Exception as e:
            return html.Div(f"Error updating status: {str(e)}", style={'color': '#dc3545'})

