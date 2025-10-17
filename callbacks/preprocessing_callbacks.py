import os
import json
import pandas as pd
from dash import Input, Output, State, callback, no_update, html
from scipy.signal import savgol_filter
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np

from config.config import *
from utils.data_processing import clean_data, low_pass_filter


@callback(
    Output('dataset-selector_', 'options'),
    Input('tabs', 'value')
)
def populate_dataset_selector(tab):
    """Populate the dataset selector with available datasets."""
    if tab == 'tab-2':
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return [{'label': filename, 'value': filename} for filename in metadata.keys()]
    return []


@callback(
    Output('preprocessed-graph', 'figure'),
    Output('dataset-status-display', 'children'),
    Input('dataset-selector_', 'value')
)
def display_dataset_and_status(dataset_name):
    """Displays selected dataset as a chart and comprehensive status."""
    if not dataset_name:
        return {}, html.Div("Select a dataset to view status", style={'color': '#6c757d', 'font-style': 'italic'})

    with open(METADATA_FILE, 'r') as f:
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
    training_dir = os.path.join(PERSISTENT_DIR, 'training_data')
    train_file = os.path.join(training_dir, f"{dataset_name}_train.csv")
    test_file = os.path.join(training_dir, f"{dataset_name}_test.csv")
    stages['training_ready'] = os.path.exists(train_file) and os.path.exists(test_file)
    
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
        actual_split_count = sum(1 for file_path in dragged_samples if os.path.exists(file_path))
        
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
            title=f"Preview of {dataset_name} ({'Signal Processed' if stages['preprocessed'] else 'Raw Data'})",
            xaxis_title="Time (seconds)",
            yaxis_title="Sensor Values",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        return fig, status_display

    return {}, status_display


@callback(
    Output('preprocessed-graph', 'figure', True),
    Output('stored-datasets', 'data', True),
    Input('clean-smooth-btn', 'n_clicks'),
    State('dataset-selector_', 'value'),
    prevent_initial_call=True
)
def clean_and_smooth_data(n_clicks, dataset_name):
    """Clean and smooth the selected dataset and display it in a graph."""
    if not dataset_name:
        return {}, {}

    file_path = os.path.join(PERSISTENT_DIR, dataset_name)
    if not os.path.exists(file_path):
        return {}, {}

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


@callback(
    Output('preprocessed-graph', 'figure', True),
    Input('save-cleaned-smoothed-btn', 'n_clicks'),
    State('dataset-selector_', 'value'),
    State('stored-datasets', 'data'),
    State('preprocessed-graph', 'figure'),
    prevent_initial_call=True
)
def save_cleaned_smoothed_data(n_clicks, dataset_name, cleaned_smoothed, processed_figure):
    """Save the cleaned and smoothed dataset to a new file and update metadata."""
    if not (dataset_name and processed_figure):
        return {}

    df = pd.DataFrame(cleaned_smoothed)
    cleaned_smoothed_file_path = os.path.join(
        PERSISTENT_DIR, f"cleaned_smoothed_{dataset_name}")
    # Use 4 decimal places precision for sensor data readability
    df.to_csv(cleaned_smoothed_file_path, index=False, float_format='%.4f')

    # Update metadata
    metadata_file = os.path.join(PERSISTENT_DIR, "metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    metadata[dataset_name]["cleaned_data_path"] = cleaned_smoothed_file_path

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f)

    return processed_figure


@callback(
    Output('interactive-sample-graph', 'figure'),
    Output('current-windows', 'data', allow_duplicate=True),
    Input('apply-time-window-btn', 'n_clicks'),
    State('dataset-selector_', 'value'),
    State('time-window-span-input', 'value'),
    prevent_initial_call=True
)
def apply_time_window(n_clicks, dataset_name, time_window_span):
    """Apply the time window span and display the dataset with draggable windows."""
    if not (dataset_name and time_window_span):
        return {}, []

    with open(METADATA_FILE, 'r') as f:
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
            'window_id': window_idx,
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
            fillcolor="rgba(255, 80, 80, 0.35)",  # Enhanced opacity for better visibility
            line=dict(
                color="rgb(220, 20, 20)", 
                width=4,  # Thicker border for better grabbing
                dash="solid"
            ),
            editable=True,
            name=f"window_{window_idx}",
            layer="above",  # Ensure rectangles are above data lines
            # Enhanced label configuration for better visibility
            label=dict(
                text=f"W{window_idx}",
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


@callback(
    Output('current-windows', 'data'),
    Output('interactive-sample-graph', 'figure', allow_duplicate=True),
    Input('add-window-btn', 'n_clicks'),
    Input('remove-window-btn', 'n_clicks'),
    Input('reset-windows-btn', 'n_clicks'),
    State('current-windows', 'data'),
    State('dataset-selector_', 'value'),
    State('time-window-span-input', 'value'),
    State('interactive-sample-graph', 'figure'),
    prevent_initial_call=True
)
def manage_windows(add_clicks, remove_clicks, reset_clicks, current_windows, dataset_name, time_window_span, current_figure):
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
    with open(METADATA_FILE, 'r') as f:
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
            'window_id': len(current_windows),
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
                'window_id': window_idx,
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
            fillcolor="rgba(255, 80, 80, 0.35)",  # Enhanced opacity for better visibility
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


@callback(
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
                            updated_windows[shape_idx]['start_time'] = value
                        elif coord_type == 'x1':
                            updated_windows[shape_idx]['end_time'] = value
        
        print(f"Updated window positions: {updated_windows}")
        return updated_windows
    
    return no_update


@callback(
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
    shapes_edited = any(key.startswith('shapes[') for key in relayout_data.keys()) if relayout_data else False
    
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


@callback(
    Output('split-samples-graph', 'figure'),
    Input('split-selected-windows-btn', 'n_clicks'),
    State('dataset-selector_', 'value'),
    State('current-windows', 'data'),
    State('interactive-sample-graph', 'figure'),
    State('time-window-span-input', 'value'),
    prevent_initial_call=True
)
def split_selected_windows(n_clicks, dataset_name, current_windows, current_figure, time_window_span):
    """Split the dataset into samples based on current window positions."""
    if not (dataset_name and time_window_span):
        print("No dataset selected or time window span not specified.")
        return {}

    print("Current windows:", current_windows)

    with open(METADATA_FILE, 'r') as f:
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

            sample_file_path = os.path.join(
                PERSISTENT_DIR, f"dragged_window_{window_id}_{dataset_name}")
            # Use 4 decimal places precision for sensor data readability
            window_data[available_cols].to_csv(sample_file_path, index=False, float_format='%.4f')
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

    # Update metadata
    if 'dragged_samples' not in metadata[dataset_name]:
        metadata[dataset_name]['dragged_samples'] = []
    metadata[dataset_name]['dragged_samples'].extend(sample_files)

    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

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


@callback(
    Output('split-dataset-selector', 'options'),
    Output('split-dataset-selector', 'value'),
    Input('split-selected-windows-btn', 'n_clicks'),
    Input('dataset-selector_', 'value'),
    Input('split-samples-graph', 'figure'),  # Add this to trigger update when split completes
    State('split-dataset-selector', 'value'),
    prevent_initial_call=False
)
def update_split_dataset_selector(split_clicks, dataset_name, split_graph, current_value):
    """Update the dropdown options for split datasets."""
    if not dataset_name:
        return [], None

    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        # Get all split window files for the current dataset
        split_files = []
        if 'dragged_samples' in metadata.get(dataset_name, {}):
            split_files = metadata[dataset_name]['dragged_samples']

        # Also check for any files in persistent_data that match the pattern
        import glob
        pattern = os.path.join(
            PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")
        existing_files = glob.glob(pattern)

        # Combine and deduplicate
        all_files = list(set(split_files + existing_files))

        options = []
        for file_path in all_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                # Extract window info from filename
                parts = filename.replace(f"_{dataset_name}", "").replace(
                    "dragged_window_", "")
                window_id = parts.split("_")[0] if "_" in parts else parts

                # Get file stats
                df = pd.read_csv(file_path)
                samples = len(df)
                duration_est = samples / 100  # Assuming 100Hz sampling rate

                display_name = f"Window {window_id} ({samples} samples, ~{duration_est:.1f}s)"
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


@callback(
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


@callback(
    Output('split-dataset-selector', 'options', allow_duplicate=True),
    Output('split-dataset-selector', 'value', allow_duplicate=True),
    Input('delete-split-window-btn', 'n_clicks'),
    State('split-dataset-selector', 'value'),
    State('dataset-selector_', 'value'),
    prevent_initial_call=True
)
def delete_split_window(n_clicks, selected_file_path, dataset_name):
    """Delete the selected split window file and update metadata."""
    if not selected_file_path or not os.path.exists(selected_file_path):
        return no_update, no_update

    try:
        # Delete the file
        os.remove(selected_file_path)
        print(f"Deleted split window file: {selected_file_path}")

        # Update metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        if dataset_name in metadata and 'dragged_samples' in metadata[dataset_name]:
            if selected_file_path in metadata[dataset_name]['dragged_samples']:
                metadata[dataset_name]['dragged_samples'].remove(
                    selected_file_path)

        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)

        # Get updated options
        import glob
        pattern = os.path.join(
            PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")
        existing_files = glob.glob(pattern)

        options = []
        for file_path in existing_files:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                parts = filename.replace(f"_{dataset_name}", "").replace(
                    "dragged_window_", "")
                window_id = parts.split("_")[0] if "_" in parts else parts

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
        return no_update, no_update


@callback(
    Output('split-dataset-selector', 'options', allow_duplicate=True),
    Output('split-dataset-selector', 'value', allow_duplicate=True),
    Output('split-samples-graph', 'figure', allow_duplicate=True),
    Output('selected-split-graph', 'figure', allow_duplicate=True),
    Output('split-window-info', 'children', allow_duplicate=True),
    Output('split-window-stats-table', 'columns', allow_duplicate=True),
    Output('split-window-stats-table', 'data', allow_duplicate=True),
    Input('clean-generated-data-btn', 'n_clicks'),
    State('dataset-selector_', 'value'),
    prevent_initial_call=True
)
def clean_all_generated_data(n_clicks, dataset_name):
    """Clean all generated split window data for the current dataset."""
    if not dataset_name:
        return [], None, {}, {}, "No dataset selected.", [], []

    try:
        import glob

        # Get all split window files for the current dataset
        pattern = os.path.join(
            PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")
        split_files = glob.glob(pattern)

        # Also get sample_window files that might have been generated
        sample_pattern = os.path.join(
            PERSISTENT_DIR, f"sample_window_*_{dataset_name}")
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
        with open(METADATA_FILE, 'r') as f:
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
        with open(METADATA_FILE, 'w') as f:
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


@callback(
    Output('training-dataset-selector', 'options'),
    Output('training-dataset-info', 'children'),
    Input('dataset-selector_', 'value'),
    Input('split-samples-graph', 'figure'),  # Add this to trigger update when split completes
    Input('split-dataset-selector', 'options'),  # Add this to trigger when split options change
    prevent_initial_call=False
)
def populate_training_dataset_selector(dataset_name, split_graph, split_options):
    """Populate training dataset selector with available split windows."""
    if not dataset_name:
        return [], "No dataset selected."

    try:
        import glob

        # Get all split window files for the current dataset
        pattern = os.path.join(
            PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")
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
                filename = os.path.basename(file_path)
                # Extract window info from filename
                parts = filename.replace(f"_{dataset_name}", "").replace(
                    "dragged_window_", "")
                window_id = parts.split("_")[0] if "_" in parts else parts

                # Get file stats
                df = pd.read_csv(file_path)
                samples = len(df)
                duration_est = samples / 100  # Assuming 100Hz sampling rate
                total_samples += samples

                display_name = f"Window {window_id} ({samples} samples, ~{duration_est:.1f}s)"
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


@callback(
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


@callback(
    Output('split-ratio-info', 'children'),
    Input('train-test-split', 'value')
)
def update_split_ratio_info(split_ratio):
    """Update split ratio information display."""
    if not split_ratio:
        return ""

    train_percent = int(split_ratio * 100)
    test_percent = 100 - train_percent

    return html.Div([
        html.Span(f"🎓 Training: {train_percent}%", style={
                  'margin-right': '20px', 'color': '#4CAF50', 'font-weight': 'bold'}),
        html.Span(f"🧪 Testing: {test_percent}%", style={
                  'color': '#2196F3', 'font-weight': 'bold'})
    ])


@callback(
    Output('preprocessed-training-data', 'data'),
    Output('preprocessing-results', 'children'),
    Input('preprocess-for-training-btn', 'n_clicks'),
    State('training-dataset-selector', 'value'),
    State('normalization-method', 'value'),
    State('feature-selection-method', 'value'),
    State('dataset-selector_', 'value'),
    prevent_initial_call=True
)
def preprocess_for_training(n_clicks, selected_files, norm_method, feature_method, dataset_name):
    """Preprocess selected split windows for model training."""
    if not (selected_files and norm_method and feature_method):
        return {}, html.Div("⚠️ Please select datasets and preprocessing options.", style={'color': '#FF9800'})

    try:
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
                # Use dataset name as activity label
                df['Activity_Label'] = dataset_name

                all_data.append(df)
                file_info.append({
                    'window_id': window_id,
                    'samples': len(df),
                    'file_path': file_path
                })

        if not all_data:
            return {}, html.Div("❌ No valid data files found.", style={'color': '#dc3545'})

        # Combine all data
        combined_df = pd.concat(all_data, ignore_index=True)

        # Feature selection
        if feature_method == 'all':
            # Use all sensor columns
            feature_cols = [col for col in combined_df.columns if col not in [
                'Time_seconds', 'Window_ID', 'Activity_Label']]
        elif feature_method == 'statistical':
            # Use only statistical features (if available) or primary sensors
            priority_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
            feature_cols = [
                col for col in priority_cols if col in combined_df.columns]
        elif feature_method == 'time_domain':
            # Use time-domain features only
            feature_cols = [col for col in combined_df.columns if col not in ['Time_seconds', 'Window_ID',
                                                                              'Activity_Label'] and not any(freq in col.lower() for freq in ['freq', 'fft', 'spectrum'])]
        else:  # custom
            feature_cols = [col for col in combined_df.columns if col not in [
                'Time_seconds', 'Window_ID', 'Activity_Label']]

        if not feature_cols:
            return {}, html.Div("❌ No features available for training.", style={'color': '#dc3545'})

        # Extract features
        X = combined_df[feature_cols].values
        y = combined_df['Activity_Label'].values
        window_ids = combined_df['Window_ID'].values

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
                    html.Span("🎯 Features: ", style={'font-weight': 'bold'}),
                    html.Span(f"{len(feature_cols)} ({feature_method})")
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
        error_msg = f"❌ Error during preprocessing: {str(e)}"
        return {}, html.Div(error_msg, style={'color': '#dc3545'})


@callback(
    Output('train-test-data', 'data'),
    Output('train-test-split-graph', 'figure'),
    Input('train-test-split-btn', 'n_clicks'),
    State('preprocessed-training-data', 'data'),
    State('train-test-split', 'value'),
    State('random-state-input', 'value'),
    prevent_initial_call=True
)
def perform_enhanced_train_test_split(n_clicks, preprocessed_data, split_ratio, random_state):
    """Perform train-test split on preprocessed data and visualize results."""
    if not (preprocessed_data and split_ratio):
        return {}, {}

    try:
        # Extract data
        X = np.array(preprocessed_data['features'])
        y = np.array(preprocessed_data['labels'])
        feature_names = preprocessed_data['feature_names']

        # Perform train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=1-split_ratio, random_state=random_state, stratify=y
        )

        # Store split data
        split_data = {
            'X_train': X_train.tolist(),
            'X_test': X_test.tolist(),
            'y_train': y_train.tolist(),
            'y_test': y_test.tolist(),
            'feature_names': feature_names,
            'split_ratio': split_ratio,
            'random_state': random_state
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

            # Training data
            fig.add_trace(go.Scatter(
                x=list(range(len(X_train))),
                y=X_train[:, feature_idx],
                mode='markers',
                name=f'{feature} (Train)',
                marker=dict(color=colors[i % len(colors)],
                            symbol='circle', size=4),
                opacity=0.7
            ))

            # Test data
            fig.add_trace(go.Scatter(
                x=list(range(len(X_train), len(X_train) + len(X_test))),
                y=X_test[:, feature_idx],
                mode='markers',
                name=f'{feature} (Test)',
                marker=dict(color=colors[i % len(colors)],
                            symbol='diamond', size=4),
                opacity=0.7
            ))

        # Add vertical line to separate train/test
        fig.add_vline(
            x=len(X_train)-0.5,
            line=dict(color="red", width=2, dash="dash"),
            annotation_text="Train | Test Split",
            annotation_position="top"
        )

        # Update layout
        fig.update_layout(
            title=f"Train-Test Split Visualization<br>"
            f"Training: {len(X_train)} samples ({split_ratio:.1%}) | "
            f"Testing: {len(X_test)} samples ({1-split_ratio:.1%})",
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
        fig.add_annotation(
            text=f"📊 Features: {len(feature_names)}<br>"
            f"🎓 Train: {len(X_train)} samples<br>"
            f"🧪 Test: {len(X_test)} samples<br>"
            f"🎲 Random State: {random_state}",
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


@callback(
    Output('preprocessing-results', 'children', allow_duplicate=True),
    Input('save-preprocessed-training-btn', 'n_clicks'),
    State('train-test-data', 'data'),
    State('dataset-selector_', 'value'),
    prevent_initial_call=True
)
def save_preprocessed_training_data(n_clicks, split_data, dataset_name):
    """Save preprocessed training data to files."""
    if not (split_data and dataset_name):
        return html.Div("⚠️ No preprocessed data to save.", style={'color': '#FF9800'})

    try:
        # Create training data directory
        training_dir = os.path.join(PERSISTENT_DIR, 'training_data')
        os.makedirs(training_dir, exist_ok=True)

        # Save training and test data
        train_df = pd.DataFrame(
            split_data['X_train'], columns=split_data['feature_names'])
        train_df['label'] = split_data['y_train']

        test_df = pd.DataFrame(
            split_data['X_test'], columns=split_data['feature_names'])
        test_df['label'] = split_data['y_test']

        # Save files
        train_file = os.path.join(training_dir, f"{dataset_name}_train.csv")
        test_file = os.path.join(training_dir, f"{dataset_name}_test.csv")

        train_df.to_csv(train_file, index=False)
        test_df.to_csv(test_file, index=False)

        # Save metadata
        metadata_file = os.path.join(
            training_dir, f"{dataset_name}_metadata.json")
        metadata = {
            'dataset_name': dataset_name,
            'feature_names': split_data['feature_names'],
            'split_ratio': split_data['split_ratio'],
            'random_state': split_data['random_state'],
            'train_samples': len(split_data['X_train']),
            'test_samples': len(split_data['X_test']),
            'train_file': train_file,
            'test_file': test_file,
            'created_at': pd.Timestamp.now().isoformat()
        }

        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Success message
        return html.Div([
            html.H5("💾 Training Data Saved Successfully", style={
                    'color': '#4CAF50', 'margin': '0 0 15px 0'}),
            html.Div([
                html.Div([
                    html.Span("📁 Training File: ", style={
                              'font-weight': 'bold'}),
                    html.Span(os.path.basename(train_file),
                              style={'font-family': 'monospace'})
                ], style={'margin-bottom': '5px'}),
                html.Div([
                    html.Span("📁 Test File: ", style={'font-weight': 'bold'}),
                    html.Span(os.path.basename(test_file),
                              style={'font-family': 'monospace'})
                ], style={'margin-bottom': '5px'}),
                html.Div([
                    html.Span("📄 Metadata: ", style={'font-weight': 'bold'}),
                    html.Span(os.path.basename(metadata_file),
                              style={'font-family': 'monospace'})
                ], style={'margin-bottom': '10px'}),
                html.P("✅ Data is ready for model training!", style={
                       'color': '#4CAF50', 'font-weight': 'bold'})
            ])
        ])

    except Exception as e:
        error_msg = f"❌ Error saving preprocessed data: {str(e)}"
        return html.Div(error_msg, style={'color': '#dc3545'})


@callback(
    Output('preprocessed-training-data', 'data', allow_duplicate=True),
    Output('train-test-data', 'data', allow_duplicate=True),
    Output('train-test-split-graph', 'figure', allow_duplicate=True),
    Output('preprocessing-results', 'children', allow_duplicate=True),
    Input('clear-training-data-btn', 'n_clicks'),
    State('dataset-selector_', 'value'),
    prevent_initial_call=True
)
def clear_training_data(n_clicks, dataset_name):
    """Clear all preprocessed training data and associated files."""
    if not dataset_name:
        return {}, {}, {}, html.Div("⚠️ No dataset selected.", style={'color': '#FF9800'})
    
    try:
        # Clear training data directory for this dataset
        training_dir = os.path.join(PERSISTENT_DIR, 'training_data')
        
        if os.path.exists(training_dir):
            # Remove files related to current dataset
            patterns = [
                f"{dataset_name}_train.csv",
                f"{dataset_name}_test.csv", 
                f"{dataset_name}_metadata.json"
            ]
            
            deleted_files = []
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


@callback(
    Output('dataset-status-display', 'children', allow_duplicate=True),
    Input('save-cleaned-smoothed-btn', 'n_clicks'),
    Input('split-selected-windows-btn', 'n_clicks'),
    Input('save-preprocessed-training-btn', 'n_clicks'),
    Input('clear-training-data-btn', 'n_clicks'),
    State('dataset-selector_', 'value'),
    prevent_initial_call=True
)
def update_dataset_status_on_operations(clean_clicks, split_clicks, save_clicks, clear_clicks, dataset_name):
    """Update dataset status when any processing operation completes."""
    if not dataset_name:
        return no_update
    
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        dataset_info = metadata.get(dataset_name, {})
        
        # Clean up metadata by removing non-existent files
        if 'dragged_samples' in dataset_info:
            existing_files = [f for f in dataset_info['dragged_samples'] if os.path.exists(f)]
            if len(existing_files) != len(dataset_info['dragged_samples']):
                # Update metadata to remove stale references
                metadata[dataset_name]['dragged_samples'] = existing_files
                with open(METADATA_FILE, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Cleaned metadata for {dataset_name}: removed {len(dataset_info['dragged_samples']) - len(existing_files)} stale references")
                dataset_info['dragged_samples'] = existing_files
        
        # Determine processing stages
        stages = {
            'raw': True,
            'preprocessed': 'cleaned_data_path' in dataset_info,
            'split': 'dragged_samples' in dataset_info and len(dataset_info.get('dragged_samples', [])) > 0,
            'training_ready': False
        }
        
        # Check if training data exists
        training_dir = os.path.join(PERSISTENT_DIR, 'training_data')
        train_file = os.path.join(training_dir, f"{dataset_name}_train.csv")
        test_file = os.path.join(training_dir, f"{dataset_name}_test.csv")
        stages['training_ready'] = os.path.exists(train_file) and os.path.exists(test_file)
        
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
            actual_split_count = sum(1 for file_path in dragged_samples if os.path.exists(file_path))
            
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
