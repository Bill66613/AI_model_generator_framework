import os
import json
import pandas as pd
import base64
import io
import plotly.express as px
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx, no_update

from config.config import *

def parse_contents(contents, filename):
    """Parse uploaded CSV file contents into a DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df


@callback(
    Output('upload-output', 'children'),
    Output('dataset-selector', 'options', True),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    prevent_initial_call=True
)
def upload_files(contents, filenames):
    """Save uploaded files to the persistent directory."""
    if not contents:
        return "No file uploaded.", []

    # Ensure contents and filenames are lists
    if isinstance(contents, str):
        contents = [contents]
        filenames = [filenames]

    # Process each file
    for content, filename in zip(contents, filenames):
        try:
            df = parse_contents(content, filename)
            if df.empty:
                raise ValueError("The uploaded file is empty.")

            # Save the file to the server
            file_path = os.path.join(PERSISTENT_DIR, filename)
            df.to_csv(file_path, index=False)

        except ValueError as ve:
            return f"⚠ {str(ve)}", no_update
        except Exception as e:
            return f"⚠ Error processing {filename}: {str(e)}", no_update

    # Update metadata
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    for filename in filenames:
        file_path = os.path.join(PERSISTENT_DIR, filename)
        metadata[filename] = {
            "path": file_path,
            "label": filename.replace('.csv', ''),
            'sampling_rate': 100
        }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

    # Update dropdown options
    options = [{'label': filename, 'value': filename} for filename in metadata.keys()]
    return f"✅ Successfully uploaded {len(filenames)} file(s)", options


@callback(
    Output('upload-output', 'children', True),
    Input('save-sampling-rate-btn', 'n_clicks'),
    State('dataset-selector', 'value'),
    State('sampling-rate-input', 'value'),
    prevent_initial_call=True
)
def save_sampling_rate(n_clicks, dataset_name, sampling_rate):
    """Save the sampling rate to the metadata file."""
    if not (dataset_name and sampling_rate):
        return "⚠ Please select a dataset and enter a valid sampling rate."

    try:
        # Load existing metadata
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update the sampling rate for the selected dataset
        if dataset_name in metadata:
            metadata[dataset_name]['sampling_rate'] = sampling_rate
        else:
            metadata[dataset_name] = {'sampling_rate': sampling_rate}

        # Save updated metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)

        return f"✅ Sampling rate for '{dataset_name}' updated to {sampling_rate} Hz."
    
    except (json.JSONDecodeError, IOError) as e:
        return f"❌ Error saving sampling rate: {str(e)}"


@callback(
    Output('dataset-selector', 'options'),
    Input('tabs', 'value')
)
def data_selector_options(tab):
    """Update the dataset selector options."""
    try:
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
            return [{'label': filename, 'value': filename} for filename in metadata.keys()]
        return []
    except (json.JSONDecodeError, IOError) as e:
        print(f"Error reading metadata: {e}")
        return []


@callback(
    Output('upload-output', 'children', True),
    Input('save-label-btn', 'n_clicks'),
    State('dataset-selector', 'value'),
    State('dataset-label', 'value'),
    prevent_initial_call=True
)
def save_label(n_clicks, dataset_name, label):
    """Save the label assigned to a dataset."""
    if not (dataset_name and label):
        return "⚠ Please select a dataset and enter a label."

    try:
        # Load existing metadata
        if os.path.exists(METADATA_FILE):
            with open(METADATA_FILE, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update metadata
        if dataset_name not in metadata:
            metadata[dataset_name] = {}
        metadata[dataset_name]["label"] = label

        # Save updated metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)

        return f"✅ Label '{label}' saved for dataset '{dataset_name}'."
    
    except (json.JSONDecodeError, IOError) as e:
        return f"❌ Error saving label: {str(e)}"


@callback(
    Output('upload-output', 'children', True),
    Input('clear-data-btn', 'n_clicks'),
    prevent_initial_call=True
)
def clear_data(n_clicks):
    """Clear all stored data and metadata."""
    # Remove all files in the persistent directory
    for file in os.listdir(PERSISTENT_DIR):
        file_path = os.path.join(PERSISTENT_DIR, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # Reset metadata
    with open(METADATA_FILE, 'w') as f:
        json.dump({}, f)

    return "✅ All data and metadata have been cleared."


@callback(
    Output('data-preview', 'figure', True),
    Input('dataset-selector', 'value'),
    prevent_initial_call=True
)
def filter_and_display_data(dataset_name):
    """Filters the dataset and displays it as a chart."""
    if dataset_name:
        # Load existing metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        sampling_rate = metadata.get(
            dataset_name, {}).get("sampling_rate", 100)

        file_path = os.path.join(PERSISTENT_DIR, dataset_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Add a time axis based on the sampling rate
            df['Time_seconds'] = pd.Series(range(df.shape[0])) / sampling_rate

            # Generate a line chart as an example
            fig = px.line(df, x="Time_seconds",
                          title=f"Filtered Preview of {dataset_name}")
            return fig
    return {}


@callback(
    Output('assigned-label', 'value'),
    Input('dataset-selector', 'value')
)
def display_label(dataset_name):
    """Displays the label assigned to a dataset."""
    if dataset_name:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return metadata.get(dataset_name, {}).get("label", "")
    return ""


@callback(
    Output('data-preview', 'figure'),
    Input('dataset-selector', 'value')
)
def display_dataset(dataset_name):
    """Displays selected dataset as a chart."""
    if dataset_name:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        sampling_rate = metadata.get(
            dataset_name, {}).get("sampling_rate", 100)

        file_path = metadata[dataset_name]["path"]
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Add a time axis based on the sampling rate
            df['Time_seconds'] = pd.Series(range(df.shape[0])) / sampling_rate
            # Generate a line chart as an example
            fig = px.line(df, x='Time_seconds', y=[df.columns[0], df.columns[1], df.columns[2],
                          df.columns[3], df.columns[4], df.columns[5]], title=f"Preview of {dataset_name}")
            return fig
    # Return an empty figure if no dataset is selected
    return {}
