import os
import json
import pandas as pd
import base64
import io
import plotly.express as px
import plotly.graph_objects as go
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

            # Get sensor columns (exclude Time_seconds)
            sensor_cols = [col for col in df.columns if col != 'Time_seconds']
            
            # Limit to first 6 columns for better visualization
            sensor_cols = sensor_cols[:6]
            
            if sensor_cols:
                # Generate a line chart with no template to avoid compatibility issues
                fig = px.line(df, x="Time_seconds", y=sensor_cols,
                              title=f"Filtered Preview of {dataset_name}",
                              template=None)
                fig.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Sensor Values",
                    height=400,
                    showlegend=True
                )
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
            
            # Get sensor columns (exclude Time_seconds)
            sensor_cols = [col for col in df.columns if col != 'Time_seconds']
            
            # Limit to first 6 columns for better visualization
            sensor_cols = sensor_cols[:6]
            
            if sensor_cols:
                # Generate a line chart with no template to avoid compatibility issues
                fig = px.line(df, x='Time_seconds', y=sensor_cols, 
                              title=f"Preview of {dataset_name}",
                              template=None)
                fig.update_layout(
                    xaxis_title="Time (seconds)",
                    yaxis_title="Sensor Values",
                    height=400,
                    showlegend=True
                )
                return fig
    # Return an empty figure if no dataset is selected
    return {}


@callback(
    Output('upload-output', 'children', allow_duplicate=True),
    Output('dataset-selector', 'options', allow_duplicate=True),
    Input('delete-dataset-btn', 'n_clicks'),
    State('dataset-selector', 'value'),
    prevent_initial_call=True
)
def delete_specific_dataset(n_clicks, dataset_name):
    """Delete a specific dataset and update metadata."""
    if not dataset_name:
        return "⚠️ Please select a dataset to delete.", no_update
    
    try:
        # Load existing metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        if dataset_name not in metadata:
            return f"⚠️ Dataset '{dataset_name}' not found in metadata.", no_update
        
        # Delete the file
        file_path = metadata[dataset_name].get("path", "")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
        
        # Delete any cleaned data files
        cleaned_path = metadata[dataset_name].get("cleaned_data_path", "")
        if cleaned_path and os.path.exists(cleaned_path):
            os.remove(cleaned_path)
        
        # Delete any split window files
        import glob
        split_pattern = os.path.join(PERSISTENT_DIR, f"dragged_window_*_{dataset_name}")
        split_files = glob.glob(split_pattern)
        for split_file in split_files:
            if os.path.exists(split_file):
                os.remove(split_file)
        
        # Remove from metadata
        del metadata[dataset_name]
        
        # Save updated metadata
        with open(METADATA_FILE, 'w') as f:
            json.dump(metadata, f)
        
        # Update dropdown options
        options = [{'label': filename, 'value': filename} for filename in metadata.keys()]
        
        return f"✅ Successfully deleted dataset '{dataset_name}' and all associated files.", options
        
    except Exception as e:
        return f"❌ Error deleting dataset: {str(e)}", no_update


@callback(
    Output('dataset-info-table', 'children'),
    Input('dataset-selector', 'value'),
    prevent_initial_call=False
)
def update_dataset_info_table(dataset_name):
    """Update the dataset information table."""
    if not dataset_name:
        return html.Div("Select a dataset to view information", style={'color': '#666', 'font-style': 'italic'})
    
    try:
        # Load metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        if dataset_name not in metadata:
            return html.Div("Dataset not found in metadata", style={'color': '#dc3545'})
        
        dataset_info = metadata[dataset_name]
        file_path = dataset_info.get("path", "")
        
        # Get file information
        file_stats = {}
        if file_path and os.path.exists(file_path):
            df = pd.read_csv(file_path)
            file_size = os.path.getsize(file_path)
            
            file_stats = {
                'File Size': f"{file_size / 1024:.1f} KB",
                'Rows': f"{len(df):,}",
                'Columns': f"{len(df.columns)}",
                'Duration (est.)': f"{len(df) / dataset_info.get('sampling_rate', 100):.1f} seconds",
                'Memory Usage': f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB"
            }
            
            # Get column names
            sensor_cols = [col for col in df.columns if col not in ['Time_seconds', 'Window']]
            file_stats['Sensor Columns'] = ', '.join(sensor_cols[:6]) + ('...' if len(sensor_cols) > 6 else '')
        
        # Create info cards
        info_cards = []
        
        # Basic info card
        basic_info = [
            {'Property': 'Label', 'Value': dataset_info.get('label', 'Not assigned')},
            {'Property': 'Sampling Rate', 'Value': f"{dataset_info.get('sampling_rate', 'Not set')} Hz"},
            {'Property': 'File Path', 'Value': os.path.basename(file_path) if file_path else 'Not found'},
            {'Property': 'Has Cleaned Data', 'Value': 'Yes' if dataset_info.get('cleaned_data_path') else 'No'}
        ]
        
        # Add file stats
        for key, value in file_stats.items():
            basic_info.append({'Property': key, 'Value': value})
        
        # Create table
        table = dash_table.DataTable(
            data=basic_info,
            columns=[
                {"name": "Property", "id": "Property"},
                {"name": "Value", "id": "Value"}
            ],
            style_cell={
                'textAlign': 'left',
                'padding': '12px',
                'fontFamily': 'Arial',
                'border': '1px solid #dee2e6'
            },
            style_header={
                'backgroundColor': '#2E86AB',
                'color': 'white',
                'fontWeight': 'bold',
                'textAlign': 'center'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': '#f8f9fa'
                }
            ],
            style_table={'margin-top': '10px'}
        )
        
        return table
        
    except Exception as e:
        return html.Div(f"Error loading dataset info: {str(e)}", style={'color': '#dc3545'})


@callback(
    Output('upload-output', 'children', allow_duplicate=True),
    Input('export-metadata-btn', 'n_clicks'),
    prevent_initial_call=True
)
def export_metadata(n_clicks):
    """Export metadata to a downloadable JSON file."""
    try:
        # Load metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        # Create export filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"HAR_metadata_export_{timestamp}.json"
        export_path = os.path.join(PERSISTENT_DIR, export_filename)
        
        # Add export timestamp to metadata
        export_data = {
            'export_timestamp': timestamp,
            'total_datasets': len(metadata),
            'datasets': metadata
        }
        
        # Save export file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return html.Div([
            html.P("✅ Metadata exported successfully!", style={'color': '#28a745', 'font-weight': 'bold'}),
            html.P(f"📁 File saved as: {export_filename}", style={'color': '#666', 'font-size': '14px'})
        ])
        
    except Exception as e:
        return f"❌ Error exporting metadata: {str(e)}"
