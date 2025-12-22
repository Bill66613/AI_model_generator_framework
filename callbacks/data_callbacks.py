import os
import json
import pandas as pd
import base64
import io
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, callback, dash_table, ctx, no_update

from config.config import *
from config.config import get_window_pattern


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
    State('working-directory-store', 'data'),
    prevent_initial_call=True
)
def upload_files(contents, filenames, base_dir):
    """Save uploaded files to the datasets subdirectory of the persistent directory."""
    if not contents:
        return "No file uploaded.", []

    # Use stored base directory or default to PERSISTENT_DIR
    if not base_dir:
        base_dir = PERSISTENT_DIR
    
    # Files should be saved to datasets subdirectory
    datasets_dir = os.path.join(base_dir, 'datasets')
    os.makedirs(datasets_dir, exist_ok=True)

    # Ensure contents and filenames are lists
    if isinstance(contents, str):
        contents = [contents]
        filenames = [filenames]

    # Track actual saved filenames (in case of renames)
    saved_filenames = []

    # Process each file
    for content, filename in zip(contents, filenames):
        try:
            original_filename = filename
            # If file name already exists, rename the new file
            file_path = os.path.join(datasets_dir, filename)
            if os.path.exists(file_path):
                base, ext = os.path.splitext(filename)
                count = 1
                while os.path.exists(file_path):
                    filename = f"{base}_{count}{ext}"
                    file_path = os.path.join(datasets_dir, filename)
                    count += 1

            df = parse_contents(content, original_filename)
            if df.empty:
                raise ValueError("The uploaded file is empty.")

            # Save the file to the datasets directory
            df.to_csv(file_path, index=False)
            
            # Track the actual saved filename
            saved_filenames.append(filename)

        except ValueError as ve:
            return f"⚠ {str(ve)}", no_update
        except Exception as e:
            return f"⚠ Error processing {filename}: {str(e)}", no_update

    # Update metadata
    # Use base_dir to get the correct metadata file path
    metadata_file = os.path.join(base_dir, 'metadata.json')
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Use saved_filenames (which includes renamed files) instead of original filenames
    for filename in saved_filenames:
        file_path = os.path.join(datasets_dir, filename)
        metadata[filename] = {
            "path": file_path,
            "label": filename.replace('.csv', ''),
            'sampling_rate': 100
        }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Update dropdown options
    options = [{'label': filename, 'value': filename}
               for filename in metadata.keys()]
    return f"✅ Successfully uploaded {len(saved_filenames)} file(s) to {datasets_dir}", options


@callback(
    Output('upload-output', 'children', True),
    Input('save-sampling-rate-btn', 'n_clicks'),
    State('dataset-selector', 'value'),
    State('sampling-rate-input', 'value'),
    State('working-directory-store', 'data'),
    prevent_initial_call=True
)
def save_sampling_rate(n_clicks, dataset_name, sampling_rate, base_dir):
    """Save the sampling rate to the metadata file."""
    if not (dataset_name and sampling_rate):
        return "⚠ Please select a dataset and enter a valid sampling rate."

    try:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        # Load existing metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update the sampling rate for the selected dataset
        if dataset_name in metadata:
            metadata[dataset_name]['sampling_rate'] = sampling_rate
        else:
            metadata[dataset_name] = {'sampling_rate': sampling_rate}

        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        return f"✅ Sampling rate for '{dataset_name}' updated to {sampling_rate} Hz."

    except (json.JSONDecodeError, IOError) as e:
        return f"❌ Error saving sampling rate: {str(e)}"


@callback(
    Output('dataset-selector', 'options'),
    Input('tabs', 'value'),
    Input('working-directory-store', 'data')
)
def data_selector_options(tab, base_dir):
    """Update the dataset selector options."""
    try:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
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
    State('working-directory-store', 'data'),
    prevent_initial_call=True
)
def save_label(n_clicks, dataset_name, label, base_dir):
    """Save the label assigned to a dataset."""
    if not (dataset_name and label):
        return "⚠ Please select a dataset and enter a label."

    try:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        # Load existing metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}

        # Update metadata
        if dataset_name not in metadata:
            metadata[dataset_name] = {}
        metadata[dataset_name]["label"] = label

        # Save updated metadata
        with open(metadata_file, 'w') as f:
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
    State('working-directory-store', 'data'),
    prevent_initial_call=True
)
def filter_and_display_data(dataset_name, base_dir):
    """Filters the dataset and displays it as a chart."""
    if dataset_name:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        # Load existing metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        sampling_rate = metadata.get(
            dataset_name, {}).get("sampling_rate", 100)

        file_path = metadata[dataset_name].get("path")
        if not file_path:
            file_path = os.path.join(base_dir, 'datasets', dataset_name)
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)

            # Add a time axis based on the sampling rate
            df['Time_seconds'] = pd.Series(range(df.shape[0])) / sampling_rate

            # Get sensor columns (exclude Time_seconds)
            sensor_cols = [col for col in df.columns if col != 'Time_seconds']

            # Limit to first 6 columns for better visualization
            sensor_cols = sensor_cols[:6]

            if sensor_cols:
                # Generate a line chart with plotly_white template to avoid compatibility issues
                fig = px.line(df, x="Time_seconds", y=sensor_cols,
                              title=f"Filtered Preview of {dataset_name}",
                              template="plotly_white")
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
    Input('dataset-selector', 'value'),
    State('working-directory-store', 'data')
)
def display_label(dataset_name, base_dir):
    """Displays the label assigned to a dataset."""
    if dataset_name:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata.get(dataset_name, {}).get("label", "")
    return ""


@callback(
    Output('data-preview', 'figure'),
    Input('dataset-selector', 'value'),
    State('working-directory-store', 'data')
)
def display_dataset(dataset_name, base_dir):
    """Displays selected dataset as a chart."""
    if dataset_name:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            return {}

        sampling_rate = metadata.get(
            dataset_name, {}).get("sampling_rate", 100)

        file_path = metadata[dataset_name].get("path")
        if not file_path:
            file_path = os.path.join(base_dir, 'datasets', dataset_name)
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
    State('working-directory-store', 'data'),
    prevent_initial_call=True
)
def delete_specific_dataset(n_clicks, dataset_name, base_dir):
    """Delete a specific dataset and update metadata."""
    if not dataset_name:
        return "⚠️ Please select a dataset to delete.", no_update

    try:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        # Load existing metadata
        with open(metadata_file, 'r') as f:
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
        split_pattern = get_window_pattern(dataset_name)
        split_files = glob.glob(split_pattern)
        for split_file in split_files:
            if os.path.exists(split_file):
                os.remove(split_file)

        # Remove from metadata
        del metadata[dataset_name]

        # Save updated metadata
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)

        # Update dropdown options
        options = [{'label': filename, 'value': filename}
                   for filename in metadata.keys()]

        return f"✅ Successfully deleted dataset '{dataset_name}' and all associated files.", options

    except Exception as e:
        return f"❌ Error deleting dataset: {str(e)}", no_update


@callback(
    Output('dataset-info-table', 'children'),
    Input('dataset-selector', 'value'),
    State('working-directory-store', 'data'),
    prevent_initial_call=False
)
def update_dataset_info_table(dataset_name, base_dir):
    """Update the dataset information table."""
    if not dataset_name:
        return html.Div("Select a dataset to view information", style={'color': '#666', 'font-style': 'italic'})

    try:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        # Load metadata
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            return html.Div("Metadata not found", style={'color': '#dc3545'})

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
            sensor_cols = [col for col in df.columns if col not in [
                'Time_seconds', 'Window']]
            file_stats['Sensor Columns'] = ', '.join(
                sensor_cols[:6]) + ('...' if len(sensor_cols) > 6 else '')

        # Create info cards
        info_cards = []

        # Basic info card
        basic_info = [
            {'Property': 'Label', 'Value': dataset_info.get(
                'label', 'Not assigned')},
            {'Property': 'Sampling Rate',
                'Value': f"{dataset_info.get('sampling_rate', 'Not set')} Hz"},
            {'Property': 'File Path', 'Value': os.path.basename(
                file_path) if file_path else 'Not found'},
            {'Property': 'Has Cleaned Data', 'Value': 'Yes' if dataset_info.get(
                'cleaned_data_path') else 'No'}
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
    State('working-directory-store', 'data'),
    prevent_initial_call=True
)
def export_metadata(n_clicks, base_dir):
    """Export metadata to a downloadable JSON file."""
    try:
        if not base_dir:
            base_dir = PERSISTENT_DIR
        metadata_file = os.path.join(base_dir, 'metadata.json')
        
        # Load metadata
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Create export filename
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        export_filename = f"HAR_metadata_export_{timestamp}.json"
        export_path = os.path.join(base_dir, export_filename)

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
            html.P("✅ Metadata exported successfully!", style={
                   'color': '#28a745', 'font-weight': 'bold'}),
            html.P(f"📁 File saved as: {export_filename}", style={
                   'color': '#666', 'font-size': '14px'})
        ])

    except Exception as e:
        return f"❌ Error exporting metadata: {str(e)}"

# ==================== Working Directory Management ====================


@callback(
    Output('current-working-dir-display', 'children'),
    Output('working-directory-store', 'data'),
    Output('directory-path-input', 'value'),
    Input('working-directory-store', 'data'),
    prevent_initial_call=False
)
def display_current_working_dir(stored_dir):
    """Display the current persistent directory on page load and populate input."""
    if not stored_dir:
        # Default to PERSISTENT_DIR on first load
        return str(PERSISTENT_DIR), str(PERSISTENT_DIR), str(PERSISTENT_DIR)
    return stored_dir, stored_dir, stored_dir


@callback(
    Output('current-working-dir-display', 'children', allow_duplicate=True),
    Output('working-directory-store', 'data', allow_duplicate=True),
    Output('directory-path-input', 'value', allow_duplicate=True),
    Output('working-dir-status', 'children'),
    Input('select-working-dir-btn', 'n_clicks'),
    Input('use-default-dir-btn', 'n_clicks'),
    State('directory-path-input', 'value'),
    prevent_initial_call=True
)
def set_working_directory(apply_clicks, default_clicks, input_path):
    """Set persistent base directory and create standard subdirectories."""
    if not ctx.triggered:
        return no_update, no_update, no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        if button_id == 'use-default-dir-btn':
            # Use default persistent directory
            base_dir = str(PERSISTENT_DIR)

        elif button_id == 'select-working-dir-btn':
            # Apply user-entered directory
            if not input_path or not input_path.strip():
                return (
                    no_update,
                    no_update,
                    no_update,
                    html.P("⚠️ Please enter a directory path",
                           style={'color': '#ffc107', 'font-weight': 'bold'})
                )

            # Normalize path
            base_dir = os.path.normpath(input_path.strip())

            # Validate path
            if not os.path.isabs(base_dir):
                return (
                    no_update,
                    no_update,
                    no_update,
                    html.P("⚠️ Please enter an absolute path (e.g., D:\\persistent_data or C:\\project\\data)",
                           style={'color': '#ffc107', 'font-weight': 'bold'})
                )
        else:
            return no_update, no_update, no_update, no_update

        # Create base directory
        os.makedirs(base_dir, exist_ok=True)

        # Create standard subdirectories
        subdirs = ['datasets', 'models', 'training', 'windows']
        created_dirs = []

        for subdir in subdirs:
            subdir_path = os.path.join(base_dir, subdir)
            os.makedirs(subdir_path, exist_ok=True)
            created_dirs.append(subdir)

        # Create metadata.json if it doesn't exist
        metadata_path = os.path.join(base_dir, 'metadata.json')
        if not os.path.exists(metadata_path):
            with open(metadata_path, 'w') as f:
                json.dump({}, f)

        # Build status message
        status_message = html.Div([
            html.P("✅ Persistent directory configured successfully",
                   style={'color': '#28a745', 'font-weight': 'bold', 'margin': '0'}),
            html.P(f"📁 Base: {base_dir}",
                   style={'color': '#666', 'font-size': '14px', 'margin': '5px 0'}),
            html.P(f"📂 Created: {', '.join(subdirs)}, metadata.json",
                   style={'color': '#666', 'font-size': '13px', 'margin': '5px 0 0 0'})
        ])

        return (
            base_dir,
            base_dir,
            base_dir,
            status_message
        )

    except Exception as e:
        return (
            no_update,
            no_update,
            no_update,
            html.P(f"❌ Error: {str(e)}", style={
                   'color': '#dc3545', 'font-weight': 'bold'})
        )

    return no_update, no_update, no_update, no_update


@callback(
    Output('working-dir-status', 'children', allow_duplicate=True),
    Input('migrate-files-btn', 'n_clicks'),
    State('working-directory-store', 'data'),
    prevent_initial_call=True
)
def migrate_old_files(n_clicks, base_dir):
    """Migrate CSV files from persistent base directory root to datasets subfolder."""
    if not n_clicks:
        return no_update

    try:
        # Use stored base directory or default to PERSISTENT_DIR
        source_dir = base_dir if base_dir else PERSISTENT_DIR
        datasets_dir = os.path.join(source_dir, 'datasets')
        os.makedirs(datasets_dir, exist_ok=True)

        # Find all CSV files in base directory root (not in subdirectories)
        # Use os.listdir for consistency with framework's string-based paths
        all_items = os.listdir(source_dir)
        root_csv_files = [
            f for f in all_items
            if os.path.isfile(os.path.join(source_dir, f)) and f.lower().endswith('.csv')
        ]

        if not root_csv_files:
            return html.Div([
                html.P("ℹ️ No CSV files found in base directory root",
                       style={'color': '#17a2b8', 'font-weight': 'bold'}),
                html.P("All files are already organized!",
                       style={'color': '#666', 'font-size': '14px'})
            ])

        # Migrate files
        migrated = []
        skipped = []
        errors = []

        for csv_file in root_csv_files:
            try:
                source_path = os.path.join(source_dir, csv_file)
                target_path = os.path.join(datasets_dir, csv_file)

                # Skip if already exists in target
                if os.path.exists(target_path):
                    skipped.append(csv_file)
                    continue

                # Move file
                import shutil
                shutil.move(source_path, target_path)
                migrated.append(csv_file)

            except Exception as file_error:
                errors.append(f"{csv_file}: {str(file_error)}")

        # Build status message
        status_items = []

        if migrated:
            status_items.append(
                html.P(f"✅ Successfully migrated {len(migrated)} files",
                       style={'color': '#28a745', 'font-weight': 'bold', 'margin': '0'})
            )
            if len(migrated) <= 10:
                status_items.append(
                    html.Ul([html.Li(f, style={'font-size': '13px'}) for f in migrated],
                            style={'margin': '5px 0', 'color': '#666'})
                )

        if skipped:
            status_items.append(
                html.P(f"ℹ️ Skipped {len(skipped)} files (already exist in target)",
                       style={'color': '#17a2b8', 'font-weight': 'bold', 'margin': '10px 0 0 0'})
            )

        if errors:
            status_items.append(
                html.P(f"❌ {len(errors)} errors occurred",
                       style={'color': '#dc3545', 'font-weight': 'bold', 'margin': '10px 0 0 0'})
            )
            status_items.append(
                html.Ul([html.Li(e, style={'font-size': '13px'}) for e in errors],
                        style={'margin': '5px 0', 'color': '#dc3545'})
            )

        if not status_items:
            return html.P("⚠️ No action taken", style={'color': '#ffc107', 'font-weight': 'bold'})

        return html.Div(status_items)

    except Exception as e:
        return html.P(f"❌ Migration error: {str(e)}",
                      style={'color': '#dc3545', 'font-weight': 'bold'})
