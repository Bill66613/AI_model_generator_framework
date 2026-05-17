import os
import json
from pathlib import Path
import pandas as pd
import base64
import io
import plotly.express as px
import plotly.graph_objects as go
from dash import dcc, html, Input, Output, State, dash_table, ctx, no_update

from config.config import PERSISTENT_DIR, METADATA_FILE, SENSOR_COLUMNS, get_window_pattern, resolve_working_dir, ROOT_DIR


def parse_contents(contents, filename):
    """Parse uploaded CSV file contents into a DataFrame."""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df


def register_callbacks(app):
    """Register all data management callbacks with the app."""

    @app.callback(
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
        base_dir = resolve_working_dir(base_dir)

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

    @app.callback(
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
            base_dir = resolve_working_dir(base_dir)
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

    @app.callback(
        Output('dataset-selector', 'options'),
        Input('tabs', 'value'),
        Input('working-directory-store', 'data')
    )
    def data_selector_options(tab, base_dir):
        """Update the dataset selector options."""
        try:
            base_dir = resolve_working_dir(base_dir)
            metadata_file = os.path.join(base_dir, 'metadata.json')

            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return [{'label': filename, 'value': filename} for filename in metadata.keys()]
            return []
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading metadata: {e}")
            return []

    @app.callback(
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
            base_dir = resolve_working_dir(base_dir)
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

    @app.callback(
        Output('upload-output', 'children', True),
        Input('clear-data-btn', 'n_clicks'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def clear_data(n_clicks, base_dir):
        """Clear all stored data and metadata."""
        base_dir = resolve_working_dir(base_dir)

        try:
            # Remove all files in the persistent directory
            for file in os.listdir(base_dir):
                file_path = os.path.join(base_dir, file)
                if os.path.isfile(file_path):
                    try:
                        os.remove(file_path)
                    except OSError as e:
                        # Log but continue — best effort removal
                        print(f"Warning: could not remove {file_path}: {e}")

            # Reset metadata
            metadata_file = os.path.join(base_dir, 'metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump({}, f)

            return "✅ All data and metadata have been cleared."
        except Exception as e:
            return f"❌ Error clearing data: {str(e)}"

    @app.callback(
        Output('data-preview', 'figure', True),
        Input('dataset-selector', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def filter_and_display_data(dataset_name, base_dir):
        """Filters the dataset and displays it as a chart."""
        if dataset_name:
            try:
                base_dir = resolve_working_dir(base_dir)
                metadata_file = os.path.join(base_dir, 'metadata.json')

                # Load existing metadata
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                sampling_rate = metadata.get(
                    dataset_name, {}).get("sampling_rate", 100)

                file_path = metadata.get(dataset_name, {}).get("path")
                if not file_path:
                    file_path = os.path.join(base_dir, 'datasets', dataset_name)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path)

                    # Add a time axis based on the sampling rate
                    df['Time_seconds'] = pd.Series(
                        range(df.shape[0])) / sampling_rate

                    # Get numeric sensor columns (exclude Time_seconds and non-numeric like labels)
                    sensor_cols = [
                        col for col in df.select_dtypes(include='number').columns if col != 'Time_seconds']

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
            except (json.JSONDecodeError, IOError, KeyError) as e:
                print(f"Error loading dataset '{dataset_name}': {e}")
        return {}

    @app.callback(
        Output('assigned-label', 'value'),
        Input('dataset-selector', 'value'),
        State('working-directory-store', 'data')
    )
    def display_label(dataset_name, base_dir):
        """Displays the label assigned to a dataset."""
        if dataset_name:
            base_dir = resolve_working_dir(base_dir)
            metadata_file = os.path.join(base_dir, 'metadata.json')

            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                return metadata.get(dataset_name, {}).get("label", "")
        return ""

    # NOTE: display_dataset was removed — filter_and_display_data (above)
    # already targets Output('data-preview', 'figure') with allow_duplicate=True
    # on the same Input('dataset-selector', 'value').  Having two callbacks for
    # the same output/input pair caused Dash duplicate-callback warnings.

    @app.callback(
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
            base_dir = resolve_working_dir(base_dir)
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

    @app.callback(
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
            base_dir = resolve_working_dir(base_dir)
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

    @app.callback(
        Output('upload-output', 'children', allow_duplicate=True),
        Input('export-metadata-btn', 'n_clicks'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def export_metadata(n_clicks, base_dir):
        """Export metadata to a downloadable JSON file."""
        try:
            base_dir = resolve_working_dir(base_dir)
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

    @app.callback(
        Output('current-working-dir-display', 'children'),
        Output('working-directory-store', 'data'),
        Output('directory-path-input', 'value'),
        Input('working-directory-store', 'data'),
        prevent_initial_call=False
    )
    def display_current_working_dir(stored_dir):
        """Display the current persistent directory on page load and populate input."""
        abs_dir = resolve_working_dir(stored_dir)
        # Compute a user-friendly display: show relative path when inside ROOT_DIR
        try:
            rel = Path(abs_dir).relative_to(ROOT_DIR)
            display = f"{abs_dir}  (relative: {rel})"
        except ValueError:
            display = abs_dir
        store_value = stored_dir if stored_dir else 'persistent_data'
        return display, store_value, stored_dir or ''

    @app.callback(
        Output('directory-path-input', 'value', allow_duplicate=True),
        Input('browse-working-dir-btn', 'n_clicks'),
        State('directory-path-input', 'value'),
        prevent_initial_call=True,
    )
    def browse_working_directory(n_clicks, current_value):
        """Open a native OS folder-picker dialog and populate the path input.

        Uses tkinter (stdlib) which works because this Dash app runs locally.
        If tkinter is unavailable the button simply does nothing.
        """
        if not n_clicks:
            return no_update
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk()
            root.withdraw()         # hide the tiny tk root window
            root.attributes('-topmost', True)  # bring dialog to front
            initial_dir = resolve_working_dir(current_value) if current_value else str(ROOT_DIR)
            chosen = filedialog.askdirectory(
                title="Select Working Directory",
                initialdir=initial_dir,
                mustexist=False,
            )
            root.destroy()
            if not chosen:
                return no_update
            # Try to return a relative path when inside ROOT_DIR (portable)
            try:
                rel = Path(chosen).relative_to(ROOT_DIR)
                return str(rel)
            except ValueError:
                return str(chosen)
        except Exception:
            return no_update

    @app.callback(
        Output('current-working-dir-display',
               'children', allow_duplicate=True),
        Output('working-directory-store', 'data', allow_duplicate=True),
        Output('directory-path-input', 'value', allow_duplicate=True),
        Output('working-dir-status', 'children'),
        Input('select-working-dir-btn', 'n_clicks'),
        Input('use-default-dir-btn', 'n_clicks'),
        State('directory-path-input', 'value'),
        prevent_initial_call=True
    )
    def set_working_directory(apply_clicks, default_clicks, input_path):
        """Set persistent base directory and create standard subdirectories.

        Accepts both **absolute** paths (``D:\\data``) and **relative** paths
        (``persistent_data``, ``../my_data``).  Relative paths are resolved
        against ROOT_DIR at runtime, so the project remains portable when the
        whole application folder is moved to another machine or drive.
        """
        if not ctx.triggered:
            return no_update, no_update, no_update, no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        try:
            if button_id == 'use-default-dir-btn':
                # Store as relative 'persistent_data' so the default is portable
                stored_value = 'persistent_data'

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
                raw = input_path.strip()
                p = Path(os.path.normpath(raw))
                # If the user entered a relative path, store it as-is (portable).
                # If absolute, store as-is.
                stored_value = raw if not p.is_absolute() else str(p)
            else:
                return no_update, no_update, no_update, no_update

            # Resolve to absolute for directory creation
            base_dir = resolve_working_dir(stored_value)

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
            is_relative = not Path(os.path.normpath(stored_value)).is_absolute()
            rel_note = f" (relative to app root: {stored_value})" if is_relative else ""
            status_message = html.Div([
                html.P("✅ Persistent directory configured successfully",
                       style={'color': '#28a745', 'font-weight': 'bold', 'margin': '0'}),
                html.P(f"📁 Base: {base_dir}{rel_note}",
                       style={'color': '#666', 'font-size': '14px', 'margin': '5px 0'}),
                html.P(f"📂 Created: {', '.join(subdirs)}, metadata.json",
                       style={'color': '#666', 'font-size': '13px', 'margin': '5px 0 0 0'})
            ])

            return (
                f"{base_dir}{rel_note}",
                stored_value,
                stored_value,
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

    @app.callback(
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
            # Resolve relative paths to absolute
            source_dir = resolve_working_dir(base_dir)
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
