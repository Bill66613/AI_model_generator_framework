"""
Feature Engineering Callbacks
Handles unified feature engineering across multiple activity labels
"""

from dash import  Input, Output, State, no_update, html, ctx
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dash import dash_table
from sklearn.model_selection import train_test_split
import glob

from config.config import (
    PERSISTENT_DIR, METADATA_FILE, WINDOWS_DIR
)
from utils.model_training import extract_time_domain_features, extract_frequency_domain_features, create_feature_vector


# Placeholder callbacks - to be implemented


def register_callbacks(app):
    """Register all callbacks with the app."""
    @app.callback(
        Output('activity-labels-selector', 'options'),
        Input('tabs', 'value'),
        State('working-directory-store', 'data')
    )
    def populate_activity_labels(tab, base_dir):
        """
        Populate the activity labels dropdown with all available datasets
        that have split windows ready for feature engineering.
        Uses the working directory from the store.
        """
        # Use stored base directory or default to PERSISTENT_DIR
        if not base_dir:
            base_dir = PERSISTENT_DIR

        metadata_file = os.path.join(base_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            return []

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Find all unique activity labels that have dragged_samples (windows)
            labels_with_windows = set()
            for dataset_name, dataset_info in metadata.items():
                if 'dragged_samples' in dataset_info and dataset_info['dragged_samples']:
                    # Get the label from metadata
                    label = dataset_info.get(
                        'label', dataset_name.replace('.csv', ''))
                    labels_with_windows.add(label)

            # Create dropdown options sorted alphabetically
            options = [
                {'label': f'📌 {label.title()}', 'value': label}
                for label in sorted(labels_with_windows)
            ]

            return options

        except Exception as e:
            print(f"Error loading activity labels: {e}")
            return []


    @app.callback(
        Output('windows-per-label-display', 'children'),
        Input('activity-labels-selector', 'value'),
        State('working-directory-store', 'data')
    )
    def update_windows_per_label(selected_labels, base_dir):
        """
        Display the number of windows available for each selected label.
        Uses the working directory from the store.
        """
        if not selected_labels:
            return html.Div("No labels selected", style={'color': '#999', 'font-style': 'italic'})

        # Use stored base directory or default to PERSISTENT_DIR
        if not base_dir:
            base_dir = PERSISTENT_DIR

        metadata_file = os.path.join(base_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            return html.Div("No metadata found", style={'color': '#ff0000'})

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Count windows per label
            label_counts = {}
            for dataset_name, dataset_info in metadata.items():
                label = dataset_info.get('label', dataset_name.replace('.csv', ''))
                if label in selected_labels and 'dragged_samples' in dataset_info:
                    # Count existing files
                    window_count = len(
                        [f for f in dataset_info['dragged_samples'] if os.path.exists(f)])
                    if label in label_counts:
                        label_counts[label] += window_count
                    else:
                        label_counts[label] = window_count

            # Create display
            total_windows = sum(label_counts.values())

            items = []
            for label in sorted(label_counts.keys()):
                count = label_counts[label]
                items.append(
                    html.Div([
                        html.Span(f"{label.title()}: ", style={
                                  'font-weight': 'bold'}),
                        html.Span(f"{count} windows", style={'color': '#28a745'})
                    ], style={'margin-bottom': '8px'})
                )

            items.append(
                html.Div([
                    html.Span("Total: ", style={
                              'font-weight': 'bold', 'color': '#2E86AB'}),
                    html.Span(f"{total_windows} windows", style={
                              'color': '#2E86AB', 'font-weight': 'bold'})
                ], style={'margin-top': '10px', 'padding-top': '10px', 'border-top': '2px solid #dee2e6'})
            )

            return html.Div(items)

        except Exception as e:
            return html.Div(f"Error: {str(e)}", style={'color': '#ff0000'})


    @app.callback(
        [Output('activity-labels-selector', 'value', allow_duplicate=True)],
        [Input('select-all-labels-btn', 'n_clicks'),
         Input('clear-labels-btn', 'n_clicks')],
        [State('activity-labels-selector', 'options')],
        prevent_initial_call=True
    )
    def handle_label_selection_buttons(select_clicks, clear_clicks, options):
        """
        Handle Select All and Clear All buttons for label selection.
        """
        if not ctx.triggered:
            return [no_update]

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if button_id == 'select-all-labels-btn':
            # Select all available labels
            return [[opt['value'] for opt in options]]
        elif button_id == 'clear-labels-btn':
            # Clear selection
            return [[]]

        return [no_update]


    @app.callback(
        Output('feature-count-display', 'children'),
        Input('global-feature-selection', 'value')
    )
    def update_feature_count(feature_selection):
        """
        Display the total number of features based on the selection method.
        """
        feature_counts = {
            'orientation_invariant_time_only': (
                '33 features',
                'Orientation-robust magnitudes: 15 stats × (acc_mag + gyro_mag) + 3 jerk stats. '
                'RECOMMENDED for deployment (C / MicroPython).'
            ),
            'orientation_invariant': (
                '47 features',
                'Orientation-robust magnitudes (33 time) + DFT on magnitudes (14 freq). '
                'Fully deployable — DFT runs on-device.'
            ),
            'time_domain': (
                '90 features',
                'Per-axis time-domain: 15 stats × 6 axes (aX, aY, aZ, gX, gY, gZ). Deployable.'
            ),
            'all': (
                '138 features',
                'Per-axis time-domain (90) + frequency-domain (48). '
                '⚠️ Per-axis FFT NOT yet deployable (use orientation_invariant for freq features).'
            ),
            'frequency_domain': (
                '48 features',
                'Per-axis frequency-domain only (FFT features). '
                '⚠️ Per-axis FFT NOT yet deployable.'
            ),
            'raw': (
                '6 features',
                'Raw sensor axes mean per window (aX, aY, aZ, gX, gY, gZ).'
            ),
        }

        if feature_selection in feature_counts:
            count, description = feature_counts[feature_selection]
            return html.Div([
                html.Div(count, style={'font-size': '20px',
                         'font-weight': 'bold', 'color': '#28a745'}),
                html.Div(description, style={
                         'font-size': '13px', 'color': '#666', 'margin-top': '5px'})
            ])

        return "Select a feature extraction method"


    @app.callback(
        Output('global-test-split-display', 'children'),
        [Input('global-train-split', 'value'),
         Input('global-val-split', 'value')]
    )
    def calculate_test_split(train_ratio, val_ratio):
        """
        Calculate and display the test split percentage.
        """
        if train_ratio is None or val_ratio is None:
            return "--"

        test_ratio = 1.0 - train_ratio - val_ratio

        # Validate
        if test_ratio < 0:
            return html.Span("Error: Invalid split!", style={'color': '#dc3545'})

        return f"{test_ratio*100:.0f}%"


    @app.callback(
        [Output('feature-engineering-results', 'children'),
         Output('engineered-dataset-stats', 'children'),
         Output('engineered-dataset-store', 'data')],
        Input('execute-feature-engineering-btn', 'n_clicks'),
        [State('activity-labels-selector', 'value'),
         State('global-feature-selection', 'value'),
         State('global-normalization-method', 'value'),
         State('global-window-size', 'value'),
         State('global-sampling-rate', 'value'),
         State('global-train-split', 'value'),
         State('global-val-split', 'value'),
         State('global-random-state', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def execute_feature_engineering(n_clicks, selected_labels, feature_method,
                                    normalization_method, target_window_size, sampling_rate,
                                    train_ratio, val_ratio, random_state, base_dir):
        """
        Main feature engineering executor.
        Applies consistent settings across all selected activity labels.
        Uses zero-padding for windows smaller than target size.

        Workflow:
        1. Load all windows from all selected labels
        2. Zero-pad windows to target size if needed
        3. Extract features uniformly (same method for all)
        4. Apply normalization uniformly (same scaler for all)
        5. Combine into single dataset with 'activity' column
        6. Perform train/val/test split on combined dataset
        7. Save to engineered-dataset-store
        8. Display statistics
        """
        if not selected_labels:
            return html.Div("⚠️ Please select at least one activity label",
                            style={'color': '#ff9800', 'padding': '15px'}), "", {}

        try:
            # Use stored base directory or default to PERSISTENT_DIR
            if not base_dir:
                base_dir = PERSISTENT_DIR

            metadata_file = os.path.join(base_dir, 'metadata.json')

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Step 1: Load all windows from all selected labels
            all_windows = []
            all_labels = []
            label_window_counts = {}

            for dataset_name, dataset_info in metadata.items():
                label = dataset_info.get('label', dataset_name.replace('.csv', ''))
                if label in selected_labels and 'dragged_samples' in dataset_info:
                    window_files = [
                        f for f in dataset_info['dragged_samples'] if os.path.exists(f)]

                    for window_file in window_files:
                        df_window = pd.read_csv(window_file)
                        all_windows.append(df_window)
                        all_labels.append(label)

                    label_window_counts[label] = label_window_counts.get(
                        label, 0) + len(window_files)

            if not all_windows:
                return html.Div("⚠️ No windows found for selected labels",
                                style={'color': '#dc3545', 'padding': '15px'}), "", {}

            # Step 2: Convert target window size from ms to samples
            target_window_samples = int(
                (target_window_size / 1000) * sampling_rate)

            # Step 3: Zero-pad windows to target size if needed
            sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
            padded_windows = []
            padding_stats = {'padded': 0, 'original_size': 0}

            for df_window in all_windows:
                current_size = len(df_window)
                if current_size < target_window_samples:
                    # Zero-pad
                    padding_needed = target_window_samples - current_size
                    padding_df = pd.DataFrame(0, index=range(
                        padding_needed), columns=df_window.columns)
                    df_padded = pd.concat(
                        [df_window, padding_df], ignore_index=True)
                    padded_windows.append(df_padded)
                    padding_stats['padded'] += 1
                elif current_size > target_window_samples:
                    # Truncate
                    df_truncated = df_window.iloc[:target_window_samples]
                    padded_windows.append(df_truncated)
                else:
                    # Exact size
                    padded_windows.append(df_window)

                padding_stats['original_size'] = current_size

            # Step 3: Extract features uniformly (with zero-padding applied)

            feature_list = []
            for df_window, label in zip(padded_windows, all_labels):
                # Map feature_method to create_feature_vector parameters
                if feature_method == 'orientation_invariant_time_only':
                    # Orientation-robust magnitude features - TIME DOMAIN ONLY
                    # 33 features (deployable to C++ devices)
                    # RECOMMENDED for deployment
                    feature_df = create_feature_vector(
                        df_window, sensor_cols, sampling_rate,
                        include_frequency=False,
                        orientation_robust=True,
                        include_per_axis=False
                    )
                elif feature_method == 'orientation_invariant':
                    # Orientation-robust magnitude features + FFT
                    # 47 features (33 time + 14 freq)
                    # WARNING: FFT features NOT supported in C++ deployment
                    feature_df = create_feature_vector(
                        df_window, sensor_cols, sampling_rate,
                        include_frequency=True,
                        orientation_robust=True,
                        include_per_axis=False
                    )
                elif feature_method == 'all':
                    # All features: per-axis time + freq
                    # 138 features (90 time + 48 freq)
                    # WARNING: FFT features NOT supported in C++ deployment
                    feature_df = create_feature_vector(
                        df_window, sensor_cols, sampling_rate,
                        include_frequency=True,
                        orientation_robust=False,
                        include_per_axis=True
                    )
                elif feature_method == 'time_domain':
                    # Per-axis time features only (DEPLOYABLE)
                    # 90 features
                    feature_df = create_feature_vector(
                        df_window, sensor_cols, sampling_rate,
                        include_frequency=False,
                        orientation_robust=False,
                        include_per_axis=True
                    )
                elif feature_method == 'frequency_domain':
                    # Per-axis frequency features only
                    # 48 features
                    # WARNING: NOT supported in C++ deployment
                    feature_df = create_feature_vector(
                        df_window, sensor_cols, sampling_rate,
                        include_frequency=True,
                        orientation_robust=False,
                        include_per_axis=True
                    )
                elif feature_method == 'raw':
                    # Raw sensor values (mean of window) - 6 features
                    features = {
                        'aX': df_window['aX'].mean(),
                        'aY': df_window['aY'].mean(),
                        'aZ': df_window['aZ'].mean(),
                        'gX': df_window['gX'].mean(),
                        'gY': df_window['gY'].mean(),
                        'gZ': df_window['gZ'].mean()
                    }
                    feature_df = pd.DataFrame([features])
                else:
                    # Default fallback: per-axis time features
                    feature_df = create_feature_vector(
                        df_window, sensor_cols, sampling_rate,
                        include_frequency=False,
                        orientation_robust=False,
                        include_per_axis=True
                    )
                
                # Convert to dict and add label
                features = feature_df.iloc[0].to_dict()
                features['activity'] = label
                feature_list.append(features)

            # Create DataFrame
            df_features = pd.DataFrame(feature_list)

            # Step 3: Store raw features (NO scaling here - scaling is done during
            # model training in EdgeMLModel.preprocess_data() to ensure the scaler
            # is saved with the model and used correctly during deployment)
            X = df_features.drop('activity', axis=1).values
            y = df_features['activity'].values
            feature_names = df_features.drop('activity', axis=1).columns.tolist()
            # Note: normalization_method is stored in metadata for reference only

            # Also save raw windowed sensor data for CNN training
            # Shape: (n_windows, window_size_samples, 6)
            raw_windows = np.array([
                w[sensor_cols].values[:target_window_samples]
                for w in padded_windows
            ], dtype=np.float32)
            raw_labels = np.array(all_labels)

            # Step 4: Split AFTER combining (prevents data leakage)
            test_ratio = 1.0 - train_ratio - val_ratio

            # Use indices to keep raw windows in sync with feature splits
            indices = np.arange(len(X))

            # First split: train+val vs test
            idx_trainval, idx_test, y_trainval, y_test = train_test_split(
                indices, y, test_size=test_ratio, random_state=random_state, stratify=y
            )
            X_trainval, X_test = X[idx_trainval], X[idx_test]
            rw_trainval, rw_test = raw_windows[idx_trainval], raw_windows[idx_test]

            # Second split: train vs val
            if val_ratio > 0:
                val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
                sub_idx_train, sub_idx_val, y_train, y_val = train_test_split(
                    np.arange(len(X_trainval)), y_trainval,
                    test_size=val_ratio_adjusted,
                    random_state=random_state, stratify=y_trainval
                )
                X_train, X_val = X_trainval[sub_idx_train], X_trainval[sub_idx_val]
                rw_train, rw_val = rw_trainval[sub_idx_train], rw_trainval[sub_idx_val]
            else:
                X_train, y_train = X_trainval, y_trainval
                rw_train = rw_trainval
                X_val, y_val = np.array([]), np.array([])
                rw_val = np.array([])

            # Step 5: Save results to files
            # Create training directory if it doesn't exist
            training_dir = os.path.join(base_dir, 'training')
            os.makedirs(training_dir, exist_ok=True)

            # Create DataFrames with feature names
            df_train = pd.DataFrame(X_train, columns=feature_names)
            df_train['label'] = y_train

            df_test = pd.DataFrame(X_test, columns=feature_names)
            df_test['label'] = y_test

            if len(X_val) > 0:
                df_val = pd.DataFrame(X_val, columns=feature_names)
                df_val['label'] = y_val

            # Create a combined dataset name from selected labels
            dataset_name = '_'.join(sorted(selected_labels)[
                                    :3])  # Use up to 3 labels
            if len(selected_labels) > 3:
                dataset_name += f"_and_{len(selected_labels)-3}_more"

            # Save to CSV files
            train_file = os.path.join(training_dir, f"{dataset_name}_train.csv")
            test_file = os.path.join(training_dir, f"{dataset_name}_test.csv")

            df_train.to_csv(train_file, index=False)
            df_test.to_csv(test_file, index=False)

            if len(X_val) > 0:
                val_file = os.path.join(training_dir, f"{dataset_name}_val.csv")
                df_val.to_csv(val_file, index=False)

            # Save raw windowed sensor data for CNN training (numpy arrays)
            np.save(os.path.join(training_dir, f"{dataset_name}_raw_train.npy"), rw_train)
            np.save(os.path.join(training_dir, f"{dataset_name}_raw_test.npy"), rw_test)
            np.save(os.path.join(training_dir, f"{dataset_name}_raw_train_labels.npy"), y_train)
            np.save(os.path.join(training_dir, f"{dataset_name}_raw_test_labels.npy"), y_test)
            if len(rw_val) > 0:
                np.save(os.path.join(training_dir, f"{dataset_name}_raw_val.npy"), rw_val)
                np.save(os.path.join(training_dir, f"{dataset_name}_raw_val_labels.npy"), y_val)

            # Save feature-engineering metadata so that the training &
            # code-generation stages can recover window_size, sampling_rate,
            # feature_method, etc.  One JSON per dataset in the training dir.
            fe_metadata = {
                'window_size_ms': target_window_size,
                'sampling_rate': sampling_rate,
                'window_size_samples': target_window_samples,
                'feature_method': feature_method,
                'normalization_method': normalization_method,
                'feature_names': feature_names,
                'num_features': len(feature_names),
                'selected_labels': selected_labels,
                'orientation_robust': feature_method in (
                    'orientation_invariant', 'orientation_invariant_time_only'),
                'include_per_axis': feature_method in (
                    'all', 'time_domain', 'frequency_domain'),
                'include_frequency': feature_method in (
                    'all', 'orientation_invariant', 'frequency_domain'),
                'train_split': train_ratio,
                'val_split': val_ratio,
                'test_split': test_ratio,
                'random_state': random_state,
            }
            fe_meta_file = os.path.join(
                training_dir, f"{dataset_name}_fe_metadata.json")
            with open(fe_meta_file, 'w') as f:
                json.dump(fe_metadata, f, indent=2)

            # Also save to engineered-dataset-store for backwards compatibility
            engineered_data = {
                'train': {'X': X_train.tolist(), 'y': y_train.tolist()},
                'val': {'X': X_val.tolist(), 'y': y_val.tolist()},
                'test': {'X': X_test.tolist(), 'y': y_test.tolist()},
                'feature_names': feature_names,
                'scaler': normalization_method,
                'labels': selected_labels,
                'feature_method': feature_method
            }

            # Step 6: Display statistics
            stats_data = []
            for label in sorted(set(y)):
                train_count = sum(y_train == label)
                val_count = sum(y_val == label) if len(y_val) > 0 else 0
                test_count = sum(y_test == label)
                total_count = train_count + val_count + test_count

                stats_data.append({
                    'Activity': label.title(),
                    'Total Windows': total_count,
                    'Train': f"{train_count} ({train_count/total_count*100:.1f}%)",
                    'Validation': f"{val_count} ({val_count/total_count*100:.1f}%)" if val_count > 0 else '0',
                    'Test': f"{test_count} ({test_count/total_count*100:.1f}%)"
                })

            # Add summary row
            total_train = len(y_train)
            total_val = len(y_val)
            total_test = len(y_test)
            total_all = total_train + total_val + total_test

            stats_data.append({
                'Activity': '📊 TOTAL',
                'Total Windows': total_all,
                'Train': f"{total_train} ({total_train/total_all*100:.1f}%)",
                'Validation': f"{total_val} ({total_val/total_all*100:.1f}%)" if total_val > 0 else '0',
                'Test': f"{total_test} ({total_test/total_all*100:.1f}%)"
            })

            stats_table = dash_table.DataTable(
                data=stats_data,
                columns=[{'name': col, 'id': col} for col in stats_data[0].keys()],
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#2E86AB',
                              'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': len(stats_data) - 1},
                     'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'}
                ]
            )

            # Success message
            success_msg = html.Div([
                html.H4("✅ Feature Engineering Complete!",
                        style={'color': '#28a745'}),
                html.P(
                    f"Successfully processed {total_all} windows from {len(selected_labels)} activity labels"),
                html.Ul([
                    html.Li(
                        f"Target window size: {target_window_size}ms ({target_window_samples} samples @ {sampling_rate}Hz)"),
                    html.Li(
                        f"Zero-padded windows: {padding_stats['padded']} / {total_all}"),
                    html.Li(
                        f"Features extracted: {len(feature_names)} features using '{feature_method}' method"),
                    html.Li("Rotation augmentation: ON (only for orientation_invariant)" if feature_method ==
                            'orientation_invariant' else "Rotation augmentation: OFF"),
                    html.Li(
                        f"Normalization: {normalization_method.title() if normalization_method != 'none' else 'None'}"),
                    html.Li(
                        f"Train/Val/Test split: {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%"),
                    html.Li(f"Random state: {random_state}"),
                    html.Li([html.Strong("💾 Saved to: "), f"{training_dir}/"])
                ])
            ], style={'padding': '15px', 'background-color': '#d4edda', 'border-radius': '5px'})

            return success_msg, stats_table, engineered_data

        except Exception as e:
            import traceback
            error_msg = html.Div([
                html.H4("❌ Error during feature engineering",
                        style={'color': '#dc3545'}),
                html.P(str(e)),
                html.Pre(traceback.format_exc(), style={
                         'fontSize': '12px', 'background': '#f8f9fa', 'padding': '10px'})
            ], style={'padding': '15px', 'background-color': '#f8d7da', 'border-radius': '5px'})
            return error_msg, "", {}

