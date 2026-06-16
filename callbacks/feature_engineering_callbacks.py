"""
Feature Engineering Callbacks
Handles unified feature engineering across multiple activity labels
"""

from dash import Input, Output, State, no_update, html, ctx
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dash import dash_table
from sklearn.model_selection import train_test_split
import glob

from config.config import (
    PERSISTENT_DIR, METADATA_FILE, WINDOWS_DIR,
    SENSOR_COLUMNS, DEFAULT_SAMPLING_RATE, resolve_metadata_path, resolve_working_dir
)
from utils.model_training import extract_time_domain_features, extract_frequency_domain_features, create_feature_vector
from utils.data_augmentation import augment_windows, AUGMENTATION_METHODS


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
        base_dir = resolve_working_dir(base_dir)

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
        base_dir = resolve_working_dir(base_dir)

        metadata_file = os.path.join(base_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            return html.Div("No metadata found", style={'color': '#ff0000'})

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Count windows per label
            label_counts = {}
            for dataset_name, dataset_info in metadata.items():
                label = dataset_info.get(
                    'label', dataset_name.replace('.csv', ''))
                if label in selected_labels and 'dragged_samples' in dataset_info:
                    # Count existing files
                    window_count = len([
                        f for f in (
                            resolve_metadata_path(path, base_dir)
                            for path in dataset_info['dragged_samples']
                        )
                        if os.path.exists(f)
                    ])
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
                        html.Span(f"{count} windows", style={
                                  'color': '#28a745'})
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
        Display the total number of features based on the selection method
        and the actual sensor columns configured.
        """
        from config.config import ACCEL_COLUMNS, GYRO_COLUMNS
        n_accel = len(ACCEL_COLUMNS)
        n_gyro = len(GYRO_COLUMNS)
        n_axes = n_accel + n_gyro
        n_mag_groups = (1 if n_accel > 0 else 0) + (1 if n_gyro > 0 else 0)
        jerk_feats = 3 if n_accel >= 2 else 0
        gyro_jerk_feats = 3 if n_gyro >= 2 else 0
        # Scalar extras: acc_sma (1) + tilt_pitch + tilt_roll (2) + autocorr_lag1 (1) + peak_count (1) = 5
        scalar_extras = (1 if n_accel >= 1 else 0) + (2 if n_accel >= 3 else 0) + (1 if n_accel > 1 else 0) + (1 if n_accel >= 2 else 0)
        # DFT features per magnitude group (10 spectral + spectral_entropy = 11)
        freq_per_mag = 11
        # Total time-domain features for orientation_invariant modes
        oi_time = 15 * n_mag_groups + jerk_feats + gyro_jerk_feats + scalar_extras

        feature_counts = {
            'orientation_invariant_time_only': (
                f'{oi_time} features',
                f'Orientation-robust magnitudes: 15 stats × {n_mag_groups} mag'
                f' + {jerk_feats} acc-jerk + {gyro_jerk_feats} gyro-jerk + {scalar_extras} scalar extras'
                f' (SMA, tilt, autocorr, peak count). '
                'RECOMMENDED for deployment — fully deployable to all targets (C / C++ / MicroPython).'
            ),
            'orientation_invariant': (
                f'{oi_time + freq_per_mag * n_mag_groups} features',
                f'Orientation-robust time ({oi_time}) '
                f'+ DFT on magnitudes ({freq_per_mag * n_mag_groups} freq, incl. spectral entropy). '
                'Fully deployable — on-device DFT uses only sin/cos, no FFT library needed.'
            ),
            'time_domain': (
                f'{15 * n_axes} features',
                f'Per-axis time-domain: 15 stats x {n_axes} axes. '
                'Fully deployable to all targets.'
            ),
            'all': (
                f'{15 * n_axes + 11 * n_axes} features',
                f'Per-axis time-domain ({15 * n_axes}) + per-axis frequency-domain ({11 * n_axes}). '
                'Per-axis freq features are NOT deployable — only orientation-robust '
                'DFT is implemented in code generators.'
            ),
            'frequency_domain': (
                f'{11 * n_axes} features',
                f'Per-axis frequency-domain only ({11 * n_axes} features). '
                'Per-axis freq features are NOT deployable. '
                'Use "Orientation Invariant" which includes deployable DFT on magnitudes.'
            ),
            'raw': (
                f'{n_axes} features',
                f'Raw sensor axes mean per window ({n_axes} axes).'
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
        [Output('global-test-split-display', 'children'),
         Output('execute-feature-engineering-btn', 'disabled')],
        [Input('global-train-split', 'value'),
         Input('global-val-split', 'value'),
         Input('split-mode-selector', 'value')]
    )
    def calculate_test_split(train_ratio, val_ratio, split_mode):
        """
        Calculate and display the test split percentage.
        Disables the execute button when the split is invalid (auto mode only).
        """
        # In manual mode the ratios are irrelevant — never disable the button
        if split_mode == 'manual':
            return "—", False

        if train_ratio is None or val_ratio is None:
            return "--", True

        test_ratio = 1.0 - train_ratio - val_ratio

        # Validate
        if test_ratio < 0.01:
            return (
                html.Span(
                    f"⚠️ Invalid split! Train ({train_ratio*100:.0f}%) + "
                    f"Val ({val_ratio*100:.0f}%) ≥ 100%",
                    style={'color': '#dc3545', 'font-size': '16px'}
                ),
                True  # disable button
            )

        return f"{test_ratio*100:.0f}%", False

    # --- Data Augmentation UI callbacks ---

    @app.callback(
        Output('augmentation-options-container', 'style'),
        Input('augmentation-enable', 'value')
    )
    def toggle_augmentation_options(enabled):
        """Show/hide augmentation options based on the enable checkbox."""
        if enabled and 'enabled' in enabled:
            return {'display': 'block', 'margin-top': '10px'}
        return {'display': 'none'}

    @app.callback(
        Output('augmentation-factor-display', 'children'),
        Input('augmentation-factor', 'value')
    )
    def update_augmentation_factor_display(factor):
        """Display the current augmentation factor with explanation."""
        if factor is None:
            factor = 2
        return html.Span(
            f"Each original window will generate {factor} augmented "
            f"version{'s' if factor > 1 else ''} → "
            f"dataset grows by ~{factor}×",
            style={'color': '#495057', 'fontSize': '13px'}
        )

    @app.callback(
        Output('auto-split-panel', 'style'),
        Output('manual-split-panel', 'style'),
        Input('split-mode-selector', 'value'),
    )
    def toggle_split_panels(mode):
        """Show/hide auto vs manual split panels."""
        if mode == 'manual':
            return {'display': 'none'}, {'display': 'block'}
        return {'display': 'block'}, {'display': 'none'}

    @app.callback(
        Output('manual-split-table-container', 'children'),
        Output('manual-split-store', 'data'),
        Output('manual-split-summary', 'children'),
        Input('load-manual-split-btn', 'n_clicks'),
        Input('auto-assign-remaining-btn', 'n_clicks'),
        State('activity-labels-selector', 'value'),
        State('working-directory-store', 'data'),
        State('manual-split-store', 'data'),
        prevent_initial_call=True
    )
    def load_manual_split_windows(load_clicks, auto_clicks, selected_labels, base_dir, existing_assignments):
        """
        Populate the manual-split table with all windows from selected labels.
        Each row shows: window filename, label, and a split-assignment dropdown.
        Existing assignments are preserved on refresh.
        """
        if not selected_labels:
            return (
                html.P("⚠️ Select at least one activity label first.",
                       style={'color': '#ff9800', 'font-style': 'italic'}),
                existing_assignments or {},
                ""
            )

        base_dir = resolve_working_dir(base_dir)
        metadata_file = os.path.join(base_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            return (
                html.P("No metadata found.", style={'color': '#dc3545'}),
                existing_assignments or {},
                ""
            )

        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            return html.P(f"Error reading metadata: {e}", style={'color': '#dc3545'}), existing_assignments or {}, ""

        assignments = dict(existing_assignments or {})

        # Collect all window files across selected labels
        rows = []
        for dataset_name, dataset_info in metadata.items():
            label = dataset_info.get('label', dataset_name.replace('.csv', ''))
            if label not in selected_labels:
                continue
            for raw_wf in dataset_info.get('dragged_samples', []):
                wf = resolve_metadata_path(raw_wf, base_dir)
                if os.path.exists(wf):
                    rows.append({'path': wf, 'label': label,
                                 'filename': os.path.basename(wf)})

        if not rows:
            return (
                html.P("No windows found for selected labels.",
                       style={'color': '#ff9800', 'font-style': 'italic'}),
                assignments,
                ""
            )

        # If "Auto-assign remaining" was clicked, distribute unassigned windows
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else ''
        if triggered_id == 'auto-assign-remaining-btn':
            unassigned = [r for r in rows if assignments.get(r['path'], 'unassigned') == 'unassigned']
            # Stratified 70/15/15 among unassigned per label
            import math, random
            label_groups = {}
            for r in unassigned:
                label_groups.setdefault(r['label'], []).append(r['path'])
            for lbl, paths in label_groups.items():
                random.shuffle(paths)
                n = len(paths)
                n_train = max(1, math.floor(n * 0.70))
                n_val = max(0, math.floor(n * 0.15))
                for i, p in enumerate(paths):
                    if i < n_train:
                        assignments[p] = 'train'
                    elif i < n_train + n_val:
                        assignments[p] = 'val'
                    else:
                        assignments[p] = 'test'

        # Build DataTable rows
        table_rows = []
        for r in rows:
            table_rows.append({
                'filename': r['filename'],
                'label': r['label'],
                'split': assignments.get(r['path'], 'unassigned'),
                'path': r['path'],
            })

        table = dash_table.DataTable(
            id='manual-split-datatable',
            data=table_rows,
            columns=[
                {'name': 'Filename', 'id': 'filename', 'editable': False},
                {'name': 'Label', 'id': 'label', 'editable': False},
                {
                    'name': 'Assign to Set',
                    'id': 'split',
                    'editable': True,
                    'presentation': 'dropdown',
                },
                {'name': 'path', 'id': 'path', 'editable': False},
            ],
            dropdown={
                'split': {
                    'options': [
                        {'label': '🏋️ Train', 'value': 'train'},
                        {'label': '🔍 Val', 'value': 'val'},
                        {'label': '🧪 Test', 'value': 'test'},
                        {'label': '— Unassigned', 'value': 'unassigned'},
                    ]
                }
            },
            style_cell={'textAlign': 'left', 'padding': '8px',
                        'overflow': 'hidden', 'textOverflow': 'ellipsis', 'maxWidth': '300px'},
            style_header={'backgroundColor': '#2E86AB', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'filter_query': '{split} = "train"'}, 'backgroundColor': '#e8f5e9'},
                {'if': {'filter_query': '{split} = "val"'}, 'backgroundColor': '#fff8e1'},
                {'if': {'filter_query': '{split} = "test"'}, 'backgroundColor': '#e3f2fd'},
                {'if': {'filter_query': '{split} = "unassigned"'}, 'backgroundColor': '#fafafa', 'color': '#999'},
            ],
            style_data={'whiteSpace': 'normal'},
            page_size=20,
            sort_action='native',
            filter_action='native',
            tooltip_duration=None,
            column_selectable=False,
            row_selectable='multi',
            selected_rows=[],
        )

        summary = _build_manual_split_summary(table_rows)
        return table, assignments, summary

    @app.callback(
        Output('manual-split-store', 'data', allow_duplicate=True),
        Output('manual-split-summary', 'children', allow_duplicate=True),
        Input('manual-split-datatable', 'data'),
        prevent_initial_call=True
    )
    def sync_manual_split_store(table_data):
        """Sync DataTable edits back into the manual-split-store."""
        if not table_data:
            return {}, ""
        assignments = {row['path']: row['split'] for row in table_data if row.get('path')}
        return assignments, _build_manual_split_summary(table_data)

    def _build_manual_split_summary(table_rows):
        """Return a small badge row showing Train/Val/Test/Unassigned counts."""
        counts = {'train': 0, 'val': 0, 'test': 0, 'unassigned': 0}
        for r in table_rows:
            counts[r.get('split', 'unassigned')] = counts.get(r.get('split', 'unassigned'), 0) + 1
        total = sum(counts.values())
        badge_style = lambda bg: {
            'display': 'inline-block', 'padding': '4px 12px', 'border-radius': '12px',
            'font-weight': 'bold', 'margin-right': '8px', 'font-size': '13px',
            'background': bg, 'color': 'white'
        }
        return html.Div([
            html.Span(f"🏋️ Train: {counts['train']}", style=badge_style('#28a745')),
            html.Span(f"🔍 Val: {counts['val']}", style=badge_style('#ffc107')),
            html.Span(f"🧪 Test: {counts['test']}", style=badge_style('#17a2b8')),
            html.Span(f"— Unassigned: {counts['unassigned']}", style=badge_style('#6c757d')),
            html.Span(f" / {total} total", style={'color': '#555', 'font-size': '13px'}),
        ])

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
         State('working-directory-store', 'data'),
         State('augmentation-enable', 'value'),
         State('augmentation-methods', 'value'),
         State('augmentation-factor', 'value'),
         State('augmentation-static-labels', 'value'),
         State('preprocessing-config', 'data'),
         State('split-mode-selector', 'value'),
         State('manual-split-store', 'data')],
        prevent_initial_call=True
    )
    def execute_feature_engineering(n_clicks, selected_labels, feature_method,
                                    normalization_method, target_window_size, sampling_rate,
                                    train_ratio, val_ratio, random_state, base_dir,
                                    augmentation_enabled, aug_methods, aug_factor,
                                    aug_static_labels_str, preprocess_config,
                                    split_mode, manual_assignments):
        """
        Main feature engineering executor.
        Applies consistent settings across all selected activity labels.
        Uses edge-value replication for windows smaller than target size.

        Workflow:
        1. Load all windows from all selected labels
        2. Pad windows to target size using edge-value replication
           (zero-padding would corrupt magnitude features for deployment)
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

        # Validate split ratios (only needed for auto mode)
        if split_mode != 'manual':
            if train_ratio is None or val_ratio is None:
                return html.Div("⚠️ Please set train and validation split ratios",
                                style={'color': '#ff9800', 'padding': '15px'}), "", {}
            test_ratio = 1.0 - train_ratio - val_ratio
            if test_ratio < 0.01:
                return html.Div(
                    f"⚠️ Invalid split: Train ({train_ratio*100:.0f}%) + "
                    f"Val ({val_ratio*100:.0f}%) leaves no room for test set",
                    style={'color': '#dc3545', 'padding': '15px'}), "", {}
        else:
            # Provide defaults so downstream code can still access these variables
            if train_ratio is None:
                train_ratio = 0.7
            if val_ratio is None:
                val_ratio = 0.15
            test_ratio = 1.0 - train_ratio - val_ratio

        try:
            # Use stored base directory or default to PERSISTENT_DIR
            base_dir = resolve_working_dir(base_dir)

            metadata_file = os.path.join(base_dir, 'metadata.json')

            # Load metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            # Step 1: Load all windows from all selected labels
            all_windows = []
            all_labels = []
            all_window_paths = []   # parallel list of source file paths (for manual split)
            label_window_counts = {}

            for dataset_name, dataset_info in metadata.items():
                label = dataset_info.get(
                    'label', dataset_name.replace('.csv', ''))
                if label in selected_labels and 'dragged_samples' in dataset_info:
                    window_files = [
                        f for f in (
                            resolve_metadata_path(path, base_dir)
                            for path in dataset_info['dragged_samples']
                        )
                        if os.path.exists(f)
                    ]

                    for window_file in window_files:
                        df_window = pd.read_csv(window_file)
                        all_windows.append(df_window)
                        all_labels.append(label)
                        all_window_paths.append(window_file)

                    label_window_counts[label] = label_window_counts.get(
                        label, 0) + len(window_files)

            if not all_windows:
                return html.Div("⚠️ No windows found for selected labels",
                                style={'color': '#dc3545', 'padding': '15px'}), "", {}

            # Step 2: Convert target window size from ms to samples
            target_window_samples = int(
                (target_window_size / 1000) * sampling_rate)

            # Minimum window size threshold: windows shorter than 30% of
            # target are discarded (too little real data for reliable features)
            min_window_samples = max(10, int(target_window_samples * 0.3))

            # Step 3: Pad windows to target size using EDGE-VALUE REPLICATION
            # IMPORTANT: Zero-padding would introduce artificial zero values
            # (e.g., acc_mag=0, gyro_mag=0) that corrupt statistical features
            # and cause massive train/deploy mismatch. Edge replication
            # preserves the signal characteristics of the real data.
            sensor_cols = SENSOR_COLUMNS
            padded_windows = []
            padding_stats = {'padded': 0, 'discarded': 0, 'original_size': 0}

            temp_labels = []   # Track labels for non-discarded windows
            temp_paths  = []   # Track source paths for non-discarded windows
            for df_window, lbl, wpath in zip(all_windows, all_labels, all_window_paths):
                current_size = len(df_window)
                if current_size < min_window_samples:
                    # Discard windows that are too short
                    padding_stats['discarded'] += 1
                    continue
                elif current_size < target_window_samples:
                    # Edge-value replication: repeat the last row to fill
                    padding_needed = target_window_samples - current_size
                    last_row = df_window.iloc[[-1]]
                    padding_df = pd.concat(
                        [last_row] * padding_needed, ignore_index=True)
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
                temp_labels.append(lbl)
                temp_paths.append(wpath)

            all_labels = temp_labels
            _window_paths_for_manual = temp_paths  # used in Step 4 manual split

            if not padded_windows:
                return html.Div(
                    f"⚠️ All windows were too short (< {min_window_samples} "
                    f"samples). Need longer data selections.",
                    style={'color': '#dc3545', 'padding': '15px'}), "", {}

            # Step 2.5: Apply data augmentation (if enabled)
            # Augmentation generates synthetic windows from real data to
            # improve model robustness.  Operates on raw sensor arrays
            # BEFORE feature extraction so that features reflect the
            # augmented signals naturally.
            aug_stats = None
            original_window_count = len(padded_windows)
            if (augmentation_enabled and 'enabled' in augmentation_enabled
                    and aug_methods and aug_factor and aug_factor >= 1):
                try:
                    # Parse user-specified static labels (comma-separated)
                    # or None for auto-detection
                    _static = None
                    if aug_static_labels_str and aug_static_labels_str.strip():
                        _static = [s.strip() for s in aug_static_labels_str.split(
                            ',') if s.strip()]
                    aug_windows, aug_labels, aug_stats = augment_windows(
                        padded_windows, all_labels, aug_methods,
                        int(aug_factor), sensor_cols,
                        random_seed=random_state,
                        static_labels=_static,
                    )
                    padded_windows.extend(aug_windows)
                    all_labels.extend(aug_labels)
                    print(f"[Augmentation] Added {len(aug_windows)} synthetic "
                          f"windows ({aug_stats['methods']}). "
                          f"Total: {len(padded_windows)} windows.")
                except Exception as aug_err:
                    print(f"[Augmentation] Warning — augmentation failed, "
                          f"continuing with original data: {aug_err}")
                    aug_stats = {'error': str(aug_err)}

            # Step 3: Extract features uniformly (with edge-padded windows)

            feature_list = []
            for df_window, label in zip(padded_windows, all_labels):
                # Map feature_method to create_feature_vector parameters
                if feature_method == 'orientation_invariant_time_only':
                    # Orientation-robust magnitude features - TIME DOMAIN ONLY
                    # 41 features (deployable to C++ devices)
                    # RECOMMENDED for deployment
                    feature_df = create_feature_vector(
                        df_window, sensor_cols, sampling_rate,
                        include_frequency=False,
                        orientation_robust=True,
                        include_per_axis=False
                    )
                elif feature_method == 'orientation_invariant':
                    # Orientation-robust magnitude features + FFT
                    # 63 features (41 time + 22 freq)
                    # Deployable in C++: orientation-robust magnitude FFT features
                    # (per-axis FFT modes remain non-deployable)
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
                    # Per-axis frequency features ONLY (no time-domain)
                    # 48 features
                    # WARNING: NOT supported in C++ deployment
                    feature_df = extract_frequency_domain_features(
                        df_window, sensor_cols, sampling_rate
                    )
                elif feature_method == 'raw':
                    # Raw sensor values (mean of window) — N features
                    features = {
                        col: df_window[col].mean()
                        for col in sensor_cols if col in df_window.columns
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
            feature_names = df_features.drop(
                'activity', axis=1).columns.tolist()
            # Note: normalization_method is stored in metadata for reference only

            # Also save raw windowed sensor data for CNN training
            # Shape: (n_windows, window_size_samples, 6)
            raw_windows = np.array([
                w[sensor_cols].values[:target_window_samples]
                for w in padded_windows
            ], dtype=np.float32)
            raw_labels = np.array(all_labels)

            # Step 4: Split AFTER combining (prevents data leakage)
            # Two modes: auto (ratio-based) or manual (per-window assignments)
            if split_mode == 'manual' and manual_assignments:
                manual_assignments = {
                    resolve_metadata_path(path, base_dir): split
                    for path, split in manual_assignments.items()
                }
                # --- Manual split ---
                # Map each window file path → set assignment.
                # The window_file_paths list was built in Step 1 (in the same
                # order as padded_windows after augmentation – augmented copies
                # are not individually tracked, so we append them to 'train').
                n_original = original_window_count  # before augmentation

                # Build per-sample assignment arrays (original windows only)
                idx_train, idx_val, idx_test = [], [], []
                for i, wpath in enumerate(_window_paths_for_manual):
                    assigned = manual_assignments.get(wpath, 'unassigned')
                    if assigned == 'train':
                        idx_train.append(i)
                    elif assigned == 'val':
                        idx_val.append(i)
                    elif assigned == 'test':
                        idx_test.append(i)
                    # 'unassigned' → excluded

                # Augmented copies (indices >= n_original) always go to train
                n_aug = len(X) - n_original
                idx_train.extend(range(n_original, n_original + n_aug))

                if not idx_train and not idx_test:
                    return html.Div(
                        "⚠️ Manual split: no windows assigned to Train or Test. "
                        "Please assign windows in Step 4 before running.",
                        style={'color': '#dc3545', 'padding': '15px'}), "", {}

                # Fall back to auto if user only assigned train (no test)
                if not idx_test:
                    return html.Div(
                        "⚠️ Manual split: no windows assigned to Test set. "
                        "Assign at least one window to Test.",
                        style={'color': '#dc3545', 'padding': '15px'}), "", {}

                idx_train = np.array(idx_train)
                idx_val   = np.array(idx_val)   if idx_val   else np.array([], dtype=int)
                idx_test  = np.array(idx_test)

                X_train, y_train = X[idx_train], y[idx_train]
                X_val,   y_val   = (X[idx_val], y[idx_val]) if len(idx_val) else (np.array([]), np.array([]))
                X_test,  y_test  = X[idx_test], y[idx_test]
                rw_train = raw_windows[idx_train]
                rw_val   = raw_windows[idx_val]   if len(idx_val) else np.array([])
                rw_test  = raw_windows[idx_test]

                # Compute effective ratios for metadata
                total_assigned = len(idx_train) + len(idx_val) + len(idx_test)
                train_ratio  = len(idx_train) / total_assigned if total_assigned else 0.7
                val_ratio    = len(idx_val)   / total_assigned if total_assigned else 0.15
                test_ratio   = len(idx_test)  / total_assigned if total_assigned else 0.15
                split_note   = "manual"
            else:
                # --- Auto split (ratio-based) ---
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
                split_note = "auto"

            # Step 5: Save results to files
            # Create training directory if it doesn't exist
            training_dir = os.path.join(base_dir, 'training')
            os.makedirs(training_dir, exist_ok=True)

            # Clean up stale training files from previous FE runs that used
            # a different feature set.  Only remove files that do NOT belong to
            # any known FE dataset (i.e. old per-file splits without metadata).
            # Build the dataset name first so we can guard against deleting our own files.

            # Encode feature method into dataset name so different FE configs
            # coexist (e.g. running_still_walking_oit33 vs _oi53).
            _FE_ABBREV = {
                'orientation_invariant_time_only': 'oit',
                'orientation_invariant': 'oi',
                'time_domain': 'td',
                'all': 'all',
                'frequency_domain': 'fd',
                'raw': 'raw',
            }
            fe_abbrev = _FE_ABBREV.get(feature_method, feature_method[:3])
            dataset_name = '_'.join(sorted(selected_labels)[:3])
            if len(selected_labels) > 3:
                dataset_name += f"_and_{len(selected_labels)-3}_more"
            dataset_name += f"_{fe_abbrev}{len(feature_names)}"

            existing_fe_meta = glob.glob(os.path.join(
                training_dir, '*_fe_metadata.json'))
            fe_dataset_names = {
                os.path.basename(f).replace('_fe_metadata.json', '')
                for f in existing_fe_meta
            }
            # Also keep the new dataset we're about to write
            fe_dataset_names.add(dataset_name)

            for f in os.listdir(training_dir):
                fpath = os.path.join(training_dir, f)
                if not os.path.isfile(fpath):
                    continue
                # Check if this file belongs to any known FE dataset
                belongs = any(f.startswith(name) for name in fe_dataset_names)
                if not belongs:
                    try:
                        os.remove(fpath)
                        print(f"Cleaned stale training file: {f}")
                    except Exception:
                        pass

            # Create DataFrames with feature names
            df_train = pd.DataFrame(X_train, columns=feature_names)
            df_train['label'] = y_train

            df_test = pd.DataFrame(X_test, columns=feature_names)
            df_test['label'] = y_test

            if len(X_val) > 0:
                df_val = pd.DataFrame(X_val, columns=feature_names)
                df_val['label'] = y_val

            # dataset_name was already computed above (before cleanup)

            # Save to CSV files
            train_file = os.path.join(
                training_dir, f"{dataset_name}_train.csv")
            test_file = os.path.join(training_dir, f"{dataset_name}_test.csv")

            df_train.to_csv(train_file, index=False)
            df_test.to_csv(test_file, index=False)

            if len(X_val) > 0:
                val_file = os.path.join(
                    training_dir, f"{dataset_name}_val.csv")
                df_val.to_csv(val_file, index=False)

            # Save raw windowed sensor data for CNN training (numpy arrays)
            np.save(os.path.join(training_dir,
                    f"{dataset_name}_raw_train.npy"), rw_train)
            np.save(os.path.join(training_dir,
                    f"{dataset_name}_raw_test.npy"), rw_test)
            np.save(os.path.join(training_dir,
                    f"{dataset_name}_raw_train_labels.npy"), y_train)
            np.save(os.path.join(training_dir,
                    f"{dataset_name}_raw_test_labels.npy"), y_test)
            if len(rw_val) > 0:
                np.save(os.path.join(training_dir,
                        f"{dataset_name}_raw_val.npy"), rw_val)
                np.save(os.path.join(training_dir,
                        f"{dataset_name}_raw_val_labels.npy"), y_val)

            # Save feature-engineering metadata so that the training &
            # code-generation stages can recover window_size, sampling_rate,
            # feature_method, etc.  One JSON per dataset in the training dir.
            fe_metadata = {
                'window_size_ms': target_window_size,
                'sampling_rate': sampling_rate,
                'window_size_samples': target_window_samples,
                'feature_method': feature_method,
                'normalization_method': normalization_method,
                'sensor_columns': sensor_cols,
                'num_channels': len(sensor_cols),
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
                'split_mode': split_note,
                'augmentation': {
                    'enabled': aug_stats is not None and 'error' not in (aug_stats or {}),
                    'methods': aug_methods if aug_stats else [],
                    'factor': int(aug_factor) if aug_factor else 0,
                    'original_windows': original_window_count,
                    'synthetic_windows': (aug_stats or {}).get('generated', 0),
                } if aug_stats else None,
                'preprocessing': preprocess_config or {},
            }
            fe_meta_file = os.path.join(
                training_dir, f"{dataset_name}_fe_metadata.json")
            with open(fe_meta_file, 'w') as f:
                json.dump(fe_metadata, f, indent=2)

            # Store lightweight reference in engineered-dataset-store
            # (full data is persisted on disk; avoid sending large arrays through dcc.Store)
            engineered_data = {
                'training_dir': training_dir,
                'dataset_name': dataset_name,
                'feature_names': feature_names,
                'scaler': normalization_method,
                'labels': selected_labels,
                'feature_method': feature_method,
                'n_train': len(y_train),
                'n_val': len(y_val),
                'n_test': len(y_test),
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
                columns=[{'name': col, 'id': col}
                         for col in stats_data[0].keys()],
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': '#2E86AB',
                              'color': 'white', 'fontWeight': 'bold'},
                style_data_conditional=[
                    {'if': {'row_index': len(stats_data) - 1},
                     'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'}
                ]
            )

            # Success message
            pad_info = f"Edge-padded windows: {padding_stats['padded']} / {original_window_count}"
            if padding_stats['discarded'] > 0:
                pad_info += f" ({padding_stats['discarded']} discarded as too short)"

            # Build augmentation info line
            if aug_stats and 'error' not in aug_stats:
                static_info = ""
                if aug_stats.get('static_labels'):
                    static_info = (
                        f" | 🛡️ {', '.join(aug_stats['static_labels'])} "
                        f"protected (micro-jitter only)"
                    )
                aug_info = (
                    f"🔄 Data augmentation: {aug_stats['generated']} "
                    f"synthetic windows added "
                    f"({', '.join(aug_stats.get('methods', []))}, "
                    f"{int(aug_factor)}× factor) — "
                    f"total {original_window_count} → {len(padded_windows)} windows"
                    f"{static_info}"
                )
                aug_li = html.Li(aug_info, style={
                                 'color': '#6f42c1', 'fontWeight': 'bold'})
            elif aug_stats and 'error' in aug_stats:
                aug_li = html.Li(
                    f"⚠️ Augmentation skipped (error: {aug_stats['error']})",
                    style={'color': '#ff9800'})
            else:
                aug_li = html.Li("Data augmentation: OFF")

            split_desc = (
                f"Manual ({len(idx_train)} train / {len(idx_val)} val / {len(idx_test)} test)"
                if split_note == 'manual'
                else f"Auto {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%"
            )

            success_msg = html.Div([
                html.H4("✅ Feature Engineering Complete!",
                        style={'color': '#28a745'}),
                html.P(
                    f"Successfully processed {total_all} windows from {len(selected_labels)} activity labels"),
                html.Ul([
                    html.Li(
                        f"Target window size: {target_window_size}ms ({target_window_samples} samples @ {sampling_rate}Hz)"),
                    html.Li(pad_info),
                    aug_li,
                    html.Li(
                        f"Features extracted: {len(feature_names)} features using '{feature_method}' method"),
                    html.Li(
                        f"Normalization: {normalization_method.title() if normalization_method != 'none' else 'None'}"),
                    html.Li(f"Train/Val/Test split: {split_desc}"),
                    html.Li(f"Random state: {random_state}"),
                    html.Li([html.Strong("💾 Saved to: "), f"{training_dir}\\"])
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
