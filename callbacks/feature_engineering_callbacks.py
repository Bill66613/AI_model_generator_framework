"""
Feature Engineering Callbacks
Handles unified feature engineering across multiple activity labels
"""

from dash import callback, Input, Output, State, no_update, html, ctx
import json
import os
import pandas as pd
import numpy as np
from pathlib import Path
from dash import dash_table
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import glob

from config.config import (
    PERSISTENT_DIR, METADATA_FILE, WINDOWS_DIR
)
from utils.model_training import extract_time_domain_features, extract_frequency_domain_features


# Placeholder callbacks - to be implemented

@callback(
    Output('activity-labels-selector', 'options'),
    Input('tabs', 'value')
)
def populate_activity_labels(tab):
    """
    Populate the activity labels dropdown with all available datasets
    that have split windows ready for feature engineering.
    """
    if not os.path.exists(METADATA_FILE):
        return []
    
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        # Find all unique activity labels that have dragged_samples (windows)
        labels_with_windows = set()
        for dataset_name, dataset_info in metadata.items():
            if 'dragged_samples' in dataset_info and dataset_info['dragged_samples']:
                # Get the label from metadata
                label = dataset_info.get('label', dataset_name.replace('.csv', ''))
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


@callback(
    Output('windows-per-label-display', 'children'),
    Input('activity-labels-selector', 'value')
)
def update_windows_per_label(selected_labels):
    """
    Display the number of windows available for each selected label.
    """
    if not selected_labels:
        return html.Div("No labels selected", style={'color': '#999', 'font-style': 'italic'})
    
    if not os.path.exists(METADATA_FILE):
        return html.Div("No metadata found", style={'color': '#ff0000'})
    
    try:
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        # Count windows per label
        label_counts = {}
        for dataset_name, dataset_info in metadata.items():
            label = dataset_info.get('label', dataset_name.replace('.csv', ''))
            if label in selected_labels and 'dragged_samples' in dataset_info:
                # Count existing files
                window_count = len([f for f in dataset_info['dragged_samples'] if os.path.exists(f)])
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
                    html.Span(f"{label.title()}: ", style={'font-weight': 'bold'}),
                    html.Span(f"{count} windows", style={'color': '#28a745'})
                ], style={'margin-bottom': '8px'})
            )
        
        items.append(
            html.Div([
                html.Span("Total: ", style={'font-weight': 'bold', 'color': '#2E86AB'}),
                html.Span(f"{total_windows} windows", style={'color': '#2E86AB', 'font-weight': 'bold'})
            ], style={'margin-top': '10px', 'padding-top': '10px', 'border-top': '2px solid #dee2e6'})
        )
        
        return html.Div(items)
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}", style={'color': '#ff0000'})


@callback(
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


@callback(
    Output('feature-count-display', 'children'),
    Input('global-feature-selection', 'value')
)
def update_feature_count(feature_selection):
    """
    Display the total number of features based on the selection method.
    """
    feature_counts = {
        'all': ('138 features', 'Time-domain (90) + Frequency-domain (48)'),
        'time_domain': ('90 features', 'Time-domain only (mean, std, max, min, etc.)'),
        'frequency_domain': ('48 features', 'Frequency-domain only (FFT features)'),
        'raw': ('6 features', 'Raw sensor axes (aX, aY, aZ, gX, gY, gZ)')
    }
    
    if feature_selection in feature_counts:
        count, description = feature_counts[feature_selection]
        return html.Div([
            html.Div(count, style={'font-size': '20px', 'font-weight': 'bold', 'color': '#28a745'}),
            html.Div(description, style={'font-size': '13px', 'color': '#666', 'margin-top': '5px'})
        ])
    
    return "Select a feature extraction method"


@callback(
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


@callback(
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
     State('global-random-state', 'value')],
    prevent_initial_call=True
)
def execute_feature_engineering(n_clicks, selected_labels, feature_method, 
                                normalization_method, target_window_size, sampling_rate,
                                train_ratio, val_ratio, random_state):
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
        # Load metadata
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        
        # Step 1: Load all windows from all selected labels
        all_windows = []
        all_labels = []
        label_window_counts = {}
        
        for dataset_name, dataset_info in metadata.items():
            label = dataset_info.get('label', dataset_name.replace('.csv', ''))
            if label in selected_labels and 'dragged_samples' in dataset_info:
                window_files = [f for f in dataset_info['dragged_samples'] if os.path.exists(f)]
                
                for window_file in window_files:
                    df_window = pd.read_csv(window_file)
                    all_windows.append(df_window)
                    all_labels.append(label)
                
                label_window_counts[label] = label_window_counts.get(label, 0) + len(window_files)
        
        if not all_windows:
            return html.Div("⚠️ No windows found for selected labels", 
                           style={'color': '#dc3545', 'padding': '15px'}), "", {}
        
        # Step 2: Zero-pad windows to target size if needed
        sensor_cols = ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']
        padded_windows = []
        padding_stats = {'padded': 0, 'original_size': 0}
        
        for df_window in all_windows:
            current_size = len(df_window)
            if current_size < target_window_size:
                # Zero-pad
                padding_needed = target_window_size - current_size
                padding_df = pd.DataFrame(0, index=range(padding_needed), columns=df_window.columns)
                df_padded = pd.concat([df_window, padding_df], ignore_index=True)
                padded_windows.append(df_padded)
                padding_stats['padded'] += 1
            elif current_size > target_window_size:
                # Truncate
                df_truncated = df_window.iloc[:target_window_size]
                padded_windows.append(df_truncated)
            else:
                # Exact size
                padded_windows.append(df_window)
            
            padding_stats['original_size'] = current_size
        
        # Step 3: Extract features uniformly (with zero-padding applied)
        
        feature_list = []
        for df_window, label in zip(padded_windows, all_labels):
            # Extract time-domain features
            time_features = {}
            if feature_method in ['all', 'time_domain']:
                time_df = extract_time_domain_features(df_window, sensor_cols)
                time_features = time_df.iloc[0].to_dict()
            
            # Extract frequency-domain features
            freq_features = {}
            if feature_method in ['all', 'frequency_domain']:
                freq_df = extract_frequency_domain_features(df_window, sensor_cols, sampling_rate=100)
                freq_features = freq_df.iloc[0].to_dict()
            
            # Combine features
            if feature_method == 'raw':
                # Use raw sensor values (mean of window)
                features = {
                    'aX': df_window['aX'].mean(),
                    'aY': df_window['aY'].mean(),
                    'aZ': df_window['aZ'].mean(),
                    'gX': df_window['gX'].mean(),
                    'gY': df_window['gY'].mean(),
                    'gZ': df_window['gZ'].mean()
                }
            else:
                features = {**time_features, **freq_features}
            
            features['activity'] = label
            feature_list.append(features)
        
        # Create DataFrame
        df_features = pd.DataFrame(feature_list)
        
        # Step 3: Apply normalization uniformly (fit on ALL data)
        X = df_features.drop('activity', axis=1).values
        y = df_features['activity'].values
        feature_names = df_features.drop('activity', axis=1).columns.tolist()
        
        scaler = None
        if normalization_method == 'standard':
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        elif normalization_method == 'minmax':
            scaler = MinMaxScaler()
            X = scaler.fit_transform(X)
        elif normalization_method == 'robust':
            scaler = RobustScaler()
            X = scaler.fit_transform(X)
        # else: no normalization
        
        # Step 4: Split AFTER combining (prevents data leakage)
        test_ratio = 1.0 - train_ratio - val_ratio
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        if val_ratio > 0:
            val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
            X_train, X_val, y_train, y_val = train_test_split(
                X_trainval, y_trainval, test_size=val_ratio_adjusted, 
                random_state=random_state, stratify=y_trainval
            )
        else:
            X_train, y_train = X_trainval, y_trainval
            X_val, y_val = np.array([]), np.array([])
        
        # Step 5: Save results
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
            style_header={'backgroundColor': '#2E86AB', 'color': 'white', 'fontWeight': 'bold'},
            style_data_conditional=[
                {'if': {'row_index': len(stats_data) - 1}, 
                 'backgroundColor': '#e3f2fd', 'fontWeight': 'bold'}
            ]
        )
        
        # Success message
        success_msg = html.Div([
            html.H4("✅ Feature Engineering Complete!", style={'color': '#28a745'}),
            html.P(f"Successfully processed {total_all} windows from {len(selected_labels)} activity labels"),
            html.Ul([
                html.Li(f"Target window size: {target_window_size} samples ({target_window_size/sampling_rate*1000:.0f}ms @ {sampling_rate}Hz)"),
                html.Li(f"Zero-padded windows: {padding_stats['padded']} / {total_all}"),
                html.Li(f"Features extracted: {len(feature_names)} features using '{feature_method}' method"),
                html.Li(f"Normalization: {normalization_method.title() if normalization_method != 'none' else 'None'}"),
                html.Li(f"Train/Val/Test split: {train_ratio*100:.0f}% / {val_ratio*100:.0f}% / {test_ratio*100:.0f}%"),
                html.Li(f"Random state: {random_state}")
            ])
        ], style={'padding': '15px', 'background-color': '#d4edda', 'border-radius': '5px'})
        
        return success_msg, stats_table, engineered_data
    
    except Exception as e:
        import traceback
        error_msg = html.Div([
            html.H4("❌ Error during feature engineering", style={'color': '#dc3545'}),
            html.P(str(e)),
            html.Pre(traceback.format_exc(), style={'fontSize': '12px', 'background': '#f8f9fa', 'padding': '10px'})
        ], style={'padding': '15px', 'background-color': '#f8d7da', 'border-radius': '5px'})
        return error_msg, "", {}
