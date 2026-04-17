"""
Training Module Callbacks for Human Activity Recognition Framework
Handles model training, evaluation, and deployment workflows
"""

import os
import json
import glob
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dash import Input, Output, State, no_update, dcc, html, ctx
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import base64
import io
import time
import traceback
from datetime import datetime, timedelta
import zipfile
import tempfile

logger = logging.getLogger(__name__)

from config.config import (
    PERSISTENT_DIR, METADATA_FILE, MODELS_DIR,
    get_model_path, get_models_metadata_path, get_training_data_path
)
from utils.model_training import EdgeMLModel, prepare_training_data, create_feature_vector
from deployment import generate_deployment_code, analyze_resource_requirements, generate_and_save_deployment_code


def _get_fe_train_files(training_dir):
    """Return only the *_train.csv files that belong to the latest Feature Engineering run.

    When FE metadata files exist in training_dir we restrict to those datasets
    instead of blindly globbing every ``*_train.csv`` (which may include stale
    per-file splits from the preprocessing tab with a different feature set).
    """
    fe_meta_files = glob.glob(os.path.join(training_dir, '*_fe_metadata.json'))
    if fe_meta_files:
        # Only use datasets that have a corresponding _fe_metadata.json
        fe_dataset_names = [
            os.path.basename(f).replace('_fe_metadata.json', '')
            for f in fe_meta_files
        ]
        train_files = []
        for name in fe_dataset_names:
            candidate = os.path.join(training_dir, f'{name}_train.csv')
            if os.path.exists(candidate):
                train_files.append(candidate)
        if train_files:
            return train_files

    # Fallback: no FE metadata → load all (backward compat)
    return glob.glob(os.path.join(training_dir, '*_train.csv'))


def clean_label(x):
    """Clean dataset-derived label: strip .csv extension and trailing _N suffix.

    Examples:
        'laying_1.csv' -> 'laying'
        'walking_downstairs_2.csv' -> 'walking_downstairs'
        'sitting' -> 'sitting'
    """
    if pd.notna(x):
        label = str(x).replace('.csv', '')
        parts = label.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return label
    return x


# ---- Utility functions (module-level for testability and reuse) ----

def save_model_metadata(model_filename, model_info, base_dir=None):
    """Save model metadata to the models database.

    Args:
        model_filename: Name of the model file (used as key in the JSON database).
        model_info: Dict with model info (model_type, timestamp, test_accuracy, etc.).
        base_dir: Base persistent-data directory. Defaults to PERSISTENT_DIR.
    """
    if not base_dir:
        base_dir = PERSISTENT_DIR

    models_dir = os.path.join(base_dir, 'models')
    # Ensure models directory exists
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    model_metadata_file = os.path.join(models_dir, 'trained_models.json')

    if os.path.exists(model_metadata_file):
        with open(model_metadata_file, 'r') as f:
            models_metadata = json.load(f)
    else:
        models_metadata = {}

    models_metadata[model_filename] = model_info

    with open(model_metadata_file, 'w') as f:
        json.dump(models_metadata, f, indent=2)


def load_trained_model_options(base_dir=None):
    """Load available trained model options for dropdown.

    Args:
        base_dir: Base persistent-data directory. Defaults to PERSISTENT_DIR.

    Returns:
        list[dict]: List of dicts with 'label' and 'value' keys suitable for
        Dash dropdown ``options`` property.
    """
    if not base_dir:
        base_dir = PERSISTENT_DIR

    model_options = []
    try:
        models_dir = os.path.join(base_dir, 'models')
        model_metadata_file = os.path.join(
            models_dir, 'trained_models.json')

        if os.path.exists(model_metadata_file):
            with open(model_metadata_file, 'r') as f:
                models_metadata = json.load(f)

            for model_filename, model_info in models_metadata.items():
                model_options.append({
                    'label': f"📊 {model_info['model_type'].replace('_', ' ').title()} - {model_info['timestamp']} (Acc: {model_info['test_accuracy']:.3f})",
                    'value': model_filename
                })
    except Exception as e:
        print(f"Error loading trained models: {e}")
    return model_options


def load_training_data_summary(base_dir=None):
    """Load and display summary of available training data.

    Args:
        base_dir: Base persistent-data directory. Defaults to PERSISTENT_DIR.

    Returns:
        dash.html.Div: A Dash HTML component showing dataset overview and
        activity class distribution, or a warning message if no data is found.
    """
    if not base_dir:
        base_dir = PERSISTENT_DIR

    try:
        training_dir = os.path.join(base_dir, 'training')
        if not os.path.exists(training_dir):
            return html.Div([
                html.P("⚠️ No training data found. Please complete the train-validation-test split in the Preprocessing tab.",
                       style={'text-align': 'center', 'color': '#856404', 'font-style': 'italic'}),
                html.P(f"Looking in: {training_dir}",
                       style={'text-align': 'center', 'color': '#999', 'font-size': '11px', 'margin-top': '10px'})
            ])

        # Find available datasets (prefer FE-produced files over stale splits)
        train_files = _get_fe_train_files(training_dir)
        if not train_files:
            return html.Div([
                html.P("⚠️ No training data found. Please complete Feature Engineering first.",
                       style={'text-align': 'center', 'color': '#856404', 'font-style': 'italic'}),
                html.P(f"Looking in: {training_dir}",
                       style={'text-align': 'center', 'color': '#999', 'font-size': '11px', 'margin-top': '10px'})
            ])

        # Collect data statistics
        all_train_dfs = []
        all_test_dfs = []
        all_val_dfs = []
        all_feature_cols = set()

        # Load all datasets to get statistics
        for train_file in train_files:
            dataset_name = os.path.basename(
                train_file).replace('_train.csv', '')
            test_file = get_training_data_path(
                dataset_name, 'test', base_dir)
            val_file = get_training_data_path(
                dataset_name, 'val', base_dir)

            if os.path.exists(train_file) and os.path.exists(test_file):
                train_df = pd.read_csv(train_file)
                test_df = pd.read_csv(test_file)

                # Get feature columns
                feature_cols = [
                    col for col in train_df.columns if col != 'label']
                all_feature_cols.update(feature_cols)

                all_train_dfs.append(train_df)
                all_test_dfs.append(test_df)

                if os.path.exists(val_file):
                    val_df = pd.read_csv(val_file)
                    all_val_dfs.append(val_df)

        # Combine datasets
        train_df = pd.concat(
            all_train_dfs, ignore_index=True) if all_train_dfs else None
        test_df = pd.concat(
            all_test_dfs, ignore_index=True) if all_test_dfs else None
        val_df = pd.concat(
            all_val_dfs, ignore_index=True) if all_val_dfs else None

        if train_df is None:
            return html.P(
                "⚠️ Error loading training data.",
                style={'text-align': 'center',
                       'color': '#721c24', 'margin': '0'}
            )

        # Clean labels
        train_df['label'] = train_df['label'].apply(clean_label)
        test_df['label'] = test_df['label'].apply(clean_label)
        if val_df is not None:
            val_df['label'] = val_df['label'].apply(clean_label)

        # Get unique labels
        unique_labels = sorted(train_df['label'].unique())
        label_counts = train_df['label'].value_counts()

        # Build summary display
        return html.Div([
            html.Div([
                html.Div([
                    html.H5("📊 Dataset Overview", style={
                            'color': '#2E86AB', 'margin-bottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.Strong("Training Samples: "),
                            html.Span(f"{len(train_df):,}", style={
                                      'color': '#28a745', 'font-size': '18px', 'font-weight': 'bold'})
                        ], style={'margin-bottom': '8px'}),
                        html.Div([
                            html.Strong("Validation Samples: "),
                            html.Span(f"{len(val_df):,}" if val_df is not None else "0",
                                      style={'color': '#17a2b8', 'font-size': '18px', 'font-weight': 'bold'})
                        ], style={'margin-bottom': '8px'}),
                        html.Div([
                            html.Strong("Test Samples: "),
                            html.Span(f"{len(test_df):,}", style={
                                      'color': '#ffc107', 'font-size': '18px', 'font-weight': 'bold'})
                        ], style={'margin-bottom': '8px'}),
                        html.Div([
                            html.Strong("Features: "),
                            html.Span(f"{len(all_feature_cols)}", style={
                                      'color': '#6610f2', 'font-size': '18px', 'font-weight': 'bold'})
                        ])
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-right': '2%'}),

                html.Div([
                    html.H5("🏷️ Activity Classes", style={
                            'color': '#2E86AB', 'margin-bottom': '15px'}),
                    html.Div([
                        html.Div([
                            html.Strong("Total Classes: "),
                            html.Span(f"{len(unique_labels)}", style={
                                      'color': '#dc3545', 'font-size': '18px', 'font-weight': 'bold'})
                        ], style={'margin-bottom': '10px'}),
                        html.Div([
                            html.Ul([
                                html.Li([
                                    html.Span(f"{label}", style={
                                              'font-weight': 'bold'}),
                                    html.Span(f" ({label_counts[label]} samples)", style={
                                              'color': '#6c757d', 'font-size': '14px'})
                                ]) for label in unique_labels
                            ], style={'margin': '0', 'padding-left': '20px'})
                        ])
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-left': '2%'})
            ]),

            html.Hr(style={'margin': '20px 0', 'border-color': '#dee2e6'}),

            html.Div([
                html.P([
                    html.Strong("📁 Datasets Loaded: "),
                    html.Span(f"{len(train_files)} activity dataset(s)", style={
                              'color': '#28a745'})
                ], style={'margin': '0', 'text-align': 'center', 'color': '#495057'})
            ])
        ])

    except Exception as e:
        print(f"Error loading training data summary: {e}")
        import traceback
        traceback.print_exc()
        return html.P(
            f"⚠️ Error loading training data: {str(e)}",
            style={'text-align': 'center',
                   'color': '#721c24', 'margin': '0'}
        )



def register_callbacks(app):
    """Register all callbacks with the app."""
    @app.callback(
        [Output('model-type-selector', 'disabled'),
         Output('start-training-btn', 'disabled'),
         Output('optimize-hyperparams-btn', 'disabled'),
         Output('cross-validate-btn', 'disabled'),
         Output('trained-model-selector', 'options'),
         Output('training-data-summary', 'children')],
        Input('tabs', 'value'),
        State('working-directory-store', 'data')
    )
    def enable_training_components(tab, base_dir):
        """Enable training components when training tab is active and load trained models."""
        # Use stored base directory or default to PERSISTENT_DIR
        if not base_dir:
            base_dir = PERSISTENT_DIR

        # Load available trained models using helper function
        model_options = load_trained_model_options(base_dir)

        # Load training data summary
        data_summary = load_training_data_summary(base_dir)

        if tab == 'tab-4':  # Model Training tab
            return False, False, False, False, model_options, data_summary
        return True, True, True, True, model_options, data_summary

    def create_training_results_display(model_info, evaluation_results, y_test, model_type, training_time):
        """Create comprehensive training results display with detailed metrics."""

        # Extract classification report for detailed metrics
        class_report = evaluation_results.get('classification_report', {})

        # Calculate overall metrics
        train_acc = model_info.get('train_accuracy', 0)
        test_acc = model_info['test_accuracy']
        val_acc = model_info.get('val_accuracy', 0)
        cv_acc = model_info.get('cv_accuracy', 0)

        # Determine performance status
        if test_acc >= 0.95:
            status_color = '#28a745'  # Excellent - Green
            status_text = "Excellent"
            status_icon = "🌟"
        elif test_acc >= 0.85:
            status_color = '#17a2b8'  # Good - Blue
            status_text = "Good"
            status_icon = "✅"
        elif test_acc >= 0.75:
            status_color = '#ffc107'  # Fair - Yellow
            status_text = "Fair"
            status_icon = "⚠️"
        else:
            status_color = '#dc3545'  # Needs Improvement - Red
            status_text = "Needs Improvement"
            status_icon = "❌"

        # Check for overfitting
        overfitting_warning = []
        if train_acc - test_acc > 0.1:
            overfitting_warning = [
                html.Div([
                    html.Strong("⚠️ Overfitting Detected: ",
                                style={'color': '#dc3545'}),
                    html.Span(
                        f"Training accuracy ({train_acc:.4f}) is significantly higher than test accuracy ({test_acc:.4f}). "),
                    html.Span(
                        "Consider: reducing model complexity, adding regularization, or collecting more training data.")
                ], style={
                    'background-color': '#fff3cd',
                    'border-left': '4px solid #ffc107',
                    'padding': '12px',
                    'margin': '15px 0',
                    'border-radius': '4px'
                })
            ]

        # Build per-class metrics table
        per_class_metrics = []
        if class_report:
            # Extract per-class metrics (excluding 'accuracy', 'macro avg', 'weighted avg')
            class_names = [k for k in class_report.keys() if k not in [
                'accuracy', 'macro avg', 'weighted avg']]

            if class_names:
                per_class_metrics = [
                    html.H5("📊 Per-Class Performance:",
                            style={'margin-top': '20px', 'color': '#2E86AB'}),
                    html.Div([
                        html.Table([
                            html.Thead(
                                html.Tr([
                                    html.Th("Activity", style={
                                            'text-align': 'left', 'padding': '10px', 'background-color': '#e9ecef'}),
                                    html.Th("Precision", style={
                                            'text-align': 'center', 'padding': '10px', 'background-color': '#e9ecef'}),
                                    html.Th("Recall", style={
                                            'text-align': 'center', 'padding': '10px', 'background-color': '#e9ecef'}),
                                    html.Th(
                                        "F1-Score", style={'text-align': 'center', 'padding': '10px', 'background-color': '#e9ecef'}),
                                    html.Th("Support", style={
                                            'text-align': 'center', 'padding': '10px', 'background-color': '#e9ecef'})
                                ])
                            ),
                            html.Tbody([
                                html.Tr([
                                    html.Td(class_name.replace('_', ' ').title(), style={
                                            'padding': '8px', 'font-weight': 'bold'}),
                                    html.Td(f"{class_report[class_name]['precision']:.3f}",
                                            style={'text-align': 'center', 'padding': '8px',
                                                   'color': '#28a745' if class_report[class_name]['precision'] >= 0.9 else '#495057'}),
                                    html.Td(f"{class_report[class_name]['recall']:.3f}",
                                            style={'text-align': 'center', 'padding': '8px',
                                                   'color': '#28a745' if class_report[class_name]['recall'] >= 0.9 else '#495057'}),
                                    html.Td(f"{class_report[class_name]['f1-score']:.3f}",
                                            style={'text-align': 'center', 'padding': '8px',
                                                   'color': '#28a745' if class_report[class_name]['f1-score'] >= 0.9 else '#495057'}),
                                    html.Td(f"{int(class_report[class_name]['support'])}",
                                            style={'text-align': 'center', 'padding': '8px'})
                                ], style={'border-bottom': '1px solid #dee2e6'})
                                for class_name in sorted(class_names)
                            ])
                        ], style={
                            'width': '100%',
                            'border-collapse': 'collapse',
                            'border': '1px solid #dee2e6',
                            'border-radius': '4px'
                        })
                    ], style={'overflow-x': 'auto'})
                ]

                # Add macro/weighted averages
                if 'macro avg' in class_report or 'weighted avg' in class_report:
                    avg_rows = []
                    if 'macro avg' in class_report:
                        avg_rows.append(
                            html.Tr([
                                html.Td("Macro Average", style={
                                        'padding': '8px', 'font-weight': 'bold', 'background-color': '#f8f9fa'}),
                                html.Td(f"{class_report['macro avg']['precision']:.3f}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'}),
                                html.Td(f"{class_report['macro avg']['recall']:.3f}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'}),
                                html.Td(f"{class_report['macro avg']['f1-score']:.3f}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'}),
                                html.Td(f"{int(class_report['macro avg']['support'])}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'})
                            ])
                        )
                    if 'weighted avg' in class_report:
                        avg_rows.append(
                            html.Tr([
                                html.Td("Weighted Average", style={
                                        'padding': '8px', 'font-weight': 'bold', 'background-color': '#f8f9fa'}),
                                html.Td(f"{class_report['weighted avg']['precision']:.3f}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'}),
                                html.Td(f"{class_report['weighted avg']['recall']:.3f}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'}),
                                html.Td(f"{class_report['weighted avg']['f1-score']:.3f}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'}),
                                html.Td(f"{int(class_report['weighted avg']['support'])}",
                                        style={'text-align': 'center', 'padding': '8px', 'background-color': '#f8f9fa'})
                            ])
                        )

                    if avg_rows:
                        per_class_metrics.append(
                            html.Table([
                                html.Tbody(avg_rows)
                            ], style={
                                'width': '100%',
                                'border-collapse': 'collapse',
                                'border': '1px solid #dee2e6',
                                'margin-top': '10px'
                            })
                        )

        # Build early stopping info
        early_stopping_info = []
        if model_info.get('early_stopped', False):
            early_stopping_info = [
                html.Div([
                    html.H5("⏱️ Early Stopping Applied", style={
                            'color': '#17a2b8', 'margin-bottom': '10px'}),
                    html.Div([
                        html.P([
                            html.Strong("Stopped at Epoch: "),
                            html.Span(f"{model_info.get('stopped_epoch', 'N/A')}",
                                      style={'color': '#17a2b8', 'font-size': '16px'})
                        ], style={'margin': '5px 0'}),
                        html.P([
                            html.Strong("Best Validation Accuracy: "),
                            html.Span(f"{model_info.get('val_accuracy', 0):.4f}", style={
                                      'color': '#28a745', 'font-size': '16px'})
                        ], style={'margin': '5px 0'}),
                        html.P([
                            html.Strong("Reason: "),
                            html.Span(
                                "No improvement for 10 consecutive epochs")
                        ], style={'margin': '5px 0'})
                    ])
                ], style={
                    'background-color': '#d1ecf1',
                    'border-left': '4px solid #17a2b8',
                    'padding': '15px',
                    'margin': '15px 0',
                    'border-radius': '4px'
                })
            ]

        return html.Div([
            # Success Header with Status Badge
            html.Div([
                html.Div([
                    html.H4("✅ Model Training Completed!", style={
                            'color': '#28a745', 'display': 'inline-block', 'margin-right': '15px'}),
                    html.Span([
                        html.Span(status_icon + " ",
                                  style={'font-size': '18px'}),
                        html.Span(status_text, style={'font-weight': 'bold'})
                    ], style={
                        'background-color': status_color,
                        'color': 'white',
                        'padding': '8px 16px',
                        'border-radius': '20px',
                        'display': 'inline-block',
                        'font-size': '14px'
                    })
                ], style={'margin-bottom': '10px'})
            ]),

            html.Hr(style={'margin': '20px 0'}),

            # Two-column layout for summary
            html.Div([
                # Left column - Training Configuration
                html.Div([
                    html.H5("⚙️ Training Configuration", style={
                            'color': '#2E86AB', 'margin-bottom': '15px'}),
                    html.Ul([
                        html.Li([html.Strong("Model Type: "),
                                model_type.replace('_', ' ').title()]),
                        html.Li([html.Strong("Features: "),
                                f"{model_info['features']}"]),
                        html.Li([html.Strong("Classes: "),
                                f"{model_info['classes']}"]),
                        html.Li([html.Strong("Training Time: "),
                                f"{training_time:.2f}s"]),
                    ], style={'list-style-type': 'none', 'padding': '0'})
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-right': '2%'}),

                # Right column - Dataset Split
                html.Div([
                    html.H5("📊 Dataset Split", style={
                            'color': '#2E86AB', 'margin-bottom': '15px'}),
                    html.Ul([
                        html.Li([html.Strong("Training: "),
                                f"{model_info['training_samples']} samples"]),
                        html.Li([html.Strong("Validation: "),
                                f"{model_info.get('val_samples', 0)} samples"]),
                        html.Li([html.Strong("Test: "),
                                f"{model_info['test_samples']} samples"]),
                        html.Li([html.Strong(
                            "Total: "), f"{model_info['training_samples'] + model_info.get('val_samples', 0) + model_info['test_samples']} samples"])
                    ], style={'list-style-type': 'none', 'padding': '0'})
                ], style={'width': '48%', 'display': 'inline-block', 'vertical-align': 'top', 'padding-left': '2%'})
            ], style={'margin-bottom': '20px'}),

            # Accuracy Metrics Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H6("Training", style={
                                'color': '#6c757d', 'margin': '0 0 5px 0', 'font-size': '12px'}),
                        html.Div(f"{train_acc:.1%}", style={
                                 'font-size': '24px', 'font-weight': 'bold', 'color': '#007bff'})
                    ], style={
                        'background-color': '#f8f9fa',
                        'padding': '15px',
                        'border-radius': '8px',
                        'text-align': 'center',
                        'border': '2px solid #007bff',
                        'width': '23%',
                        'display': 'inline-block',
                        'margin-right': '2%'
                    })
                ]),
                html.Div([
                    html.Div([
                        html.H6("Validation", style={
                                'color': '#6c757d', 'margin': '0 0 5px 0', 'font-size': '12px'}),
                        html.Div(f"{val_acc:.1%}" if val_acc > 0 else "N/A",
                                 style={'font-size': '24px', 'font-weight': 'bold', 'color': '#17a2b8'})
                    ], style={
                        'background-color': '#f8f9fa',
                        'padding': '15px',
                        'border-radius': '8px',
                        'text-align': 'center',
                        'border': '2px solid #17a2b8',
                        'width': '23%',
                        'display': 'inline-block',
                        'margin-right': '2%'
                    })
                ]) if val_acc > 0 else html.Div(),
                html.Div([
                    html.Div([
                        html.H6("Test", style={
                                'color': '#6c757d', 'margin': '0 0 5px 0', 'font-size': '12px'}),
                        html.Div(f"{test_acc:.1%}", style={
                                 'font-size': '24px', 'font-weight': 'bold', 'color': '#28a745'})
                    ], style={
                        'background-color': '#f8f9fa',
                        'padding': '15px',
                        'border-radius': '8px',
                        'text-align': 'center',
                        'border': '2px solid #28a745',
                        'width': '23%',
                        'display': 'inline-block',
                        'margin-right': '2%' if cv_acc == 0 else '2%'
                    })
                ]),
                html.Div([
                    html.Div([
                        html.H6(
                            "Cross-Validation", style={'color': '#6c757d', 'margin': '0 0 5px 0', 'font-size': '12px'}),
                        html.Div(f"{cv_acc:.1%}", style={
                                 'font-size': '24px', 'font-weight': 'bold', 'color': '#ffc107'})
                    ], style={
                        'background-color': '#f8f9fa',
                        'padding': '15px',
                        'border-radius': '8px',
                        'text-align': 'center',
                        'border': '2px solid #ffc107',
                        'width': '23%',
                        'display': 'inline-block'
                    })
                ]) if cv_acc > 0 else html.Div()
            ], style={'margin': '20px 0'}),

            # Overfitting Warning
            *overfitting_warning,

            # Early Stopping Info
            *early_stopping_info,

            # Per-Class Metrics
            *per_class_metrics,

            html.Hr(style={'margin': '25px 0'}),

            # Model Saved Info
            html.Div([
                html.H5("💾 Model Saved Successfully", style={
                        'color': '#28a745', 'margin-bottom': '10px'}),
                html.P([
                    html.Strong("Filename: "),
                    html.Code(os.path.basename(model_info['model_path']), style={
                        'background-color': '#f8f9fa',
                        'padding': '4px 8px',
                        'border-radius': '4px',
                        'color': '#212529'
                    })
                ], style={'margin': '5px 0'}),
                html.P([
                    html.Strong("Location: "),
                    html.Code(os.path.dirname(model_info['model_path']), style={
                        'background-color': '#f8f9fa',
                        'padding': '4px 8px',
                        'border-radius': '4px',
                        'color': '#212529',
                        'font-size': '11px'
                    })
                ], style={'margin': '5px 0'})
            ], style={
                'background-color': '#d4edda',
                'border-left': '4px solid #28a745',
                'padding': '15px',
                'border-radius': '4px',
                'margin-bottom': '25px'
            }),

            # Visualizations
            html.Hr(style={'margin': '25px 0'}),
            dcc.Graph(
                figure=create_training_results_visualization(
                    evaluation_results, y_test, model_type, model_info)
            )
        ])

    @app.callback(
        [Output('training-output', 'children'),
         Output('trained-model-selector', 'options', allow_duplicate=True)],
        [Input('start-training-btn', 'n_clicks'),
         Input('optimize-hyperparams-btn', 'n_clicks'),
         Input('cross-validate-btn', 'n_clicks')],
        [State('model-type-selector', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def handle_training_actions(train_clicks, optimize_clicks, cv_clicks, model_type, base_dir):
        """Handle different training actions based on which button was clicked.

        Note: Features are pre-computed in Feature Engineering tab.
        This training pipeline loads features from CSV files.
        """
        if not ctx.triggered:
            return no_update, no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        if not model_type:
            return (html.Div([
                html.H4("⚠ Please select a model type first.",
                        style={'color': 'orange'})
            ]), no_update)

        if not base_dir:
            base_dir = PERSISTENT_DIR

        # Feature opts for metadata (training uses pre-computed features from CSV)
        # Auto-detect feature configuration from actual feature column names
        feature_opts = {
            'orientation_robust': True,  # Will be updated below
            'include_per_axis': False,
            'include_frequency': True
        }

        # Feature-engineering acquisition parameters (window_size, sampling_rate)
        # Used by code generators to set matching #define WINDOW_SIZE / SAMPLING_RATE
        fe_config = {}

        try:
            from config.config import get_training_data_path

            # Get list of available training datasets
            training_dir = os.path.join(base_dir, 'training')
            if not os.path.exists(training_dir):
                return (html.Div([
                    html.H4("❌ No training data found.",
                            style={'color': 'red'}),
                    html.P("Please perform train-validation-test split first.")
                ]), no_update)

            # Find available datasets (prefer FE-produced files over stale splits)
            train_files = _get_fe_train_files(training_dir)
            if not train_files:
                return (html.Div([
                    html.H4("❌ No training data found.",
                            style={'color': 'red'}),
                    html.P("Please complete Feature Engineering first.")
                ]), no_update)

            # Load and combine ALL training files
            all_train_dfs = []
            all_test_dfs = []
            all_val_dfs = []

            # First pass: collect all unique feature columns
            all_feature_cols = set()
            for train_file in train_files:
                temp_df = pd.read_csv(train_file)
                feature_cols = [
                    col for col in temp_df.columns if col != 'label']
                all_feature_cols.update(feature_cols)
                print(
                    f"DEBUG: {os.path.basename(train_file)} has {len(feature_cols)} features")

            all_feature_cols = sorted(list(all_feature_cols))
            print(
                f"DEBUG: Total unique features across all files: {len(all_feature_cols)}")

            # Auto-detect feature configuration from actual column names
            has_acc_mag = any(col.startswith('acc_mag_') for col in all_feature_cols)
            has_gyro_mag = any(col.startswith('gyro_mag_') for col in all_feature_cols)
            has_per_axis = any(col.startswith(('aX_', 'aY_', 'aZ_', 'gX_', 'gY_', 'gZ_'))
                               for col in all_feature_cols)
            has_frequency = any('dominant_frequency' in col or 'spectral_energy' in col
                                for col in all_feature_cols)

            if has_acc_mag and has_gyro_mag:
                feature_opts['orientation_robust'] = True
            elif has_per_axis:
                feature_opts['orientation_robust'] = False
            # else keep default True

            feature_opts['include_per_axis'] = has_per_axis
            feature_opts['include_frequency'] = has_frequency

            logger.debug(f"DEBUG: Auto-detected feature_opts: {feature_opts}")

            # Load FE metadata (window_size_ms, sampling_rate, etc.) if available
            fe_meta_files = glob.glob(os.path.join(training_dir, '*_fe_metadata.json'))
            if fe_meta_files:
                try:
                    with open(fe_meta_files[0], 'r') as f:
                        fe_config = json.load(f)
                    logger.debug(f"Loaded FE metadata: window={fe_config.get('window_size_ms')}ms, "
                          f"rate={fe_config.get('sampling_rate')}Hz, "
                          f"method={fe_config.get('feature_method')}")
                    # Also override feature_opts from FE metadata if present
                    if 'orientation_robust' in fe_config:
                        feature_opts['orientation_robust'] = fe_config['orientation_robust']
                    if 'include_per_axis' in fe_config:
                        feature_opts['include_per_axis'] = fe_config['include_per_axis']
                    if 'include_frequency' in fe_config:
                        feature_opts['include_frequency'] = fe_config['include_frequency']
                except Exception as e:
                    print(f"WARNING: Could not load FE metadata: {e}")

            # Second pass: load data and align columns
            for train_file in train_files:
                dataset_name = os.path.basename(
                    train_file).replace('_train.csv', '')
                test_file = get_training_data_path(
                    dataset_name, 'test', base_dir)
                val_file = get_training_data_path(
                    dataset_name, 'val', base_dir)

                if os.path.exists(train_file) and os.path.exists(test_file):
                    # Load and clean labels
                    train_df = pd.read_csv(train_file)
                    test_df = pd.read_csv(test_file)

                    # Clean labels: remove .csv extension and trailing numeric suffix only
                    # "laying_1.csv" -> "laying", "walking_downstairs_2.csv" -> "walking_downstairs"
                    train_df['label'] = train_df['label'].apply(clean_label)
                    test_df['label'] = test_df['label'].apply(clean_label)

                    # Align columns - add missing features with 0
                    for col in all_feature_cols:
                        if col not in train_df.columns:
                            train_df[col] = 0.0
                        if col not in test_df.columns:
                            test_df[col] = 0.0

                    # Reorder columns to match
                    train_df = train_df[all_feature_cols + ['label']]
                    test_df = test_df[all_feature_cols + ['label']]

                    all_train_dfs.append(train_df)
                    all_test_dfs.append(test_df)

                    if os.path.exists(val_file):
                        val_df = pd.read_csv(val_file)
                        val_df['label'] = val_df['label'].apply(clean_label)

                        # Align columns
                        for col in all_feature_cols:
                            if col not in val_df.columns:
                                val_df[col] = 0.0
                        val_df = val_df[all_feature_cols + ['label']]

                        all_val_dfs.append(val_df)

            if not all_train_dfs or not all_test_dfs:
                return (html.Div([
                    html.H4("❌ No valid training data found.",
                            style={'color': 'red'}),
                    html.P("Please check your training data files.")
                ]), no_update)

            # Combine all datasets
            train_df = pd.concat(all_train_dfs, ignore_index=True)
            test_df = pd.concat(all_test_dfs, ignore_index=True)

            # Debug: Check what's in the data
            logger.debug(f"DEBUG: Combined {len(all_train_dfs)} training files")
            logger.debug(f"DEBUG: Train shape: {train_df.shape}")
            print(
                f"DEBUG: Unique labels in train: {train_df['label'].unique()}")
            print(
                f"DEBUG: Label counts in train:\n{train_df['label'].value_counts()}")

            # Handle NaN values before training
            # Check for NaN in features
            feature_cols = [col for col in train_df.columns if col != 'label']
            nan_counts_train = train_df[feature_cols].isna().sum()
            nan_counts_test = test_df[feature_cols].isna().sum()

            if nan_counts_train.sum() > 0:
                print(
                    f"WARNING: Found {nan_counts_train.sum()} NaN values in training features")
                print(
                    f"NaN counts per feature:\n{nan_counts_train[nan_counts_train > 0]}")
                # Replace NaN with 0
                train_df[feature_cols] = train_df[feature_cols].fillna(0)

            if nan_counts_test.sum() > 0:
                print(
                    f"WARNING: Found {nan_counts_test.sum()} NaN values in test features")
                # Replace NaN with 0
                test_df[feature_cols] = test_df[feature_cols].fillna(0)

            # Check for NaN in labels
            if train_df['label'].isna().any():
                print(
                    f"WARNING: Found {train_df['label'].isna().sum()} NaN labels in training data - removing these rows")
                train_df = train_df.dropna(subset=['label'])

            if test_df['label'].isna().any():
                print(
                    f"WARNING: Found {test_df['label'].isna().sum()} NaN labels in test data - removing these rows")
                test_df = test_df.dropna(subset=['label'])

            X_train = train_df.drop('label', axis=1)
            y_train = train_df['label'].values
            X_test = test_df.drop('label', axis=1)
            y_test = test_df['label'].values

            logger.debug(f"DEBUG: y_train unique: {np.unique(y_train)}")
            logger.debug(f"DEBUG: y_test unique: {np.unique(y_test)}")
            print(
                f"DEBUG: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

            # Check for validation data
            X_val, y_val = None, None
            if all_val_dfs:
                val_df = pd.concat(all_val_dfs, ignore_index=True)

                # Handle NaN in validation data
                feature_cols = [
                    col for col in val_df.columns if col != 'label']
                if val_df[feature_cols].isna().sum().sum() > 0:
                    print(
                        f"WARNING: Found {val_df[feature_cols].isna().sum().sum()} NaN values in validation features")
                    val_df[feature_cols] = val_df[feature_cols].fillna(0)

                if val_df['label'].isna().any():
                    print(
                        f"WARNING: Found {val_df['label'].isna().sum()} NaN labels in validation data - removing these rows")
                    val_df = val_df.dropna(subset=['label'])

                X_val = val_df.drop('label', axis=1)
                y_val = val_df['label'].values
                logger.debug(f"DEBUG: y_val unique: {np.unique(y_val)}")

            # Create model
            model = EdgeMLModel(model_type)

            # For CNN: load raw windowed data instead of features
            if model_type == 'pytorch_cnn':
                training_dir = os.path.join(base_dir, 'training')
                # Prefer FE-matched raw files, fallback to all
                fe_meta_files = glob.glob(os.path.join(training_dir, '*_fe_metadata.json'))
                if fe_meta_files:
                    fe_names = [os.path.basename(f).replace('_fe_metadata.json', '') for f in fe_meta_files]
                    raw_train_files = [os.path.join(training_dir, f'{n}_raw_train.npy')
                                       for n in fe_names
                                       if os.path.exists(os.path.join(training_dir, f'{n}_raw_train.npy'))]
                else:
                    raw_train_files = glob.glob(os.path.join(training_dir, '*_raw_train.npy'))
                if not raw_train_files:
                    return (html.Div([
                        html.H4("❌ No raw window data found for CNN", style={'color': 'red'}),
                        html.P("Please re-run Feature Engineering to generate raw window data."),
                    ]), no_update)
                # Load and concatenate if multiple datasets
                rw_trains, rw_tests, rw_vals = [], [], []
                rl_trains, rl_tests, rl_vals = [], [], []
                for rf in raw_train_files:
                    prefix = rf.replace('_raw_train.npy', '')
                    rw_trains.append(np.load(rf))
                    rl_trains.append(np.load(prefix + '_raw_train_labels.npy', allow_pickle=True))
                    test_f = prefix + '_raw_test.npy'
                    if os.path.exists(test_f):
                        rw_tests.append(np.load(test_f))
                        rl_tests.append(np.load(prefix + '_raw_test_labels.npy', allow_pickle=True))
                    val_f = prefix + '_raw_val.npy'
                    if os.path.exists(val_f):
                        rw_vals.append(np.load(val_f))
                        rl_vals.append(np.load(prefix + '_raw_val_labels.npy', allow_pickle=True))

                X_train = np.concatenate(rw_trains, axis=0)
                y_train = np.concatenate(rl_trains, axis=0)
                X_test = np.concatenate(rw_tests, axis=0) if rw_tests else X_train[:1]
                y_test = np.concatenate(rl_tests, axis=0) if rl_tests else y_train[:1]
                X_val = np.concatenate(rw_vals, axis=0) if rw_vals else None
                y_val = np.concatenate(rl_vals, axis=0) if rw_vals else None
                print(f"CNN raw windows: train={X_train.shape}, test={X_test.shape}")

            if button_id == 'start-training-btn':
                return perform_basic_training(model, X_train, X_test, y_train, y_test, model_type, X_val, y_val, base_dir, feature_opts, fe_config)
            elif button_id == 'optimize-hyperparams-btn':
                return perform_hyperparameter_optimization(model, X_train, X_test, y_train, y_test, model_type, X_val, y_val, base_dir, feature_opts, fe_config)
            elif button_id == 'cross-validate-btn':
                # CV doesn't use validation set
                return perform_cross_validation(model, X_train, y_train, model_type, base_dir, feature_opts, fe_config)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return (html.Div([
                html.H4("❌ Training failed", style={'color': 'red'}),
                html.P(f"Error: {str(e)}"),
                html.Pre(error_details, style={
                         'font-size': '10px', 'max-height': '200px', 'overflow': 'auto'})
            ]), no_update)

    def perform_basic_training(model, X_train, X_test, y_train, y_test, model_type, X_val=None, y_val=None, base_dir=None, feature_opts=None, fe_config=None):
        """Perform basic model training with optional validation set."""
        if not base_dir:
            base_dir = PERSISTENT_DIR

        if feature_opts is None:
            feature_opts = {'orientation_robust': True,
                            'include_per_axis': False, 'include_frequency': True}
        if not base_dir:
            base_dir = PERSISTENT_DIR

        start_time = time.time()

        # Training - use CV only if no validation set provided
        use_cv = (X_val is None or y_val is None)

        # Pass validation data to train() for early stopping
        training_results = model.train(
            X_train, y_train,
            use_cross_validation=use_cv,
            X_val=X_val,
            y_val=y_val
        )

        # Evaluation on validation set if available (not needed for early stopping models)
        val_accuracy = None
        if not use_cv:
            # Check if already evaluated during training (early stopping)
            if 'best_val_accuracy' in training_results:
                val_accuracy = training_results['best_val_accuracy']
            else:
                val_predictions = model.predict(X_val)
                val_accuracy = np.mean(val_predictions == y_val)

        # Evaluation on test set
        evaluation_results = model.evaluate(X_test, y_test)

        training_time = time.time() - start_time

        # Save trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_har_model_{timestamp}.joblib"
        model_path = get_model_path(model_filename, base_dir)
        model.save_model(model_path)

        # Update metadata with model info
        model_info = {
            'model_path': model_path,
            'model_type': model_type,
            'training_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'test_samples': len(X_test),
            'features': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'num_features_extracted': len(fe_config.get('feature_names', [])) if fe_config else None,
            'classes': len(set(y_train)),
            'train_accuracy': training_results['train_accuracy'],
            'val_accuracy': val_accuracy if val_accuracy is not None else 0,
            'test_accuracy': evaluation_results['test_accuracy'],
            'cv_accuracy': training_results.get('cv_mean_accuracy', 0) if use_cv else 0,
            'used_validation': not use_cv,
            'early_stopped': training_results.get('early_stopped', False),
            'stopped_epoch': training_results.get('stopped_epoch', None),
            # Include full training results for visualization
            'performance_metrics': training_results,
            'training_time': training_time,
            'timestamp': timestamp,
            # Feature configuration for deployment
            'feature_config': feature_opts,
            # Feature-engineering acquisition parameters (window_size_ms, sampling_rate)
            # Used by code generators to match training conditions exactly
            'fe_config': fe_config if fe_config else {}
        }

        # Store model metadata
        save_model_metadata(model_filename, model_info, base_dir)

        # Create results summary
        training_output = create_training_results_display(
            model_info, evaluation_results, y_test, model_type, training_time)
        updated_options = load_trained_model_options(base_dir)
        return training_output, updated_options

    def perform_hyperparameter_optimization(model, X_train, X_test, y_train, y_test, model_type, X_val=None, y_val=None, base_dir=None, feature_opts=None, fe_config=None):
        """Perform hyperparameter optimization with optional validation set."""
        if not base_dir:
            base_dir = PERSISTENT_DIR

        if feature_opts is None:
            feature_opts = {'orientation_robust': True,
                            'include_per_axis': False, 'include_frequency': True}

        start_time = time.time()

        # Hyperparameter optimization - uses validation set if available, otherwise CV
        optimization_results = model.optimize_hyperparameters(
            X_train, y_train,
            X_val=X_val,
            y_val=y_val
        )

        # Evaluate on validation set if available (already done during optimization)
        val_accuracy = None
        if X_val is not None and y_val is not None:
            val_predictions = model.predict(X_val)
            val_accuracy = np.mean(val_predictions == y_val)

        # Evaluate optimized model on test set
        evaluation_results = model.evaluate(X_test, y_test)

        # Get training accuracy for performance metrics
        train_predictions = model.predict(X_train)
        train_accuracy = np.mean(train_predictions == y_train)

        training_time = time.time() - start_time

        # Save optimized model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_optimized_{timestamp}.joblib"
        model_path = get_model_path(model_filename, base_dir)
        model.save_model(model_path)

        # Build performance metrics for detailed evaluation
        performance_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': evaluation_results['test_accuracy'],
            'confusion_matrix': evaluation_results.get('confusion_matrix', []),
            'classification_report': evaluation_results.get('classification_report', {}),
            'label_names': evaluation_results.get('label_names', [])
        }

        # Add validation metrics if available
        if val_accuracy is not None:
            performance_metrics['val_accuracy'] = val_accuracy

        # Update metadata
        model_info = {
            'model_path': model_path,
            'model_type': model_type,
            'training_samples': len(X_train),
            'val_samples': len(X_val) if X_val is not None else 0,
            'test_samples': len(X_test),
            'features': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'num_features_extracted': len(fe_config.get('feature_names', [])) if fe_config else None,
            'classes': len(set(y_train)),
            'train_accuracy': train_accuracy,
            'val_accuracy': val_accuracy if val_accuracy is not None else 0,
            'test_accuracy': evaluation_results['test_accuracy'],
            'best_params': optimization_results['best_params'],
            'optimization_score': optimization_results['best_score'],
            'optimization_method': optimization_results.get('method', 'cross_validation'),
            'used_validation': (X_val is not None and y_val is not None),
            'performance_metrics': performance_metrics,
            'training_time': training_time,
            'timestamp': timestamp,
            'optimized': True,
            # Feature configuration for deployment
            'feature_config': feature_opts,
            'fe_config': fe_config if fe_config else {}
        }

        save_model_metadata(model_filename, model_info, base_dir)

        # Determine optimization method used
        opt_method = optimization_results.get('method', 'cross_validation')
        method_badge = "Validation Set" if opt_method == 'validation_set' else "Cross-Validation"
        method_color = '#17a2b8' if opt_method == 'validation_set' else '#ffc107'

        # Calculate improvement info if available
        test_acc = evaluation_results['test_accuracy']
        opt_score = optimization_results['best_score']

        optimization_output = html.Div([
            # Header with method badge
            html.Div([
                html.H4("✅ Hyperparameter Optimization Completed!",
                        style={'color': '#28a745', 'display': 'inline-block', 'margin-right': '15px'}),
                html.Span(method_badge, style={
                    'background-color': method_color,
                    'color': 'white',
                    'padding': '8px 16px',
                    'border-radius': '20px',
                    'font-size': '14px',
                    'font-weight': 'bold'
                })
            ], style={'margin-bottom': '15px'}),

            html.Hr(style={'margin': '20px 0'}),

            # Optimization Summary Cards
            html.Div([
                html.Div([
                    html.Div([
                        html.H6("Optimization Score", style={
                                'color': '#6c757d', 'margin': '0 0 8px 0', 'font-size': '13px'}),
                        html.Div(f"{opt_score:.1%}", style={
                                 'font-size': '28px', 'font-weight': 'bold', 'color': '#007bff'}),
                        html.P(f"via {method_badge}", style={
                               'font-size': '11px', 'color': '#6c757d', 'margin': '5px 0 0 0'})
                    ], style={
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'color': 'white',
                        'padding': '20px',
                        'border-radius': '10px',
                        'text-align': 'center',
                        'width': '30%',
                        'display': 'inline-block',
                        'margin-right': '3%',
                        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'
                    })
                ]),
                html.Div([
                    html.Div([
                        html.H6("Test Accuracy", style={
                                'color': '#6c757d', 'margin': '0 0 8px 0', 'font-size': '13px'}),
                        html.Div(f"{test_acc:.1%}", style={
                                 'font-size': '28px', 'font-weight': 'bold', 'color': '#28a745'}),
                        html.P("on holdout set", style={
                               'font-size': '11px', 'color': '#6c757d', 'margin': '5px 0 0 0'})
                    ], style={
                        'background': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)',
                        'color': 'white',
                        'padding': '20px',
                        'border-radius': '10px',
                        'text-align': 'center',
                        'width': '30%',
                        'display': 'inline-block',
                        'margin-right': '3%',
                        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'
                    })
                ]),
                html.Div([
                    html.Div([
                        html.H6("Training Time", style={
                                'color': '#6c757d', 'margin': '0 0 8px 0', 'font-size': '13px'}),
                        html.Div(f"{training_time:.1f}s", style={
                                 'font-size': '28px', 'font-weight': 'bold', 'color': '#ffc107'}),
                        html.P("optimization", style={
                               'font-size': '11px', 'color': '#6c757d', 'margin': '5px 0 0 0'})
                    ], style={
                        'background': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
                        'color': 'white',
                        'padding': '20px',
                        'border-radius': '10px',
                        'text-align': 'center',
                        'width': '30%',
                        'display': 'inline-block',
                        'box-shadow': '0 4px 6px rgba(0,0,0,0.1)'
                    })
                ])
            ], style={'margin': '25px 0'}),

            # Best Parameters Section
            html.Div([
                html.H5("⚙️ Optimized Hyperparameters", style={
                        'color': '#2E86AB', 'margin-bottom': '15px'}),
                html.Div([
                    html.Table([
                        html.Thead(
                            html.Tr([
                                html.Th("Parameter", style={
                                        'text-align': 'left', 'padding': '12px', 'background-color': '#e9ecef', 'width': '40%'}),
                                html.Th("Value", style={
                                        'text-align': 'left', 'padding': '12px', 'background-color': '#e9ecef', 'width': '60%'})
                            ])
                        ),
                        html.Tbody([
                            html.Tr([
                                html.Td(param.replace('_', ' ').title(),
                                        style={'padding': '10px', 'font-weight': 'bold', 'color': '#495057'}),
                                html.Td(html.Code(str(value), style={
                                    'background-color': '#f8f9fa',
                                    'padding': '4px 8px',
                                    'border-radius': '4px',
                                    'color': '#007bff',
                                    'font-weight': 'bold'
                                }), style={'padding': '10px'})
                            ], style={'border-bottom': '1px solid #dee2e6'})
                            for param, value in optimization_results['best_params'].items()
                        ])
                    ], style={
                        'width': '100%',
                        'border-collapse': 'collapse',
                        'border': '1px solid #dee2e6',
                        'border-radius': '6px',
                        'overflow': 'hidden'
                    })
                ])
            ], style={
                'background-color': '#ffffff',
                'padding': '20px',
                'border-radius': '8px',
                'border': '1px solid #dee2e6',
                'margin': '20px 0'
            }),

            # Model Info
            html.Div([
                html.H5("💾 Model Information", style={
                        'color': '#28a745', 'margin-bottom': '10px'}),
                html.Div([
                    html.Div([
                        html.Strong("Filename: "),
                        html.Code(model_filename, style={
                            'background-color': '#f8f9fa',
                            'padding': '4px 8px',
                            'border-radius': '4px',
                            'color': '#212529'
                        })
                    ], style={'margin-bottom': '8px'}),
                    html.Div([
                        html.Strong("Type: "),
                        html.Span(model_type.replace('_', ' ').title())
                    ], style={'margin-bottom': '8px'}),
                    html.Div([
                        html.Strong("Optimization Method: "),
                        html.Span(method_badge, style={
                                  'color': method_color, 'font-weight': 'bold'})
                    ], style={'margin-bottom': '8px'}),
                    html.Div([
                        html.Strong("Timestamp: "),
                        html.Span(timestamp)
                    ])
                ])
            ], style={
                'background-color': '#d4edda',
                'border-left': '4px solid #28a745',
                'padding': '15px',
                'border-radius': '4px',
                'margin': '20px 0'
            }),

            html.Hr(style={'margin': '25px 0'}),

            # Visualizations
            dcc.Graph(
                figure=create_training_results_visualization(
                    evaluation_results, y_test, model_type, model_info)
            )
        ])
        updated_options = load_trained_model_options(base_dir)
        return optimization_output, updated_options

    def perform_cross_validation(model, X_train, y_train, model_type, base_dir=None, feature_opts=None, fe_config=None):
        """Perform cross-validation analysis and optionally save the trained model."""
        if not base_dir:
            base_dir = PERSISTENT_DIR

        if feature_opts is None:
            feature_opts = {'orientation_robust': True,
                            'include_per_axis': False, 'include_frequency': True}

        start_time = time.time()

        # Perform training with cross-validation
        training_results = model.train(
            X_train, y_train, use_cross_validation=True)

        # Split data for test evaluation (20% holdout)
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )

        # Retrain on full training set for final model
        model.train(X_train_full, y_train_full, use_cross_validation=False)

        # Evaluate on test set
        evaluation_results = model.evaluate(X_test, y_test)

        # Get training accuracy
        train_predictions = model.predict(X_train_full)
        train_accuracy = np.mean(train_predictions == y_train_full)

        training_time = time.time() - start_time

        # Save the trained model with full evaluation data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_type}_cv_{timestamp}.joblib"
        model_path = get_model_path(model_filename, base_dir)
        model.save_model(model_path)

        # Build performance metrics for detailed evaluation
        performance_metrics = {
            'train_accuracy': train_accuracy,
            'test_accuracy': evaluation_results['test_accuracy'],
            'cv_mean_accuracy': training_results.get('cv_mean_accuracy', 0),
            'cv_std_accuracy': training_results.get('cv_std_accuracy', 0),
            'confusion_matrix': evaluation_results.get('confusion_matrix', []),
            'classification_report': evaluation_results.get('classification_report', {}),
            'label_names': evaluation_results.get('label_names', [])
        }

        # Save model metadata
        model_info = {
            'model_path': model_path,
            'model_type': model_type,
            'training_samples': len(X_train_full),
            'val_samples': 0,
            'test_samples': len(X_test),
            'features': X_train.shape[1] if len(X_train.shape) > 1 else 1,
            'num_features_extracted': len(fe_config.get('feature_names', [])) if fe_config else None,
            'classes': len(set(y_train)),
            'train_accuracy': train_accuracy,
            'val_accuracy': 0,
            'test_accuracy': evaluation_results['test_accuracy'],
            'cv_accuracy': training_results.get('cv_mean_accuracy', 0),
            'used_validation': False,
            'performance_metrics': performance_metrics,
            'training_time': training_time,
            'timestamp': timestamp,
            'cv_std_accuracy': training_results.get('cv_std_accuracy', 0),
            # Feature configuration for deployment
            'feature_config': feature_opts,
            'fe_config': fe_config if fe_config else {}
        }

        save_model_metadata(model_filename, model_info, base_dir)

        # Create cross-validation visualization
        cv_results = training_results

        cv_output = html.Div([
            html.H4("✅ Cross-Validation Analysis Completed!",
                    style={'color': 'green'}),
            html.Hr(),
            html.H5("📊 Cross-Validation Results:"),
            html.Ul([
                html.Li(f"Model Type: {model_type.replace('_', ' ').title()}"),
                html.Li(
                    f"Mean CV Accuracy: {cv_results.get('cv_mean_accuracy', 0):.4f}"),
                html.Li(
                    f"Standard Deviation: {cv_results.get('cv_std_accuracy', 0):.4f}"),
                html.Li(
                    f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}"),
                html.Li(f"Training Time: {training_time:.2f} seconds"),
                html.Li(f"Model Saved: {model_filename}", style={
                        'color': '#28a745', 'font-weight': 'bold'})
            ]),
            html.Div([
                dcc.Graph(
                    figure=create_cv_visualization(cv_results, model_type)
                )
            ])
        ])

        # Return updated model options since we saved a model
        updated_options = load_trained_model_options(base_dir)
        return cv_output, updated_options

    def create_cv_visualization(cv_results, model_type):
        """Create cross-validation results visualization."""
        # Create a simple bar chart for CV results
        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['CV Mean Accuracy'],
            y=[cv_results.get('cv_mean_accuracy', 0)],
            error_y=dict(
                type='data',
                array=[cv_results.get('cv_std_accuracy', 0)],
                visible=True
            ),
            marker_color='skyblue',
            name='Cross-Validation Score'
        ))

        fig.update_layout(
            title=f"Cross-Validation Results - {model_type.replace('_', ' ').title()}",
            yaxis_title="Accuracy Score",
            yaxis=dict(range=[0, 1]),
            height=400
        )

        return fig

    def create_training_results_visualization(evaluation_results, y_test, model_type, model_info=None):
        """Create comprehensive visualization of training results."""

        # Check if we have training history (for early stopping visualization)
        has_training_history = (model_info is not None and
                                'train_accuracies' in model_info.get('performance_metrics', {}) or
                                ('train_accuracies' in model_info if model_info else False))

        # Adjust layout based on whether we have training history
        if has_training_history:
            # 3 rows for training history graph
            fig = make_subplots(
                rows=3, cols=2,
                subplot_titles=('Confusion Matrix', 'Class Distribution',
                                'Training History', 'Model Performance'),
                specs=[[{'type': 'heatmap'}, {'type': 'bar'}],
                       [{'type': 'scatter', 'colspan': 2}, None],
                       [{'type': 'bar'}, {'type': 'indicator'}]],
                row_heights=[0.35, 0.3, 0.35]
            )
            perf_row, perf_col = 3, 2
        else:
            # Original 2x2 layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Confusion Matrix', 'Class Distribution',
                                'Feature Importance', 'Model Performance'),
                specs=[[{'type': 'heatmap'}, {'type': 'bar'}],
                       [{'type': 'bar'}, {'type': 'indicator'}]]
            )
            perf_row, perf_col = 2, 2

        # 1. Confusion Matrix with proper label names
        conf_matrix = evaluation_results['confusion_matrix']

        # Get label names from evaluation results or y_test
        label_names = evaluation_results.get('label_names', None)
        if label_names is None:
            # Fallback to unique values from y_test
            class_names = sorted([str(label) for label in set(y_test)])
        else:
            # Use the actual label names in order
            class_names = [str(name) for name in label_names]

        # Format class names for display (replace underscores, capitalize)
        display_names = [name.replace('_', ' ').title()
                         for name in class_names]

        fig.add_trace(
            go.Heatmap(
                z=conf_matrix,
                x=display_names,
                y=display_names,
                colorscale='Blues',
                showscale=True,
                text=conf_matrix,
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
            ),
            row=1, col=1
        )

        # 2. Class Distribution with proper formatting
        class_counts = pd.Series(y_test).value_counts()
        # Format the index names for display
        formatted_index = [str(label).replace('_', ' ').title()
                           for label in class_counts.index]

        fig.add_trace(
            go.Bar(
                x=formatted_index,
                y=class_counts.values,
                name='Test Set Distribution',
                marker_color='skyblue',
                text=class_counts.values,
                textposition='outside',
                hovertemplate='%{x}<br>Count: %{y}<extra></extra>'
            ),
            row=1, col=2
        )

        # 3. Training History (if available)
        if has_training_history:
            # Get training history from model_info
            perf_metrics = model_info.get('performance_metrics', model_info)
            train_accs = perf_metrics.get('train_accuracies', [])
            val_accs = perf_metrics.get('val_accuracies', [])

            epochs = list(range(1, len(train_accs) + 1))

            # Training accuracy line
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=train_accs,
                    mode='lines+markers',
                    name='Training Accuracy',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )

            # Validation accuracy line
            fig.add_trace(
                go.Scatter(
                    x=epochs,
                    y=val_accs,
                    mode='lines+markers',
                    name='Validation Accuracy',
                    line=dict(color='red', width=2),
                    marker=dict(size=6)
                ),
                row=2, col=1
            )

            # Mark early stopping point if applicable
            if model_info.get('early_stopped', False):
                stopped_epoch = model_info.get('stopped_epoch', len(epochs))
                fig.add_vline(
                    x=stopped_epoch,
                    line_dash="dash",
                    line_color="green",
                    annotation_text="Early Stop",
                    row=2, col=1
                )

            fig.update_xaxes(title_text="Epoch", row=2, col=1)
            fig.update_yaxes(title_text="Accuracy", row=2, col=1)

        # 4. Performance Indicator
        test_accuracy = evaluation_results['test_accuracy']
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=test_accuracy,
                delta={'reference': 0.8},
                gauge={'axis': {'range': [None, 1]},
                       'bar': {'color': "darkblue"},
                       'steps': [
                           {'range': [0, 0.5], 'color': "lightgray"},
                           {'range': [0.5, 0.8], 'color': "yellow"},
                           {'range': [0.8, 1], 'color': "green"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                     'thickness': 0.75, 'value': 0.9}},
                title={
                    'text': f"Test Accuracy<br>{model_type.replace('_', ' ').title()}"}
            ),
            row=perf_row, col=perf_col
        )

        # Update layout with better spacing
        fig.update_layout(
            title={
                'text': f"Model Training Results - {model_type.replace('_', ' ').title()}",
                'y': 0.98,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            height=1100 if has_training_history else 850,
            showlegend=has_training_history,
            margin=dict(l=80, r=80, t=100, b=80),
            font=dict(size=11)
        )

        # Update subplot titles and axes with better formatting
        fig.update_xaxes(title_text="Predicted", row=1,
                         col=1, title_font=dict(size=12))
        fig.update_yaxes(title_text="Actual", row=1,
                         col=1, title_font=dict(size=12))
        fig.update_xaxes(title_text="Activity Classes", row=1,
                         col=2, title_font=dict(size=12), tickangle=-45)
        fig.update_yaxes(title_text="Sample Count", row=1,
                         col=2, title_font=dict(size=12))

        # Adjust confusion matrix labels for better readability
        fig.update_xaxes(tickangle=-45, row=1, col=1)
        fig.update_yaxes(tickangle=0, row=1, col=1)

        return fig

    @app.callback(
        [Output('model-performance-graph', 'figure'),
         Output('detailed-evaluation-results', 'children')],
        [Input('evaluate-model-btn', 'n_clicks'),
         Input('feature-importance-btn', 'n_clicks')],
        [State('trained-model-selector', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def evaluate_trained_model(eval_clicks, feature_clicks, model_filename, base_dir):
        """Evaluate a trained model and show detailed performance metrics."""
        if not model_filename:
            return {}, html.Div()

        if not ctx.triggered:
            return no_update, no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]

        try:
            model_path = get_model_path(model_filename, base_dir)
            if not os.path.exists(model_path):
                return {}, html.Div()

            # Load trained model
            model = EdgeMLModel.load_model(model_path)

            if button_id == 'evaluate-model-btn':
                # Load model metadata with all performance metrics
                model_metadata_file = get_models_metadata_path(base_dir)
                model_info = {}

                if os.path.exists(model_metadata_file):
                    with open(model_metadata_file, 'r') as f:
                        models_metadata = json.load(f)
                        model_info = models_metadata.get(model_filename, {})

                logger.debug(f"DEBUG: Creating evaluation for {model_filename}")
                logger.debug(f"DEBUG: model_info keys: {list(model_info.keys())}")
                print(
                    f"DEBUG: Has performance_metrics: {'performance_metrics' in model_info}")

                graph = create_model_evaluation_plot(
                    model, model_filename, base_dir)
                detailed_results = create_detailed_evaluation_display(
                    model_info, model_filename)

                print(
                    f"DEBUG: detailed_results type: {type(detailed_results)}")
                logger.debug(f"DEBUG: Returning graph and detailed results")

                return graph, detailed_results

            elif button_id == 'feature-importance-btn':
                graph = create_feature_importance_plot(
                    model, model_filename, base_dir)
                return graph, html.Div()

            return {}, html.Div()

        except Exception as e:
            print(f"Error evaluating model: {e}")
            import traceback
            traceback.print_exc()
            return {}, html.Div([
                html.H5("❌ Error loading model evaluation",
                        style={'color': '#dc3545'}),
                html.P(f"Error: {str(e)}")
            ])

    def create_model_evaluation_plot(model, model_filename, base_dir=None):
        """Create comprehensive model evaluation visualization."""
        # Load model metadata
        model_metadata_file = get_models_metadata_path(base_dir)
        model_info = {}

        if os.path.exists(model_metadata_file):
            with open(model_metadata_file, 'r') as f:
                models_metadata = json.load(f)
                model_info = models_metadata.get(model_filename, {})

        # Create evaluation dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Model Performance Overview',
                'Training vs Test Accuracy',
                'Model Complexity Analysis',
                'Resource Requirements'
            ),
            specs=[
                [{'type': 'indicator'}, {'type': 'bar'}],
                [{'type': 'bar'}, {'type': 'table'}]
            ]
        )

        # Performance indicator
        test_accuracy = model_info.get('test_accuracy', 0)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=test_accuracy,
                delta={'reference': 0.8},
                gauge={
                    'axis': {'range': [None, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.5], 'color': "lightgray"},
                        {'range': [0.5, 0.8], 'color': "yellow"},
                        {'range': [0.8, 1], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 0.9
                    }
                },
                title={'text': "Test Accuracy"}
            ),
            row=1, col=1
        )

        # Training vs Test comparison
        accuracies = [
            model_info.get('train_accuracy', 0),
            model_info.get('test_accuracy', 0),
            model_info.get('cv_accuracy', 0)
        ]

        fig.add_trace(
            go.Bar(
                x=['Training', 'Test', 'Cross-Val'],
                y=accuracies,
                marker_color=['blue', 'orange', 'green'],
                name='Accuracy Comparison'
            ),
            row=1, col=2
        )

        # Model complexity
        complexity_metrics = [
            model_info.get('features', 0),
            model_info.get('training_samples', 0),
            model_info.get('classes', 0)
        ]

        fig.add_trace(
            go.Bar(
                x=['Features', 'Samples', 'Classes'],
                y=complexity_metrics,
                marker_color=['purple', 'cyan', 'red'],
                name='Model Complexity'
            ),
            row=2, col=1
        )

        # Resource requirements table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Metric', 'Value'],
                    fill_color='paleturquoise',
                    align='left'
                ),
                cells=dict(
                    values=[
                        ['Model Type', 'Training Time', 'Features', 'Memory Est.'],
                        [
                            model_info.get('model_type', 'Unknown').replace(
                                '_', ' ').title(),
                            f"{model_info.get('training_time', 0):.2f}s",
                            str(model_info.get('features', 0)),
                            f"~{model_info.get('features', 0) * 4}KB"
                        ]
                    ],
                    fill_color='lavender',
                    align='left'
                )
            ),
            row=2, col=2
        )

        fig.update_layout(
            title=f"Model Evaluation Dashboard - {model_filename}",
            height=600,
            showlegend=False
        )

        return fig

    def create_feature_importance_plot(model, model_filename, base_dir=None):
        """Create feature importance visualization."""
        feature_importance = model.get_feature_importance()

        if not feature_importance:
            # Create a placeholder plot
            fig = go.Figure()
            fig.add_annotation(
                text=f"Feature importance not available for {model.model_type}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False
            )
            fig.update_layout(
                title="Feature Importance Analysis",
                height=400
            )
            return fig

        # Sort features by importance
        sorted_features = sorted(feature_importance.items(),
                                 key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:20]  # Top 20 features

        features, importances = zip(*top_features)

        fig = go.Figure(go.Bar(
            x=list(importances),
            y=list(features),
            orientation='h',
            marker_color='lightcoral'
        ))

        fig.update_layout(
            title=f"Top 20 Feature Importance - {model.model_type.replace('_', ' ').title()}",
            xaxis_title="Importance Score",
            yaxis_title="Features",
            height=600
        )

        return fig

    def create_detailed_evaluation_display(model_info, model_filename):
        """Create comprehensive detailed evaluation results display with confusion matrix and metrics."""

        print(
            f"DEBUG: create_detailed_evaluation_display called for {model_filename}")
        logger.debug(f"DEBUG: model_info is None: {model_info is None}")
        print(
            f"DEBUG: model_info keys: {list(model_info.keys()) if model_info else 'None'}")

        if not model_info or 'performance_metrics' not in model_info:
            logger.debug("DEBUG: No performance_metrics found, returning info message")
            return html.Div([
                html.H4("ℹ️ No detailed evaluation data available", style={
                    'color': '#6c757d', 'text-align': 'center', 'padding': '20px'
                })
            ])

        perf_metrics = model_info.get('performance_metrics', {})
        class_report = perf_metrics.get('classification_report', {})
        confusion_mat = perf_metrics.get('confusion_matrix', [])
        label_names = perf_metrics.get('label_names', [])

        # Format label names for display
        if label_names:
            display_names = [name.replace('_', ' ').title()
                             for name in label_names]
        else:
            display_names = [f"Class {i}" for i in range(len(confusion_mat))]

        # Create confusion matrix heatmap
        confusion_fig = go.Figure(data=go.Heatmap(
            z=confusion_mat,
            x=display_names,
            y=display_names,
            colorscale='Blues',
            text=confusion_mat,
            texttemplate='%{text}',
            textfont={"size": 14},
            hovertemplate='Predicted: %{x}<br>Actual: %{y}<br>Count: %{z}<extra></extra>'
        ))

        confusion_fig.update_layout(
            title='Confusion Matrix',
            xaxis_title='Predicted Label',
            yaxis_title='Actual Label',
            height=500,
            margin=dict(l=100, r=50, t=80, b=100),
            xaxis={'tickangle': -45}
        )

        # Build per-class metrics table
        per_class_rows = []
        for class_name in label_names if label_names else []:
            if class_name in class_report:
                metrics = class_report[class_name]
                per_class_rows.append(html.Tr([
                    html.Td(class_name.replace('_', ' ').title(),
                            style={'font-weight': 'bold'}),
                    html.Td(f"{metrics.get('precision', 0):.3f}",
                            style={'text-align': 'center'}),
                    html.Td(f"{metrics.get('recall', 0):.3f}",
                            style={'text-align': 'center'}),
                    html.Td(f"{metrics.get('f1-score', 0):.3f}",
                            style={'text-align': 'center'}),
                    html.Td(f"{int(metrics.get('support', 0))}",
                            style={'text-align': 'center'})
                ]))

        # Add macro and weighted averages
        if 'macro avg' in class_report:
            macro_avg = class_report['macro avg']
            per_class_rows.append(html.Tr([
                html.Td("Macro Average", style={
                        'font-weight': 'bold', 'background-color': '#e9ecef'}),
                html.Td(f"{macro_avg.get('precision', 0):.3f}", style={
                        'text-align': 'center', 'background-color': '#e9ecef', 'font-weight': 'bold'}),
                html.Td(f"{macro_avg.get('recall', 0):.3f}", style={
                        'text-align': 'center', 'background-color': '#e9ecef', 'font-weight': 'bold'}),
                html.Td(f"{macro_avg.get('f1-score', 0):.3f}", style={'text-align': 'center',
                        'background-color': '#e9ecef', 'font-weight': 'bold'}),
                html.Td(f"{int(macro_avg.get('support', 0))}", style={
                        'text-align': 'center', 'background-color': '#e9ecef'})
            ]))

        if 'weighted avg' in class_report:
            weighted_avg = class_report['weighted avg']
            per_class_rows.append(html.Tr([
                html.Td("Weighted Average", style={
                        'font-weight': 'bold', 'background-color': '#f8f9fa'}),
                html.Td(f"{weighted_avg.get('precision', 0):.3f}", style={
                        'text-align': 'center', 'background-color': '#f8f9fa', 'font-weight': 'bold'}),
                html.Td(f"{weighted_avg.get('recall', 0):.3f}", style={
                        'text-align': 'center', 'background-color': '#f8f9fa', 'font-weight': 'bold'}),
                html.Td(f"{weighted_avg.get('f1-score', 0):.3f}", style={
                        'text-align': 'center', 'background-color': '#f8f9fa', 'font-weight': 'bold'}),
                html.Td(f"{int(weighted_avg.get('support', 0))}", style={
                        'text-align': 'center', 'background-color': '#f8f9fa'})
            ]))

        # Overall metrics summary
        train_acc = model_info.get('train_accuracy', 0)
        val_acc = model_info.get('val_accuracy', 0)
        test_acc = model_info.get('test_accuracy', 0)

        # Overfitting detection
        overfitting_warning = []
        if train_acc > 0 and test_acc > 0 and (train_acc - test_acc) > 0.1:
            overfitting_warning = [html.Div([
                html.P("⚠️ Potential Overfitting Detected", style={
                    'color': '#ff6b6b', 'font-weight': 'bold', 'margin-bottom': '5px'
                }),
                html.P(f"Training accuracy ({train_acc:.1%}) is significantly higher than test accuracy ({test_acc:.1%}). "
                       "Consider adding regularization or collecting more training data.", style={
                    'color': '#666', 'font-size': '14px'
                })
            ], style={
                'background-color': '#fff3cd',
                'border-left': '4px solid #ff6b6b',
                'padding': '15px',
                'border-radius': '5px',
                'margin-bottom': '20px'
            })]

        # Training details
        training_details = []
        if model_info.get('early_stopped'):
            training_details.append(html.Li([
                html.Strong("Early Stopping: "),
                f"Stopped at epoch {model_info.get('stopped_epoch', 'N/A')} ",
                f"(Best Val Acc: {perf_metrics.get('best_val_accuracy', 0):.3f})"
            ]))

        if model_info.get('used_validation'):
            training_details.append(html.Li([
                html.Strong("Validation Set: "),
                f"Used ({model_info.get('val_samples', 0)} samples)"
            ]))

        if model_info.get('optimized'):
            training_details.append(html.Li([
                html.Strong("Hyperparameter Optimization: "),
                f"{model_info.get('optimization_method', 'Unknown')} ",
                f"(Score: {model_info.get('optimization_score', 0):.3f})"
            ]))

        return html.Div([
            html.H3("📊 Detailed Model Evaluation Results", style={
                'color': '#2E86AB', 'margin-bottom': '20px', 'margin-top': '20px'
            }),

            # Overall Performance Summary
            html.Div([
                html.H4("🎯 Overall Performance", style={
                        'color': '#495057', 'margin-bottom': '15px'}),
                html.Div([
                    html.Div([
                        html.H2(f"{train_acc:.1%}", style={
                                'color': '#4CAF50', 'margin': '0'}),
                        html.P("Training Accuracy", style={
                               'color': '#666', 'margin': '5px 0'})
                    ], style={
                        'display': 'inline-block', 'width': '30%', 'text-align': 'center',
                        'background': 'linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%)',
                        'padding': '20px', 'border-radius': '10px', 'margin-right': '3%'
                    }),
                    html.Div([
                        html.H2(f"{val_acc:.1%}" if val_acc > 0 else "N/A",
                                style={'color': '#2196F3', 'margin': '0'}),
                        html.P("Validation Accuracy", style={
                               'color': '#666', 'margin': '5px 0'})
                    ], style={
                        'display': 'inline-block', 'width': '30%', 'text-align': 'center',
                        'background': 'linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%)',
                        'padding': '20px', 'border-radius': '10px', 'margin-right': '3%'
                    }),
                    html.Div([
                        html.H2(f"{test_acc:.1%}", style={
                                'color': '#FF9800', 'margin': '0'}),
                        html.P("Test Accuracy", style={
                               'color': '#666', 'margin': '5px 0'})
                    ], style={
                        'display': 'inline-block', 'width': '30%', 'text-align': 'center',
                        'background': 'linear-gradient(135deg, #fff3e0 0%, #ffe0b2 100%)',
                        'padding': '20px', 'border-radius': '10px'
                    })
                ], style={'margin-bottom': '20px'})
            ], style={'margin-bottom': '30px'}),

            # Overfitting Warning
            *overfitting_warning,

            # Confusion Matrix
            html.Div([
                html.H4("🔢 Confusion Matrix", style={
                        'color': '#495057', 'margin-bottom': '15px'}),
                dcc.Graph(figure=confusion_fig, config={
                          'displayModeBar': False})
            ], style={
                'background-color': '#ffffff',
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 2px 10px rgba(0,0,0,0.05)',
                'margin-bottom': '30px'
            }),

            # Per-Class Performance Metrics
            html.Div([
                html.H4("📈 Per-Class Performance Metrics",
                        style={'color': '#495057', 'margin-bottom': '15px'}),
                html.Table([
                    html.Thead(html.Tr([
                        html.Th("Activity", style={
                                'text-align': 'left', 'padding': '12px', 'background-color': '#2E86AB', 'color': 'white'}),
                        html.Th("Precision", style={
                                'text-align': 'center', 'padding': '12px', 'background-color': '#2E86AB', 'color': 'white'}),
                        html.Th("Recall", style={
                                'text-align': 'center', 'padding': '12px', 'background-color': '#2E86AB', 'color': 'white'}),
                        html.Th("F1-Score", style={'text-align': 'center', 'padding': '12px',
                                'background-color': '#2E86AB', 'color': 'white'}),
                        html.Th("Support", style={
                                'text-align': 'center', 'padding': '12px', 'background-color': '#2E86AB', 'color': 'white'})
                    ])),
                    html.Tbody(per_class_rows)
                ], style={
                    'width': '100%',
                    'border-collapse': 'collapse',
                    'box-shadow': '0 2px 10px rgba(0,0,0,0.05)'
                })
            ], style={
                'background-color': '#ffffff',
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 2px 10px rgba(0,0,0,0.05)',
                'margin-bottom': '30px'
            }),

            # Training Configuration
            html.Div([
                html.H4("⚙️ Training Configuration", style={
                        'color': '#495057', 'margin-bottom': '15px'}),
                html.Div([
                    html.Div([
                        html.P([html.Strong("Model Type: "), model_info.get(
                            'model_type', 'Unknown').replace('_', ' ').title()]),
                        html.P([html.Strong("Training Samples: "), str(
                            model_info.get('training_samples', 0))]),
                        html.P([html.Strong("Features: "), str(
                            model_info.get('features', 0))]),
                        html.P([html.Strong("Classes: "), str(
                            model_info.get('classes', 0))])
                    ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    html.Div([
                        html.P([html.Strong("Training Time: "),
                               f"{model_info.get('training_time', 0):.2f}s"]),
                        html.P([html.Strong("Test Samples: "), str(
                            model_info.get('test_samples', 0))]),
                        html.P([html.Strong("Validation Samples: "),
                               str(model_info.get('val_samples', 0))]),
                        html.P([html.Strong("Timestamp: "),
                               model_info.get('timestamp', 'Unknown')])
                    ], style={'width': '45%', 'display': 'inline-block', 'vertical-align': 'top', 'margin-left': '5%'})
                ]),
                html.Div([
                    html.H5("Training Details:", style={
                            'color': '#495057', 'margin-top': '15px', 'margin-bottom': '10px'}),
                    html.Ul(training_details if training_details else [
                            html.Li("Standard training (no special configurations)")])
                ]) if training_details or True else None
            ], style={
                'background-color': '#f8f9fa',
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0 2px 10px rgba(0,0,0,0.05)'
            })
        ])

    @app.callback(
        Output('confirm-remove-model', 'displayed'),
        Output('confirm-remove-model', 'message'),
        Input('remove-model-btn', 'n_clicks'),
        State('trained-model-selector', 'value'),
        prevent_initial_call=True
    )
    def show_remove_confirmation(n_clicks, model_filename):
        """Show confirmation dialog for model removal."""
        if n_clicks and model_filename:
            message = f"Are you sure you want to remove the model '{model_filename}'?\n\nThis action will:\n• Delete the model file from storage\n• Remove it from the trained models list\n• Delete any generated deployment code\n\nThis action cannot be undone!"
            return True, message
        return False, ""

    @app.callback(
        [Output('remove-model-alert', 'children'),
         Output('remove-model-alert', 'style'),
         Output('trained-model-selector', 'value', allow_duplicate=True),
         Output('trained-model-selector', 'options', allow_duplicate=True)],
        Input('confirm-remove-model', 'submit_n_clicks'),
        [State('trained-model-selector', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def remove_trained_model(submit_n_clicks, model_filename, base_dir):
        """Remove a trained model and update the UI."""
        if not submit_n_clicks or not model_filename:
            return no_update, no_update, no_update

        try:
            # Remove model file
            model_path = get_model_path(model_filename, base_dir)
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.debug(f"DEBUG: Removed model file: {model_path}")

            # Update metadata
            model_metadata_file = get_models_metadata_path(base_dir)
            if os.path.exists(model_metadata_file):
                with open(model_metadata_file, 'r') as f:
                    models_metadata = json.load(f)

                # Remove model from metadata
                if model_filename in models_metadata:
                    del models_metadata[model_filename]

                    # Save updated metadata
                    with open(model_metadata_file, 'w') as f:
                        json.dump(models_metadata, f, indent=2)
                    print(
                        f"DEBUG: Removed model from metadata: {model_filename}")

            # Remove any generated deployment code for this model
            try:
                # Extract model type from filename or metadata for cleanup
                model_type = 'unknown'
                if 'random_forest' in model_filename:
                    model_type = 'random_forest'
                elif 'neural_network' in model_filename:
                    model_type = 'neural_network'
                elif 'svm' in model_filename:
                    model_type = 'svm'

                # Remove generated code folders for this model
                generated_dir = "generated"
                model_folder = f"{model_type}_models"
                full_path = os.path.join(generated_dir, model_folder)

                if os.path.exists(full_path):
                    import shutil
                    shutil.rmtree(full_path)
                    logger.debug(f"DEBUG: Removed generated code folder: {full_path}")
            except Exception as e:
                logger.debug(f"DEBUG: Error removing generated code: {e}")

            # Update dropdown options
            updated_options = []
            if os.path.exists(model_metadata_file):
                with open(model_metadata_file, 'r') as f:
                    models_metadata = json.load(f)

                for filename, info in models_metadata.items():
                    model_type = info.get('model_type', 'unknown')
                    timestamp = info.get('timestamp', '')
                    accuracy = info.get('test_accuracy', 0)

                    label = f"📊 {model_type.replace('_', ' ').title()} - {timestamp} (Acc: {accuracy:.3f})"
                    updated_options.append({'label': label, 'value': filename})

            # Success message
            success_alert = html.Div([
                html.H5("✅ Model Removed Successfully!", style={
                        'color': 'green', 'margin-bottom': '10px'}),
                html.P(f"Model '{model_filename}' has been permanently removed.", style={
                       'margin-bottom': '8px'}),
                html.P("• Model file deleted from storage", style={
                       'margin': '2px 0', 'font-size': '14px'}),
                html.P("• Removed from trained models list", style={
                       'margin': '2px 0', 'font-size': '14px'}),
                html.P("• Generated deployment code cleaned up",
                       style={'margin': '2px 0', 'font-size': '14px'})
            ], style={
                'background-color': '#d4edda',
                'border': '1px solid #c3e6cb',
                'border-radius': '5px',
                'padding': '15px',
                'margin-bottom': '20px'
            })

            return success_alert, {'margin-bottom': '20px', 'display': 'block'}, None, updated_options

        except Exception as e:
            # Error message
            error_alert = html.Div([
                html.H5("❌ Error Removing Model", style={
                        'color': 'red', 'margin-bottom': '10px'}),
                html.P(f"Failed to remove model '{model_filename}'", style={
                       'margin-bottom': '8px'}),
                html.P(f"Error: {str(e)}", style={
                       'font-family': 'monospace', 'background-color': '#f8f9fa', 'padding': '8px'})
            ], style={
                'background-color': '#f8d7da',
                'border': '1px solid #f5c6cb',
                'border-radius': '5px',
                'padding': '15px',
                'margin-bottom': '20px'
            })

            return error_alert, {'margin-bottom': '20px', 'display': 'block'}, no_update, no_update

    def remove_model_metadata(model_filename, base_dir=None):
        """Remove model from metadata file."""
        try:
            model_metadata_file = get_models_metadata_path(base_dir)

            if os.path.exists(model_metadata_file):
                with open(model_metadata_file, 'r') as f:
                    models_metadata = json.load(f)

                if model_filename in models_metadata:
                    del models_metadata[model_filename]

                    with open(model_metadata_file, 'w') as f:
                        json.dump(models_metadata, f, indent=2)

                    return True
            return False
        except Exception as e:
            print(f"Error removing model metadata: {e}")
            return False
    # Clear remove model alert when model selection changes

    @app.callback(
        Output('remove-model-alert', 'style', allow_duplicate=True),
        Input('model-dropdown', 'value'),
        prevent_initial_call=True
    )
    def clear_remove_model_alert(model_value):
        """Clear the remove model alert when model selection changes."""
        return {'margin-bottom': '20px', 'display': 'none'}

    @app.callback(
        Output('deployment-output', 'children'),
        [Input('training-generate-code-btn', 'n_clicks'),
         Input('resource-analysis-btn', 'n_clicks')],
        [State('trained-model-selector', 'value'),
         State('deployment-platform', 'value'),
         State('optimization-level', 'value'),
         State('working-directory-store', 'data')],
        prevent_initial_call=True
    )
    def handle_deployment_actions(generate_clicks, resource_clicks, model_filename, platform, optimization, base_dir):
        """Handle deployment actions including code generation and resource analysis."""
        print(
            f"DEBUG: Deployment callback triggered - clicks: {generate_clicks}, {resource_clicks}")
        print(
            f"DEBUG: Model: {model_filename}, Platform: {platform}, Optimization: {optimization}")

        if not ctx.triggered:
            logger.debug("DEBUG: No context triggered")
            return no_update

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        logger.debug(f"DEBUG: Button clicked: {button_id}")

        if not model_filename:
            logger.debug("DEBUG: No model selected")
            error_msg = html.Div([
                html.H4("⚠ Please select a trained model first.",
                        style={'color': 'orange'})
            ])
            return error_msg

        if not platform:
            platform = 'seeed_xiao'  # Default platform

        if not optimization:
            optimization = 'balanced'  # Default optimization

        try:
            # Load model and metadata
            model_path = get_model_path(model_filename, base_dir)
            if not os.path.exists(model_path):
                logger.debug(f"DEBUG: Model path not found: {model_path}")
                error_msg = html.Div([
                    html.H4("❌ Model file not found.", style={'color': 'red'}),
                    html.P(f"Looking for: {model_path}", style={
                           'font-size': '12px', 'color': '#666'})
                ])
                return error_msg

            logger.debug(f"DEBUG: Loading model from: {model_path}")
            model = EdgeMLModel.load_model(model_path)

            # Get model metadata
            model_metadata_file = get_models_metadata_path()
            model_info = {}
            if os.path.exists(model_metadata_file):
                with open(model_metadata_file, 'r') as f:
                    models_metadata = json.load(f)
                    model_info = models_metadata.get(model_filename, {})

            # Get feature names - try from model first, then from training metadata
            feature_names = model.feature_names
            if not feature_names:
                logger.debug("DEBUG: Feature names not in model, checking training metadata...")
                # Try to get from any training metadata file
                training_dir = os.path.join(PERSISTENT_DIR, 'training')
                metadata_files = glob.glob(
                    os.path.join(training_dir, '*_metadata.json'))
                if metadata_files:
                    with open(metadata_files[0], 'r') as f:
                        training_metadata = json.load(f)
                        feature_names = training_metadata.get(
                            'feature_names', [])
                        print(
                            f"DEBUG: Found {len(feature_names)} feature names from training metadata")

            if not feature_names:
                logger.debug("DEBUG: Still no feature names, loading from training file...")
                # Last resort: load from FE-matched training CSV file
                train_files = _get_fe_train_files(training_dir)
                if train_files:
                    temp_df = pd.read_csv(train_files[0])
                    feature_names = [
                        col for col in temp_df.columns if col != 'label']
                    print(
                        f"DEBUG: Extracted {len(feature_names)} feature names from CSV")

            # Prepare model data for code generation
            model_data = {
                'model_type': model.model_type,
                'feature_names': feature_names or [],
                'classes': list(model.label_encoder.classes_) if model.label_encoder else ['activity_1', 'activity_2'],
                'model_params': model.model_params,
                'performance_metrics': model.performance_metrics,
                'model_object': model  # Pass the actual model object for parameter extraction
            }

            if button_id == 'training-generate-code-btn':
                return generate_deployment_code_display(model_data, platform, optimization, model_filename)
            elif button_id == 'resource-analysis-btn':
                return generate_resource_analysis_display(model_data, model_info, model_filename)

            return no_update

        except Exception as e:
            logger.debug(f"DEBUG: Exception in deployment callback: {str(e)}")
            traceback.print_exc()
            error_msg = html.Div([
                html.H4("❌ Deployment failed", style={'color': 'red'}),
                html.P(f"Error: {str(e)}")
            ])
            return error_msg

    def generate_deployment_code_display(model_data, platform, optimization, model_filename):
        """Generate and display deployment code."""
        logger.debug(f"DEBUG: Starting code generation for {model_filename}")
        logger.debug(f"DEBUG: Platform: {platform}, Optimization: {optimization}")
        logger.debug(f"DEBUG: Model data keys: {list(model_data.keys())}")

        try:
            # Generate code for the selected platform and save to organized folders
            output_dir = "generated"  # Base directory for organized structure
            saved_files = generate_and_save_deployment_code(
                model_data['model_type'], model_data, platform, output_dir, optimization)

            print(
                f"DEBUG: Successfully generated and saved {len(saved_files)} code files")
            logger.debug(f"DEBUG: Saved files: {list(saved_files.keys())}")

            # Also generate in-memory for display purposes
            generated_code = generate_deployment_code(
                model_data['model_type'], model_data, platform, optimization)

            # Create deployment summary with organized folder info
            deployment_summary = html.Div([
                html.H4("✅ Deployment Code Generated!",
                        style={'color': 'green'}),
                html.Hr(),
                html.H5("📋 Deployment Configuration:"),
                html.Ul([
                    html.Li(f"Model: {model_filename}"),
                    html.Li(
                        f"Target Platform: {platform.replace('_', ' ').title()}"),
                    html.Li(f"Optimization: {optimization.title()}"),
                    html.Li(f"Generated Files: {len(generated_code)} files"),
                ]),
                html.H5("� Organized File Structure:"),
                html.Div([
                    html.P(f"✅ Files saved to organized folders:", style={
                           'font-weight': 'bold', 'color': '#2E86AB'}),
                    html.Ul([
                        html.Li(f"📂 {os.path.relpath(filepath, output_dir)}",
                                style={'font-family': 'monospace', 'background-color': '#f8f9fa', 'padding': '2px 5px'})
                        for filepath in saved_files.keys()
                    ], style={'margin-left': '20px'})
                ], style={'background-color': '#e8f5e8', 'padding': '15px', 'border-radius': '5px', 'margin': '10px 0'}),
                html.H5("�🚀 Quick Download:"),
                html.Div([
                    dcc.Download(id="download-all-files-component"),
                    html.Button(
                        f"📦 Download All Files ({len(generated_code)} files)",
                        id="download-all-files-btn",
                        style={
                            'background-color': '#007bff',
                            'border': 'none',
                            'padding': '12px 20px',
                            'font-size': '14px',
                            'border-radius': '6px',
                            'color': 'white',
                            'cursor': 'pointer',
                            'margin-bottom': '20px',
                            'font-weight': 'bold'
                        }
                    )
                ]),
                html.H5("🎯 Next Steps:"),
                html.Ol([
                    html.Li(
                        "Check the organized folders in your project directory"),
                    html.Li("Set up your target hardware platform"),
                    html.Li("Upload the code to your device"),
                    html.Li("Test the real-time activity recognition")
                ]),
                html.Hr(),
                html.H4("📁 Generated Code Files:"),
                html.Div([
                    create_code_file_display(filename, code_content)
                    for filename, code_content in generated_code.items()
                ])
            ])

            logger.debug("DEBUG: Created deployment summary with organized folder structure")
            return deployment_summary

        except Exception as e:
            logger.debug(f"DEBUG: Error in code generation: {str(e)}")
            traceback.print_exc()
            error_msg = html.Div([
                html.H4("❌ Code generation failed", style={'color': 'red'}),
                html.P(f"Error: {str(e)}")
            ])
            return error_msg

    def generate_resource_analysis_display(model_data, model_info, model_filename):
        """Generate and display resource analysis."""
        try:
            # Analyze resource requirements
            resource_analysis = analyze_resource_requirements(model_data)

            # Create resource analysis display
            analysis_summary = html.Div([
                html.H4("📊 Resource Analysis Completed!",
                        style={'color': 'green'}),
                html.Hr(),
                html.H5("💾 Memory Requirements:"),
                html.Ul([
                    html.Li(
                        f"Feature Memory: {resource_analysis['memory_requirements']['feature_memory_bytes']} bytes"),
                    html.Li(
                        f"Buffer Memory: {resource_analysis['memory_requirements']['buffer_memory_bytes']} bytes"),
                    html.Li(
                        f"Model Memory: {resource_analysis['memory_requirements']['model_memory_bytes']} bytes"),
                    html.Li(
                        f"Total Memory: {resource_analysis['memory_requirements']['total_memory_kb']} KB"),
                ]),
                html.H5("⚡ Computational Requirements:"),
                html.Ul([
                    html.Li(
                        f"Operations per Prediction: {resource_analysis['computational_requirements']['operations_per_prediction']}"),
                    html.Li(
                        f"Estimated Inference Time: {resource_analysis['computational_requirements']['estimated_inference_time_ms']} ms"),
                    html.Li(
                        f"Power Consumption: {resource_analysis['computational_requirements']['power_consumption_estimate']}"),
                ]),
                html.H5("🎯 Platform Compatibility:"),
                html.Div([
                    create_compatibility_badge(platform, compatible)
                    for platform, compatible in resource_analysis['platform_compatibility'].items()
                ])
            ])

            return analysis_summary

        except Exception as e:
            logger.debug(f"DEBUG: Error in resource analysis: {str(e)}")
            error_msg = html.Div([
                html.H4("❌ Resource analysis failed", style={'color': 'red'}),
                html.P(f"Error: {str(e)}")
            ])
            return error_msg

    def create_code_file_display(filename, code_content):
        """Create a code file display component for UI viewing only (no additional saving)."""
        # Note: Files are already saved to organized directories by generate_and_save_deployment_code
        # This function is purely for UI display purposes

        return html.Div([
            html.H6(f"📄 {filename}", style={
                    'color': '#2E86AB', 'margin-bottom': '10px'}),
            html.Pre(
                html.Code(
                    code_content[:1000] + "\n\n... (truncated for display)" if len(
                        code_content) > 1000 else code_content,
                    style={
                        'background-color': '#f8f9fa',
                        'padding': '15px',
                        'border-radius': '5px',
                        'font-family': 'Courier New, monospace',
                        'font-size': '12px',
                        'white-space': 'pre-wrap',
                        'overflow-x': 'auto'
                    }
                )
            ),
            html.P([
                "💡 ",
                html.Strong("Note: "),
                f"File saved to organized directory structure (see above for path)"
            ], style={
                'color': '#6c757d',
                'font-size': '12px',
                'margin-top': '10px',
                'padding': '8px',
                'background-color': '#f1f3f4',
                'border-radius': '4px'
            })
        ], style={'margin-bottom': '20px'})

    def create_compatibility_badge(platform, compatible):
        """Create a platform compatibility badge."""
        color = '#28a745' if compatible else '#dc3545'
        icon = '✅' if compatible else '❌'

        return html.Span([
            f"{icon} {platform.replace('_', ' ').title()}"
        ], style={
            'background-color': color,
            'color': 'white',
            'padding': '4px 8px',
            'border-radius': '4px',
            'margin': '2px',
            'display': 'inline-block',
            'font-size': '12px'
        })

    @app.callback(
        Output('session-stats', 'children'),
        Input('tabs', 'value'),
        State('working-directory-store', 'data'),
        prevent_initial_call=True
    )
    def update_training_statistics(tab, base_dir):
        """Update training session statistics display."""
        if tab != 'tab-4':
            return no_update

        try:
            # Load training session statistics
            stats = get_training_session_stats(base_dir)

            return html.Div([
                html.Div([
                    html.H6("Training Status", style={
                            'color': '#666', 'margin-bottom': '5px'}),
                    html.P(stats['status'], style={
                        'color': stats['status_color'],
                        'font-size': '18px',
                        'font-weight': 'bold'
                    })
                ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'}),

                html.Div([
                    html.H6("Models Trained", style={
                            'color': '#666', 'margin-bottom': '5px'}),
                    html.P(str(stats['models_trained']), style={
                        'color': '#28a745',
                        'font-size': '18px',
                        'font-weight': 'bold'
                    })
                ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'}),

                html.Div([
                    html.H6("Best Accuracy", style={
                            'color': '#666', 'margin-bottom': '5px'}),
                    html.P(f"{stats['best_accuracy']:.3f}" if stats['best_accuracy'] > 0 else "N/A", style={
                        'color': '#007bff',
                        'font-size': '18px',
                        'font-weight': 'bold'
                    })
                ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'}),

                html.Div([
                    html.H6("Total Time", style={
                            'color': '#666', 'margin-bottom': '5px'}),
                    html.P(stats['total_time'], style={
                        'color': '#6f42c1',
                        'font-size': '18px',
                        'font-weight': 'bold'
                    })
                ], style={'text-align': 'center', 'width': '25%', 'display': 'inline-block'})
            ], style={'padding': '20px'})

        except Exception as e:
            print(f"Error updating training statistics: {e}")
            return html.P("Error loading statistics", style={'text-align': 'center', 'color': 'red'})

    def get_training_session_stats(base_dir=None):
        """Get current training session statistics."""
        if not base_dir:
            base_dir = PERSISTENT_DIR

        try:
            # Load trained models metadata
            models_dir = os.path.join(base_dir, 'models')
            model_metadata_file = os.path.join(
                models_dir, 'trained_models.json')

            if not os.path.exists(model_metadata_file):
                return {
                    'status': 'No session active',
                    'status_color': '#6c757d',
                    'models_trained': 0,
                    'best_accuracy': 0,
                    'total_time': '00:00:00'
                }

            with open(model_metadata_file, 'r') as f:
                models_metadata = json.load(f)

            if not models_metadata:
                return {
                    'status': 'No models trained',
                    'status_color': '#6c757d',
                    'models_trained': 0,
                    'best_accuracy': 0,
                    'total_time': '00:00:00'
                }

            # Calculate statistics
            models_count = len(models_metadata)
            best_accuracy = max(
                [model_info.get('test_accuracy', 0)
                 for model_info in models_metadata.values()],
                default=0
            )
            total_training_time = sum(
                [model_info.get('training_time', 0)
                 for model_info in models_metadata.values()]
            )

            # Format total time
            hours = int(total_training_time // 3600)
            minutes = int((total_training_time % 3600) // 60)
            seconds = int(total_training_time % 60)
            total_time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

            # Determine status
            if models_count > 0:
                status = "Active session"
                status_color = "#28a745"
            else:
                status = "No session active"
                status_color = "#6c757d"

            return {
                'status': status,
                'status_color': status_color,
                'models_trained': models_count,
                'best_accuracy': best_accuracy,
                'total_time': total_time_str
            }

        except Exception as e:
            print(f"Error calculating training statistics: {e}")
            return {
                'status': 'Error',
                'status_color': '#dc3545',
                'models_trained': 0,
                'best_accuracy': 0,
                'total_time': '00:00:00'
            }

    # Additional utility functions for training callbacks

    def load_available_models(base_dir=None):
        """Load list of available trained models."""
        model_metadata_file = get_models_metadata_path(base_dir)
        if os.path.exists(model_metadata_file):
            with open(model_metadata_file, 'r') as f:
                return json.load(f)
        return {}

    # Dynamic download callbacks for generated files

    @app.callback(
        Output('download-har_model-h-component', 'data'),
        Input('download-har_model-h-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_har_model_h(n_clicks):
        """Download har_model.h file."""
        if n_clicks:
            file_path = os.path.join(os.path.dirname(
                PERSISTENT_DIR), 'deployment', 'har_model.h')
            if os.path.exists(file_path):
                return dcc.send_file(file_path)
        return no_update

    @app.callback(
        Output('download-har_model-cpp-component', 'data'),
        Input('download-har_model-cpp-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_har_model_cpp(n_clicks):
        """Download har_model.cpp file."""
        if n_clicks:
            file_path = os.path.join(os.path.dirname(
                PERSISTENT_DIR), 'deployment', 'har_model.cpp')
            if os.path.exists(file_path):
                return dcc.send_file(file_path)
        return no_update

    @app.callback(
        Output('download-har_example-ino-component', 'data'),
        Input('download-har_example-ino-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_har_example_ino(n_clicks):
        """Download har_example.ino file."""
        if n_clicks:
            file_path = os.path.join(os.path.dirname(
                PERSISTENT_DIR), 'deployment', 'har_example.ino')
            if os.path.exists(file_path):
                return dcc.send_file(file_path)
        return no_update

    @app.callback(
        Output('download-har_model_cortex-c-component', 'data'),
        Input('download-har_model_cortex-c-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_har_model_cortex_c(n_clicks):
        """Download har_model_cortex.c file."""
        if n_clicks:
            file_path = os.path.join(os.path.dirname(
                PERSISTENT_DIR), 'deployment', 'har_model_cortex.c')
            if os.path.exists(file_path):
                return dcc.send_file(file_path)
        return no_update

    @app.callback(
        Output('download-all-files-component', 'data'),
        Input('download-all-files-btn', 'n_clicks'),
        prevent_initial_call=True
    )
    def download_all_deployment_files(n_clicks):
        """Download all generated deployment files as a ZIP."""
        if n_clicks:
            deployment_dir = os.path.join(
                os.path.dirname(PERSISTENT_DIR), 'deployment')

            if os.path.exists(deployment_dir):
                # Create a temporary ZIP file
                temp_dir = tempfile.mkdtemp()
                zip_path = os.path.join(temp_dir, 'har_deployment_code.zip')

                with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for filename in os.listdir(deployment_dir):
                        file_path = os.path.join(deployment_dir, filename)
                        if os.path.isfile(file_path) and not filename.startswith('.'):
                            zipf.write(file_path, filename)

                return dcc.send_file(zip_path, filename='har_deployment_code.zip')

        return no_update
