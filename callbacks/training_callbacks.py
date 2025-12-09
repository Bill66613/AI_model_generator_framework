"""
Training Module Callbacks for Human Activity Recognition Framework
Handles model training, evaluation, and deployment workflows
"""

import os
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from dash import Input, Output, State, callback, no_update, dcc, html, ctx
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io
import time
import traceback
from datetime import datetime, timedelta
import zipfile
import tempfile

from config.config import *
from utils.model_training import EdgeMLModel, prepare_training_data, create_feature_vector
from deployment import generate_deployment_code, analyze_resource_requirements, generate_and_save_deployment_code


@callback(
    [Output('model-type-selector', 'disabled'),
     Output('start-training-btn', 'disabled'),
     Output('optimize-hyperparams-btn', 'disabled'),
     Output('cross-validate-btn', 'disabled'),
     Output('trained-model-selector', 'options')],
    Input('tabs', 'value')
)
def enable_training_components(tab):
    """Enable training components when training tab is active and load trained models."""
    # Load available trained models using helper function
    model_options = load_trained_model_options()

    if tab == 'tab-3':
        return False, False, False, False, model_options
    return True, True, True, True, model_options


# Utility functions
def save_model_metadata(model_filename, model_info):
    """Save model metadata to the models database."""
    model_metadata_file = os.path.join(PERSISTENT_DIR, "trained_models.json")

    if os.path.exists(model_metadata_file):
        with open(model_metadata_file, 'r') as f:
            models_metadata = json.load(f)
    else:
        models_metadata = {}

    models_metadata[model_filename] = model_info

    with open(model_metadata_file, 'w') as f:
        json.dump(models_metadata, f, indent=2)


def load_trained_model_options():
    """Load available trained model options for dropdown."""
    model_options = []
    try:
        model_metadata_file = os.path.join(PERSISTENT_DIR, "trained_models.json")
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


def create_training_results_display(model_info, evaluation_results, y_test, model_type, training_time):
    """Create comprehensive training results display."""
    return html.Div([
        html.H4("✅ Model Training Completed!", style={'color': 'green'}),
        html.Hr(),
        html.H5("📊 Training Summary:"),
        html.Ul([
            html.Li(f"Model Type: {model_type.replace('_', ' ').title()}"),
            html.Li(f"Training Samples: {model_info['training_samples']}"),
            html.Li(f"Test Samples: {model_info['test_samples']}"),
            html.Li(f"Features: {model_info['features']}"),
            html.Li(f"Activity Classes: {model_info['classes']}"),
            html.Li(f"Training Time: {training_time:.2f} seconds"),
        ]),
        html.H5("🎯 Performance Metrics:"),
        html.Ul([
            html.Li(
                f"Training Accuracy: {model_info.get('train_accuracy', 0):.4f}"),
            html.Li(f"Test Accuracy: {model_info['test_accuracy']:.4f}"),
            html.Li(
                f"Cross-Validation Accuracy: {model_info.get('cv_accuracy', 0):.4f}"),
        ]),
        html.H5("💾 Model Saved:"),
        html.P(f"Model saved as: {os.path.basename(model_info['model_path'])}", style={
            'font-family': 'monospace',
            'background-color': '#f0f0f0',
            'padding': '10px'
        }),
        html.Hr(),
        dcc.Graph(
            figure=create_training_results_visualization(
                evaluation_results, y_test, model_type)
        )
    ])


@callback(
    [Output('training-output', 'children'),
     Output('trained-model-selector', 'options', allow_duplicate=True)],
    [Input('start-training-btn', 'n_clicks'),
     Input('optimize-hyperparams-btn', 'n_clicks'),
     Input('cross-validate-btn', 'n_clicks')],
    [State('model-type-selector', 'value')],
    prevent_initial_call=True
)
def handle_training_actions(train_clicks, optimize_clicks, cv_clicks, model_type):
    """Handle different training actions based on which button was clicked."""
    if not ctx.triggered:
        return no_update, no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if not model_type:
        return (html.Div([
            html.H4("⚠ Please select a model type first.",
                    style={'color': 'orange'})
        ]), no_update)

    try:
        # Load metadata to find processed windows
        if not os.path.exists(METADATA_FILE):
            return (html.Div([
                html.H4("❌ No datasets found.", style={'color': 'red'}),
                html.P("Please upload and preprocess data first.")
            ]), no_update)

        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)

        # Collect all processed window files
        window_files = []
        labels = []

        for dataset_name, dataset_info in metadata.items():
            if 'dragged_samples' in dataset_info:
                dataset_label = dataset_info.get(
                    'label', dataset_name.replace('.csv', ''))
                for sample_file in dataset_info['dragged_samples']:
                    if os.path.exists(sample_file):
                        window_files.append(sample_file)
                        labels.append(dataset_label)

        if not window_files:
            return (html.Div([
                html.H4("❌ No processed windows found.",
                        style={'color': 'red'}),
                html.P("Please preprocess your data and create time windows first.")
            ]), no_update)

        # Prepare training data - use only time-domain features (90 features)
        # Frequency features excluded for training-deployment parity
        X, y = prepare_training_data(window_files, labels, include_frequency=False)

        if X.empty:
            return (html.Div([
                html.H4("❌ Failed to prepare training data.",
                        style={'color': 'red'}),
                html.P("Check your window files.")
            ]), no_update)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create model
        model = EdgeMLModel(model_type)

        if button_id == 'start-training-btn':
            return perform_basic_training(model, X_train, X_test, y_train, y_test, model_type)
        elif button_id == 'optimize-hyperparams-btn':
            return perform_hyperparameter_optimization(model, X_train, X_test, y_train, y_test, model_type)
        elif button_id == 'cross-validate-btn':
            return perform_cross_validation(model, X_train, y_train, model_type)

    except Exception as e:
        return (html.Div([
            html.H4("❌ Training failed", style={'color': 'red'}),
            html.P(f"Error: {str(e)}")
        ]), no_update)


def perform_basic_training(model, X_train, X_test, y_train, y_test, model_type):
    """Perform basic model training."""
    start_time = time.time()

    # Training
    training_results = model.train(X_train, y_train, use_cross_validation=True)

    # Evaluation
    evaluation_results = model.evaluate(X_test, y_test)

    training_time = time.time() - start_time

    # Save trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_har_model_{timestamp}.joblib"
    model_path = os.path.join(PERSISTENT_DIR, model_filename)
    model.save_model(model_path)

    # Update metadata with model info
    model_info = {
        'model_path': model_path,
        'model_type': model_type,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X_train.columns),
        'classes': len(set(y_train)),
        'train_accuracy': training_results['train_accuracy'],
        'test_accuracy': evaluation_results['test_accuracy'],
        'cv_accuracy': training_results.get('cv_mean_accuracy', 0),
        'training_time': training_time,
        'timestamp': timestamp
    }

    # Store model metadata
    save_model_metadata(model_filename, model_info)

    # Create results summary
    training_output = create_training_results_display(model_info, evaluation_results, y_test, model_type, training_time)
    updated_options = load_trained_model_options()
    return training_output, updated_options


def perform_hyperparameter_optimization(model, X_train, X_test, y_train, y_test, model_type):
    """Perform hyperparameter optimization."""
    start_time = time.time()

    # Hyperparameter optimization
    optimization_results = model.optimize_hyperparameters(X_train, y_train)

    # Evaluate optimized model
    evaluation_results = model.evaluate(X_test, y_test)

    training_time = time.time() - start_time

    # Save optimized model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_type}_optimized_{timestamp}.joblib"
    model_path = os.path.join(PERSISTENT_DIR, model_filename)
    model.save_model(model_path)

    # Update metadata
    model_info = {
        'model_path': model_path,
        'model_type': model_type,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': len(X_train.columns),
        'classes': len(set(y_train)),
        'test_accuracy': evaluation_results['test_accuracy'],
        'best_params': optimization_results['best_params'],
        'optimization_score': optimization_results['best_score'],
        'training_time': training_time,
        'timestamp': timestamp,
        'optimized': True
    }

    save_model_metadata(model_filename, model_info)

    optimization_output = html.Div([
        html.H4("✅ Hyperparameter Optimization Completed!",
                style={'color': 'green'}),
        html.Hr(),
        html.H5("🎯 Optimization Results:"),
        html.Ul([
            html.Li(f"Model Type: {model_type.replace('_', ' ').title()}"),
            html.Li(
                f"Best CV Score: {optimization_results['best_score']:.4f}"),
            html.Li(
                f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}"),
            html.Li(f"Training Time: {training_time:.2f} seconds"),
        ]),
        html.H5("⚙️ Best Parameters:"),
        html.Ul([
            html.Li(f"{param}: {value}")
            for param, value in optimization_results['best_params'].items()
        ]),
        html.P(f"Optimized model saved as: {model_filename}", style={
            'font-family': 'monospace',
            'background-color': '#f0f0f0',
            'padding': '10px'
        }),
        dcc.Graph(
            figure=create_training_results_visualization(
                evaluation_results, y_test, model_type)
        )
    ])
    updated_options = load_trained_model_options()
    return optimization_output, updated_options


def perform_cross_validation(model, X_train, y_train, model_type):
    """Perform cross-validation analysis."""
    start_time = time.time()

    # Perform training with cross-validation
    training_results = model.train(X_train, y_train, use_cross_validation=True)

    training_time = time.time() - start_time

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
            html.Li(f"Training Time: {training_time:.2f} seconds"),
        ]),
        html.Div([
            dcc.Graph(
                figure=create_cv_visualization(cv_results, model_type)
            )
        ])
    ])
    # Cross-validation doesn't save a model, so return no_update for model options
    return cv_output, no_update


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


def create_training_results_visualization(evaluation_results, y_test, model_type):
    """Create comprehensive visualization of training results."""

    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Confusion Matrix', 'Class Distribution',
                        'Feature Importance', 'Model Performance'),
        specs=[[{'type': 'heatmap'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'indicator'}]]
    )

    # 1. Confusion Matrix
    conf_matrix = evaluation_results['confusion_matrix']
    class_names = sorted(set(y_test))

    fig.add_trace(
        go.Heatmap(
            z=conf_matrix,
            x=class_names,
            y=class_names,
            colorscale='Blues',
            showscale=True,
            text=conf_matrix,
            texttemplate="%{text}",
            textfont={"size": 12}
        ),
        row=1, col=1
    )

    # 2. Class Distribution
    class_counts = pd.Series(y_test).value_counts()
    fig.add_trace(
        go.Bar(
            x=class_counts.index,
            y=class_counts.values,
            name='Test Set Distribution',
            marker_color='skyblue'
        ),
        row=1, col=2
    )

    # 3. Performance Indicator
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
        row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Model Training Results - {model_type.replace('_', ' ').title()}",
        height=800,
        showlegend=False
    )

    # Update subplot titles
    fig.update_xaxes(title_text="Predicted", row=1, col=1)
    fig.update_yaxes(title_text="Actual", row=1, col=1)
    fig.update_xaxes(title_text="Activity Classes", row=1, col=2)
    fig.update_yaxes(title_text="Sample Count", row=1, col=2)

    return fig


@callback(
    Output('model-performance-graph', 'figure'),
    [Input('evaluate-model-btn', 'n_clicks'),
     Input('feature-importance-btn', 'n_clicks')],
    State('trained-model-selector', 'value'),
    prevent_initial_call=True
)
def evaluate_trained_model(eval_clicks, feature_clicks, model_filename):
    """Evaluate a trained model and show detailed performance metrics."""
    if not model_filename:
        return {}

    if not ctx.triggered:
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    try:
        model_path = os.path.join(PERSISTENT_DIR, model_filename)
        if not os.path.exists(model_path):
            return {}

        # Load trained model
        model = EdgeMLModel.load_model(model_path)

        if button_id == 'evaluate-model-btn':
            return create_model_evaluation_plot(model, model_filename)
        elif button_id == 'feature-importance-btn':
            return create_feature_importance_plot(model, model_filename)

        return {}

    except Exception as e:
        print(f"Error evaluating model: {e}")
        return {}


def create_model_evaluation_plot(model, model_filename):
    """Create comprehensive model evaluation visualization."""
    # Load model metadata
    model_metadata_file = os.path.join(PERSISTENT_DIR, "trained_models.json")
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


def create_feature_importance_plot(model, model_filename):
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


@callback(
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


@callback(
    [Output('remove-model-alert', 'children'),
     Output('remove-model-alert', 'style'),
     Output('trained-model-selector', 'value', allow_duplicate=True),
     Output('trained-model-selector', 'options', allow_duplicate=True)],
    Input('confirm-remove-model', 'submit_n_clicks'),
    State('trained-model-selector', 'value'),
    prevent_initial_call=True
)
def remove_trained_model(submit_n_clicks, model_filename):
    """Remove a trained model and update the UI."""
    if not submit_n_clicks or not model_filename:
        return no_update, no_update, no_update

    try:
        # Remove model file
        model_path = os.path.join(PERSISTENT_DIR, model_filename)
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"DEBUG: Removed model file: {model_path}")

        # Update metadata
        model_metadata_file = os.path.join(
            PERSISTENT_DIR, "trained_models.json")
        if os.path.exists(model_metadata_file):
            with open(model_metadata_file, 'r') as f:
                models_metadata = json.load(f)

            # Remove model from metadata
            if model_filename in models_metadata:
                del models_metadata[model_filename]

                # Save updated metadata
                with open(model_metadata_file, 'w') as f:
                    json.dump(models_metadata, f, indent=2)
                print(f"DEBUG: Removed model from metadata: {model_filename}")

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
                print(f"DEBUG: Removed generated code folder: {full_path}")
        except Exception as e:
            print(f"DEBUG: Error removing generated code: {e}")

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


def remove_model_metadata(model_filename):
    """Remove model from metadata file."""
    try:
        model_metadata_file = os.path.join(
            PERSISTENT_DIR, "trained_models.json")

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


@callback(
    Output('remove-model-alert', 'style', allow_duplicate=True),
    Input('model-dropdown', 'value'),
    prevent_initial_call=True
)
def clear_remove_model_alert(model_value):
    """Clear the remove model alert when model selection changes."""
    return {'margin-bottom': '20px', 'display': 'none'}


@callback(
    Output('deployment-output', 'children'),
    [Input('generate-code-btn', 'n_clicks'),
     Input('resource-analysis-btn', 'n_clicks')],
    [State('trained-model-selector', 'value'),
     State('deployment-platform', 'value'),
     State('optimization-level', 'value')],
    prevent_initial_call=True
)
def handle_deployment_actions(generate_clicks, resource_clicks, model_filename, platform, optimization):
    """Handle deployment actions including code generation and resource analysis."""
    print(
        f"DEBUG: Deployment callback triggered - clicks: {generate_clicks}, {resource_clicks}")
    print(
        f"DEBUG: Model: {model_filename}, Platform: {platform}, Optimization: {optimization}")

    if not ctx.triggered:
        print("DEBUG: No context triggered")
        return no_update

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    print(f"DEBUG: Button clicked: {button_id}")

    if not model_filename:
        print("DEBUG: No model selected")
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
        model_path = os.path.join(PERSISTENT_DIR, model_filename)
        if not os.path.exists(model_path):
            error_msg = html.Div([
                html.H4("❌ Model file not found.", style={'color': 'red'})
            ])
            return error_msg, no_update

        model = EdgeMLModel.load_model(model_path)

        # Get model metadata
        model_metadata_file = os.path.join(
            PERSISTENT_DIR, "trained_models.json")
        model_info = {}
        if os.path.exists(model_metadata_file):
            with open(model_metadata_file, 'r') as f:
                models_metadata = json.load(f)
                model_info = models_metadata.get(model_filename, {})

        # Prepare model data for code generation
        model_data = {
            'model_type': model.model_type,
            'feature_names': model.feature_names or [],
            'classes': list(model.label_encoder.classes_) if model.label_encoder else ['activity_1', 'activity_2'],
            'model_params': model.model_params,
            'performance_metrics': model.performance_metrics,
            'model_object': model  # Pass the actual model object for parameter extraction
        }

        if button_id == 'generate-code-btn':
            return generate_deployment_code_display(model_data, platform, optimization, model_filename)
        elif button_id == 'resource-analysis-btn':
            return generate_resource_analysis_display(model_data, model_info, model_filename)

        return no_update

    except Exception as e:
        print(f"DEBUG: Exception in deployment callback: {str(e)}")
        traceback.print_exc()
        error_msg = html.Div([
            html.H4("❌ Deployment failed", style={'color': 'red'}),
            html.P(f"Error: {str(e)}")
        ])
        return error_msg


def generate_deployment_code_display(model_data, platform, optimization, model_filename):
    """Generate and display deployment code."""
    print(f"DEBUG: Starting code generation for {model_filename}")
    print(f"DEBUG: Platform: {platform}, Optimization: {optimization}")
    print(f"DEBUG: Model data keys: {list(model_data.keys())}")

    try:
        # Generate code for the selected platform and save to organized folders
        output_dir = "generated"  # Base directory for organized structure
        saved_files = generate_and_save_deployment_code(
            model_data['model_type'], model_data, platform, output_dir, optimization)

        print(
            f"DEBUG: Successfully generated and saved {len(saved_files)} code files")
        print(f"DEBUG: Saved files: {list(saved_files.keys())}")

        # Also generate in-memory for display purposes
        generated_code = generate_deployment_code(
            model_data['model_type'], model_data, platform, optimization)

        # Create deployment summary with organized folder info
        deployment_summary = html.Div([
            html.H4("✅ Deployment Code Generated!", style={'color': 'green'}),
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
                html.Li("Check the organized folders in your project directory"),
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

        print("DEBUG: Created deployment summary with organized folder structure")
        return deployment_summary

    except Exception as e:
        print(f"DEBUG: Error in code generation: {str(e)}")
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
        print(f"DEBUG: Error in resource analysis: {str(e)}")
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


@callback(
    Output('session-stats', 'children'),
    Input('tabs', 'value'),
    prevent_initial_call=True
)
def update_training_statistics(tab):
    """Update training session statistics display."""
    if tab != 'tab-3':
        return no_update

    try:
        # Load training session statistics
        stats = get_training_session_stats()

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


def get_training_session_stats():
    """Get current training session statistics."""
    try:
        # Load trained models metadata
        model_metadata_file = os.path.join(
            PERSISTENT_DIR, "trained_models.json")

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
def load_available_models():
    """Load list of available trained models."""
    model_metadata_file = os.path.join(PERSISTENT_DIR, "trained_models.json")
    if os.path.exists(model_metadata_file):
        with open(model_metadata_file, 'r') as f:
            return json.load(f)
    return {}


# Dynamic download callbacks for generated files
@callback(
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


@callback(
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


@callback(
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


@callback(
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


@callback(
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
