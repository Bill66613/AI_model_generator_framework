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
from dash import Input, Output, State, callback, no_update, dcc, html
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import base64
import io

from config.config import *
from utils.model_training import EdgeMLModel, prepare_training_data, create_feature_vector


@callback(
    Output('model-type-selector', 'disabled'),
    Output('start-training-btn', 'disabled'),
    Input('tabs', 'value')
)
def enable_training_components(tab):
    """Enable training components when training tab is active."""
    if tab == 'tab-3':
        return False, False
    return True, True


@callback(
    Output('training-output', 'children'),
    Input('start-training-btn', 'n_clicks'),
    State('model-type-selector', 'value'),
    prevent_initial_call=True
)
def start_model_training(n_clicks, model_type):
    """Start the model training process."""
    if not model_type:
        return "⚠ Please select a model type first."

    try:
        # Load metadata to find processed windows
        if not os.path.exists(METADATA_FILE):
            return "❌ No datasets found. Please upload and preprocess data first."

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
            return "❌ No processed windows found. Please preprocess your data and create time windows first."

        # Prepare training data
        X, y = prepare_training_data(window_files, labels)

        if X.empty:
            return "❌ Failed to prepare training data. Check your window files."

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Create and train model
        model = EdgeMLModel(model_type)

        # Training
        training_results = model.train(
            X_train, y_train, use_cross_validation=True)

        # Evaluation
        evaluation_results = model.evaluate(X_test, y_test)

        # Save trained model
        model_filename = f"{model_type}_har_model.joblib"
        model_path = os.path.join(PERSISTENT_DIR, model_filename)
        model.save_model(model_path)

        # Update metadata with model info
        model_info = {
            'model_path': model_path,
            'model_type': model_type,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': len(X.columns),
            'classes': len(set(y)),
            'train_accuracy': training_results['train_accuracy'],
            'test_accuracy': evaluation_results['test_accuracy'],
            'cv_accuracy': training_results.get('cv_mean_accuracy', 0)
        }

        # Store model metadata
        model_metadata_file = os.path.join(
            PERSISTENT_DIR, "trained_models.json")
        if os.path.exists(model_metadata_file):
            with open(model_metadata_file, 'r') as f:
                models_metadata = json.load(f)
        else:
            models_metadata = {}

        models_metadata[model_filename] = model_info

        with open(model_metadata_file, 'w') as f:
            json.dump(models_metadata, f, indent=2)

        # Create results summary
        results_html = html.Div([
            html.H4("✅ Model Training Completed!", style={'color': 'green'}),
            html.Hr(),
            html.H5("📊 Training Summary:"),
            html.Ul([
                html.Li(f"Model Type: {model_type.replace('_', ' ').title()}"),
                html.Li(f"Training Samples: {len(X_train)}"),
                html.Li(f"Test Samples: {len(X_test)}"),
                html.Li(f"Features: {len(X.columns)}"),
                html.Li(
                    f"Activity Classes: {len(set(y))} ({', '.join(sorted(set(y)))})"),
            ]),
            html.H5("🎯 Performance Metrics:"),
            html.Ul([
                html.Li(
                    f"Training Accuracy: {training_results['train_accuracy']:.4f}"),
                html.Li(
                    f"Test Accuracy: {evaluation_results['test_accuracy']:.4f}"),
                html.Li(
                    f"Cross-Validation Accuracy: {training_results.get('cv_mean_accuracy', 0):.4f} ± {training_results.get('cv_std_accuracy', 0):.4f}"),
            ]),
            html.H5("💾 Model Saved:"),
            html.P(f"Model saved as: {model_filename}", style={
                   'font-family': 'monospace', 'background-color': '#f0f0f0', 'padding': '10px'}),
            html.Hr(),
            dcc.Graph(id='training-results-graph', figure=create_training_results_visualization(
                evaluation_results, y_test, model_type
            ))
        ])

        return results_html

    except Exception as e:
        return f"❌ Training failed: {str(e)}"


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
    Output('model-performance-graph', 'figure', True),
    Input('evaluate-model-btn', 'n_clicks'),
    State('trained-model-selector', 'value'),
    prevent_initial_call=True
)
def evaluate_trained_model(n_clicks, model_filename):
    """Evaluate a trained model and show detailed performance metrics."""
    if not model_filename:
        return {}

    try:
        model_path = os.path.join(PERSISTENT_DIR, model_filename)
        if not os.path.exists(model_path):
            return {}

        # Load trained model
        model = EdgeMLModel.load_model(model_path)

        # Get feature importance if available
        feature_importance = model.get_feature_importance()

        if feature_importance:
            # Create feature importance plot
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())

            # Sort by importance
            sorted_indices = np.argsort(importances)[-20:]  # Top 20 features
            sorted_features = [features[i] for i in sorted_indices]
            sorted_importances = [importances[i] for i in sorted_indices]

            fig = go.Figure(go.Bar(
                x=sorted_importances,
                y=sorted_features,
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

        return {}

    except Exception as e:
        return {}


# Additional utility functions for training callbacks
def load_available_models():
    """Load list of available trained models."""
    model_metadata_file = os.path.join(PERSISTENT_DIR, "trained_models.json")
    if os.path.exists(model_metadata_file):
        with open(model_metadata_file, 'r') as f:
            return json.load(f)
    return {}


def generate_deployment_code(model_filename, target_platform='arduino'):
    """Generate deployment code for edge devices."""
    # This would contain code generation logic for different platforms
    # Implementation depends on specific requirements
    pass
