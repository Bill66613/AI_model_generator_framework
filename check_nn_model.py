"""Check Neural Network Model Hyperparameters"""
import joblib
import json

model_path = 'persistent_data/neural_network_optimized_20251207_224633.joblib'
model = joblib.load(model_path)

print("="*60)
print("NEURAL NETWORK MODEL ANALYSIS")
print("="*60)

print(f"\n📊 Model Type: {model.get('model_type', 'N/A')}")

# Model architecture
nn_model = model['model']
print(f"\n🏗️ Architecture:")
print(f"   Hidden Layers: {nn_model.hidden_layer_sizes}")
print(f"   Total Layers: {len(nn_model.hidden_layer_sizes) + 2} (input + hidden + output)")

# Hyperparameters
print(f"\n⚙️ Hyperparameters:")
print(f"   Activation: {nn_model.activation}")
print(f"   Solver: {nn_model.solver}")
print(f"   Alpha (L2 reg): {nn_model.alpha}")
print(f"   Learning Rate: {nn_model.learning_rate_init}")
print(f"   Max Iterations: {nn_model.max_iter}")

# Model params stored
params = model.get('model_params', {})
print(f"\n📝 Stored Params:")
for key, val in params.items():
    print(f"   {key}: {val}")

# Performance metrics
print(f"\n📈 Performance Metrics:")
perf = model.get('performance_metrics', {})
for key, val in perf.items():
    if key not in ['confusion_matrix', 'classification_report', 'predictions', 'actual_labels', 'prediction_probabilities']:
        print(f"   {key}: {val}")

# Check if overfitting
if 'train_accuracy' in perf and 'test_accuracy' in perf:
    train_acc = perf['train_accuracy']
    test_acc = perf['test_accuracy']
    gap = train_acc - test_acc
    print(f"\n⚠️ Overfitting Analysis:")
    print(f"   Training Accuracy: {train_acc:.4f}")
    print(f"   Test Accuracy: {test_acc:.4f}")
    print(f"   Gap: {gap:.4f}")
    if gap > 0.1:
        print(f"   Status: ❌ OVERFITTING (gap > 0.10)")
    elif gap > 0.05:
        print(f"   Status: ⚠️ Slight overfitting (gap > 0.05)")
    else:
        print(f"   Status: ✅ Good generalization")

print("\n" + "="*60)
