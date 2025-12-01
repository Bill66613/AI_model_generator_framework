"""Regenerate deployment code with scaling fix"""
from deployment import generate_and_save_deployment_code
from utils.model_training import EdgeMLModel
import joblib

print("="*70)
print("REGENERATING DEPLOYMENT CODE WITH FEATURE SCALING FIX")
print("="*70)

# Load the trained model (90 time-domain features only)
model_file = 'persistent_data/neural_network_time_only_20251126_025232.joblib'
print(f"\n1. Loading model: {model_file}")

m_data = joblib.load(model_file)
model_obj = EdgeMLModel.load_model(model_file)

print(f"   Model type: {model_obj.model_type}")
print(f"   Features: {len(m_data.get('feature_names', []))}")
print(f"   Classes: {list(m_data['label_encoder'].classes_)}")

# Prepare model data
model_data = {
    'model_type': 'neural_network',
    'feature_names': m_data.get('feature_names', []),
    'classes': [str(c) for c in m_data['label_encoder'].classes_],
    'model_object': model_obj
}

print(f"\n2. Generating code for seeed_xiao platform (balanced optimization)...")

# Generate and save
files = generate_and_save_deployment_code(
    'neural_network',
    model_data,
    'seeed_xiao',
    'generated',
    'balanced'
)

print(f"\n3. Generated {len(files)} files:")
for filepath, status in files.items():
    print(f"   {filepath}")
    print(f"     {status}")

print("\n" + "="*70)
print("✅ CODE REGENERATION COMPLETE!")
print("="*70)
print("\nThe new code includes feature scaling (StandardScaler):")
print(
    "  scaled_features[i] = (features[i] - feature_means[i]) / feature_stds[i]")
print("\nNow upload the new .ino file to your Seeed XIAO nRF52840!")
print("="*70)
