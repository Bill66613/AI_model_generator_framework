"""
Check what's in the model file
"""

import joblib

model_data = joblib.load(
    'persistent_data_new/models/neural_network_har_model_20251223_230846.joblib')

print("Model data keys:", model_data.keys()
      if isinstance(model_data, dict) else "Not a dict")
print("\nModel data type:", type(model_data))

if isinstance(model_data, dict):
    for key, value in model_data.items():
        print(f"\n{key}: {type(value)}")
        if key == 'model_object':
            print(f"  Has 'model' attr: {hasattr(value, 'model')}")
            if hasattr(value, 'model'):
                print(f"  Model type: {type(value.model)}")
                print(f"  Has 'coefs_': {hasattr(value.model, 'coefs_')}")
                if hasattr(value.model, 'coefs_'):
                    print(
                        f"  Number of weight layers: {len(value.model.coefs_)}")
                    for i, coef in enumerate(value.model.coefs_):
                        print(f"    Layer {i}: {coef.shape}")
