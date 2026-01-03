import joblib

model_data = joblib.load('persistent_data_new/models/neural_network_har_model_20251223_230846.joblib')
model = model_data['model']

print(f"Number of layers: {model.n_layers_}")
print(f"Hidden layer sizes: {model.hidden_layer_sizes}")
print(f"\nAll coefficient shapes:")
for i, coef in enumerate(model.coefs_):
    print(f"  Layer {i} -> {i+1}: {coef.shape}")
print(f"\nAll bias shapes:")
for i, intercept in enumerate(model.intercepts_):
    print(f"  Layer {i+1}: {intercept.shape}")