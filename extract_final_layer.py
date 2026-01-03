import joblib
import numpy as np

# Load the model
model_data = joblib.load('persistent_data_new/models/neural_network_har_model_20251223_230846.joblib')
model = model_data['model']

# Extract the missing final layer weights (50x5)
print("="*60)
print("EXTRACTING MISSING FINAL LAYER WEIGHTS")
print("="*60)

final_weights = model.coefs_[2]  # Shape: (50, 5)
final_biases = model.intercepts_[2]  # Shape: (5,)

print(f"\nFinal layer weights (50x5):")
print("const float final_weights[50][5] = {")
for i in range(50):
    row = ", ".join([f"{final_weights[i, j]:.3f}" for j in range(5)])
    print(f"    {{{row}}},")
print("};")

print(f"\nFinal layer biases (5):")
print("const float final_biases[5] = {")
bias_str = ", ".join([f"{b:.3f}" for b in final_biases])
print(f"    {bias_str}")
print("};")
