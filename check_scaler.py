import joblib
import numpy as np

model_dict = joblib.load(
    r'd:\Workspaces\Master\ComputerScience\Thesis\GUI_app\persistent_data_new\models\neural_network_har_model_20260205_005815.joblib')
scaler = model_dict['scaler']

print('First 15 feature means (full precision):')
for i in range(15):
    print(f'  [{i:2d}] {scaler.mean_[i]:12.8f}')

print('\nWith 1 decimal rounding (power optimization):')
for i in range(15):
    rounded = round(scaler.mean_[i], 1)
    print(f'  [{i:2d}] {rounded:5.1f}')

print('\n=== THE ISSUE ===')
print('When all values are < 0.05, rounding to 1 decimal makes them ALL 0.0!')
print('This destroys the scaler information.')
print('\nSolution: Use at least 2-3 decimal places for power optimization.')
