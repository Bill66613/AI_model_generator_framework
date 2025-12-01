import joblib

m = joblib.load(
    'persistent_data/neural_network_optimized_20251124_234953.joblib')
le = m['label_encoder']

print('Label Encoder Class Mapping:')
print('='*50)
for i, cls in enumerate(le.classes_):
    print(f'  {i}: {cls}')
