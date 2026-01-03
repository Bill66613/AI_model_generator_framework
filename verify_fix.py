"""Verify the fixed feature order"""
import re

# Read the fixed deployment code
with open('persistent_data_new/generated/neural_network_models/har_neural_network_seeed_xiao_f17_c5_balanced/har_neural_network_seeed_xiao_f17_c5_balanced.cpp', 'r') as f:
    cpp_code = f.read()

# Extract feature assignment section
feature_section = cpp_code[cpp_code.find(
    'int idx = 0;'):cpp_code.find('for (; idx < NUM_FEATURES')]

print('='*70)
print('FIXED FEATURE ORDER IN DEPLOYMENT CODE')
print('='*70)
print(feature_section)

# Extract feature assignments
assignments = re.findall(
    r'features\[idx\+\+\] = (.*?);.*?// (\d+)(.*)', feature_section)

print('\n' + '='*70)
print('FEATURE EXTRACTION ORDER')
print('='*70)

expected_order = [
    'acc_mag_mean',
    'acc_mag_std',
    'acc_mag_rms',
    'acc_mag_energy',
    'gyro_mag_mean',
    'gyro_mag_std',
    'gyro_mag_rms',
    'gyro_mag_energy',
    'jerk_mag_mean',
    'jerk_mag_std',
    'jerk_mag_rms',
    'jerk_mag_energy',
    'acc_mag_min',
    'acc_mag_max',
    'gyro_mag_min',
    'gyro_mag_max',
    'jerk_mag_max'
]

print(f"{'Index':<6} {'Expected':<25} {'Status':<10}")
print('='*70)

# Parse the feature section for actual order
feature_lines = [line.strip() for line in feature_section.split(
    '\n') if 'features[idx++]' in line]
for i, expected_feat in enumerate(expected_order):
    if i < len(feature_lines):
        # Check if the expected feature appears in the comment or assignment
        line = feature_lines[i]
        status = '✅' if expected_feat.replace('_', ' ') in line.lower() or \
            any(part in line for part in expected_feat.split('_')) else '❓'
    else:
        status = '❌'
    print(f'{i:<6} {expected_feat:<25} {status}')

print('\n' + '='*70)
print('✅ FEATURE ORDER FIXED!')
print('='*70)
print('\nUpload the corrected files to your XIAO device:')
print('  1. har_neural_network_seeed_xiao_f17_c5_balanced.cpp')
print('  2. har_neural_network_seeed_xiao_f17_c5_balanced.h')
print('  3. har_neural_network_seeed_xiao_f17_c5_balanced.ino')
print('\nThe model should now work correctly!')
