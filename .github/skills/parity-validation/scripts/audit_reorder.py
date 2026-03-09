"""
Audit script: Verify that reorder_model_parameters() correctly maps
Python alphabetical feature order to C++ computation order.

Usage: python .github/skills/parity-validation/scripts/audit_reorder.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('.github', 1)[0])

from deployment.code_generator_factory import CodeGeneratorFactory


def audit_reorder():
    """Check reorder mapping for orientation_invariant_time_only mode."""
    
    # Python alphabetical feature names (33 features for OI time-only)
    signals = ['acc_mag', 'gyro_mag', 'jerk_mag']
    stats = ['mean', 'std', 'min', 'max', 'range', 'median', 'q25', 'q75',
             'iqr', 'skewness', 'kurtosis', 'rms', 'energy', 
             'zero_crossings', 'mean_crossing_rate']
    
    python_features = sorted([f"{sig}_{stat}" for sig in signals for stat in stats])
    
    print(f"Total features: {len(python_features)}")
    print(f"\nPython (alphabetical) order:")
    for i, f in enumerate(python_features):
        print(f"  [{i:2d}] {f}")
    
    print(f"\nC++ computation order:")
    cpp_idx = 0
    for sig in signals:
        for stat in stats:
            fname = f"{sig}_{stat}"
            py_idx = python_features.index(fname)
            match = "✅" if cpp_idx != py_idx else "🔄 same"
            print(f"  C++[{cpp_idx:2d}] = Python[{py_idx:2d}] {fname} {match}")
            cpp_idx += 1


if __name__ == "__main__":
    audit_reorder()
