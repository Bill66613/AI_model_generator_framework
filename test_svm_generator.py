"""
Test SVM Code Generator
Verify that the SVM generator produces valid Arduino code
"""

import numpy as np
from deployment.svm_generator import SVMCodeGenerator

# Create test model data
test_model_data = {
    'model_type': 'svm',
    'feature_names': [f'feature_{i}' for i in range(10)],
    'classes': ['class_0', 'class_1', 'class_2'],
    'feature_means': [0.5] * 10,
    'feature_stds': [1.0] * 10,

    # SVM-specific parameters
    'support_vectors': [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    ],
    'dual_coefficients': [
        [0.5, -0.3, 0.2],  # Class 0
        [-0.5, 0.3, -0.2],  # Class 1
        [0.1, -0.1, 0.05],  # Class 2
    ],
    'intercept': [0.1, -0.2, 0.3],
    'gamma': 0.05
}


def test_svm_generator():
    """Test SVM code generator."""
    print("🧪 Testing SVM Code Generator...")

    try:
        # Create generator
        generator = SVMCodeGenerator(
            model_data=test_model_data,
            platform='seeed_xiao',
            optimization='balanced'
        )
        print("✅ Generator created successfully")

        # Generate header
        header = generator.generate_header()
        print("\n📄 Header file preview (first 500 chars):")
        print(header[:500])
        print("...")

        # Check for required elements
        assert 'NUM_SUPPORT_VECTORS' in header, "Missing NUM_SUPPORT_VECTORS define"
        assert 'NUM_FEATURES 10' in header, "Missing NUM_FEATURES define"
        assert 'NUM_CLASSES 3' in header, "Missing NUM_CLASSES define"
        assert 'rbf_kernel' in header, "Missing rbf_kernel declaration"
        print("✅ Header contains required declarations")

        # Generate implementation
        impl = generator.generate_implementation("test_svm.h")
        print("\n📄 Implementation file preview (first 800 chars):")
        print(impl[:800])
        print("...")

        # Check for required elements
        assert 'support_vectors[' in impl, "Missing support_vectors array"
        assert 'dual_coef[' in impl, "Missing dual_coef array"
        assert 'intercepts[' in impl, "Missing intercepts array"
        assert 'svm_gamma' in impl, "Missing gamma parameter"
        assert 'rbf_kernel' in impl, "Missing RBF kernel function"
        assert 'har_predict_internal' in impl, "Missing prediction function"
        assert 'exp(-gamma * sum)' in impl, "Missing RBF kernel computation"
        print("✅ Implementation contains required SVM components")

        # Check for compilation issues
        assert 'scaled_features[i] = (features[i] - feature_means[i]) / feature_stds[i]' not in impl, \
            "ERROR: Duplicate feature scaling found in har_predict_internal"
        print("✅ No duplicate feature scaling")

        # Check dual coefficient structure
        assert 'dual_coef[cls][sv]' in impl, "Missing correct dual coefficient indexing"
        print("✅ Correct multi-class SVM structure")

        # Generate example sketch
        sketch = generator.generate_example_sketch("test_svm.h")
        print("\n📄 Sketch file preview (first 500 chars):")
        print(sketch[:500])
        print("...")

        assert '#include "test_svm.h"' in sketch, "Missing header include"
        assert 'LSM6DS3' in sketch or 'IMU' in sketch.upper(), "Missing IMU initialization"
        print("✅ Sketch contains required components")

        print("\n" + "="*60)
        print("🎉 ALL TESTS PASSED!")
        print("="*60)
        print("\nSVM Generator Summary:")
        print(
            f"  • Support vectors: {len(test_model_data['support_vectors'])}")
        print(f"  • Features: {len(test_model_data['feature_names'])}")
        print(f"  • Classes: {len(test_model_data['classes'])}")
        print(f"  • Gamma: {test_model_data['gamma']}")
        print(f"  • Platform: seeed_xiao")
        print(f"  • Optimization: balanced")

        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_svm_generator()
    exit(0 if success else 1)
