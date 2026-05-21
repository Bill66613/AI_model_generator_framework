"""Diagnostic: validate SVM OvO→OvR conversion against sklearn predict()."""
import sys
import numpy as np
import joblib
import glob
import os

# Find the latest SVM model
model_dir = r"C:\Users\ntmhb\OneDrive\Documents\Education\Master\training-xiao-ankle\models"
svm_files = sorted(glob.glob(os.path.join(model_dir, "svm_*.joblib")))
if not svm_files:
    print("No SVM model found")
    sys.exit(1)

model_path = svm_files[-1]
print(f"Loading: {model_path}")
data = joblib.load(model_path)
model_obj = data if hasattr(data, 'model') else None
if model_obj is None and isinstance(data, dict):
    # Try EdgeMLModel inside dict
    model_obj = data.get('model_object', data)

svm = model_obj.model if hasattr(model_obj, 'model') else data.get('model')
scaler = model_obj.scaler if hasattr(model_obj, 'scaler') else data.get('scaler')
le = model_obj.label_encoder if hasattr(model_obj, 'label_encoder') else data.get('label_encoder')
feature_names = model_obj.feature_names if hasattr(model_obj, 'feature_names') else data.get('feature_names', [])
feature_names = feature_names or []

print(f"Model type: {type(svm).__name__}")
print(f"Classes: {svm.classes_}")
print(f"Label encoder classes: {le.classes_ if le else 'N/A'}")
print(f"n_support_: {svm.n_support_}")
print(f"n_SV total: {len(svm.support_vectors_)}")
print(f"gamma param: {svm.gamma}")
if hasattr(svm, '_gamma'):
    print(f"_gamma (resolved): {svm._gamma}")
print(f"SV shape: {svm.support_vectors_.shape}")
print(f"dual_coef_ shape: {svm.dual_coef_.shape}")
print(f"intercept_ shape: {svm.intercept_.shape}")
print(f"intercept_ values: {svm.intercept_}")
print(f"Feature names ({len(feature_names)}): {feature_names[:5]}...{feature_names[-3:]}")
print()

# === OvO → OvR conversion (same as extract_svm_parameters) ===
n_classes = len(svm.classes_)
n_sv = len(svm.support_vectors_)
n_support = list(int(x) for x in svm.n_support_)

class_start = [0]
for ns in n_support:
    class_start.append(class_start[-1] + ns)
print(f"Class SV ranges: {list(zip(class_start[:-1], class_start[1:]))}")

ovr_coef = np.zeros((n_classes, n_sv))
ovr_intercept = np.zeros(n_classes)

pair_idx = 0
for ci in range(n_classes):
    for cj in range(ci + 1, n_classes):
        pair_coef = np.zeros(n_sv)
        for s in range(class_start[ci], class_start[ci + 1]):
            pair_coef[s] = svm.dual_coef_[cj - 1, s]
        for s in range(class_start[cj], class_start[cj + 1]):
            pair_coef[s] = svm.dual_coef_[ci, s]
        ovr_coef[ci] += pair_coef
        ovr_intercept[ci] += svm.intercept_[pair_idx]
        ovr_coef[cj] -= pair_coef
        ovr_intercept[cj] -= svm.intercept_[pair_idx]
        pair_idx += 1

print(f"OvR intercept: {ovr_intercept}")
print(f"OvR coef sums per class: {ovr_coef.sum(axis=1)}")
print()

# === Test on synthetic data: use SVs themselves as test points ===
# Take a few SVs from each class
test_indices = []
for k in range(n_classes):
    start, end = class_start[k], class_start[k + 1]
    test_indices.extend(range(start, min(start + 3, end)))

X_test = svm.support_vectors_[test_indices]  # Already scaled
print(f"Testing on {len(X_test)} support vectors")

# sklearn prediction (ground truth)
sklearn_pred = svm.predict(X_test)
sklearn_dec = svm.decision_function(X_test)
print(f"\nsklearn predict: {sklearn_pred}")
print(f"sklearn decision_function shape: {sklearn_dec.shape}")

# Our OvR prediction
gamma = float(svm._gamma) if hasattr(svm, '_gamma') else 1.0 / (X_test.shape[1] * np.var(svm.support_vectors_))

def our_predict(x, sv, ovr_coef, ovr_intercept, gamma):
    """Replicate the C++ har_model_predict logic."""
    scores = np.zeros(n_classes)
    for c in range(n_classes):
        s = ovr_intercept[c]
        for sv_idx in range(n_sv):
            dist2 = np.sum((x - sv[sv_idx]) ** 2)
            k = np.exp(-gamma * dist2)
            s += ovr_coef[c, sv_idx] * k
        scores[c] = s
    return scores

print("\n=== Per-sample comparison ===")
mismatches = 0
for i, x in enumerate(X_test):
    ovr_scores = our_predict(x, svm.support_vectors_, ovr_coef, ovr_intercept, gamma)
    ovr_pred = svm.classes_[np.argmax(ovr_scores)]
    sk_pred = sklearn_pred[i]
    
    # Softmax of scores
    exp_scores = np.exp(ovr_scores - np.max(ovr_scores))
    probs = exp_scores / exp_scores.sum()
    
    match = "OK" if ovr_pred == sk_pred else "MISMATCH"
    if ovr_pred != sk_pred:
        mismatches += 1
    print(f"  SV[{test_indices[i]:3d}] class={svm.classes_[test_indices[i] // max(1, n_sv // n_classes)]}: "
          f"sklearn={sk_pred}, OvR={ovr_pred}, scores={ovr_scores}, "
          f"softmax={probs}, max_prob={probs.max():.4f} {match}")

# Also test with training data if available
train_dir = r"C:\Users\ntmhb\OneDrive\Documents\Education\Master\training-xiao-ankle\training"
train_files = sorted(glob.glob(os.path.join(train_dir, "*and_2_more_oi63_train.csv")))

if train_files:
    import pandas as pd
    print(f"\n=== Testing on training data: {train_files[-1]} ===")
    df = pd.read_csv(train_files[-1])
    if 'activity' in df.columns:
        X = df.drop(columns=['activity']).values
        y = df['activity'].values
        if scaler:
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X
        
        # sklearn predictions
        sk_preds = svm.predict(X_scaled)
        sk_acc = np.mean(sk_preds == y)
        
        # Our OvR predictions on first 20 samples
        n_test = min(20, len(X_scaled))
        our_preds = []
        for i in range(n_test):
            scores = our_predict(X_scaled[i], svm.support_vectors_, ovr_coef, ovr_intercept, gamma)
            pred_class = svm.classes_[np.argmax(scores)]
            our_preds.append(pred_class)
            
            exp_s = np.exp(scores - np.max(scores))
            probs = exp_s / exp_s.sum()
            match = "OK" if pred_class == sk_preds[i] else "XX"
            print(f"  [{i:3d}] true={y[i]:10s} sklearn={sk_preds[i]:10s} OvR={pred_class:10s} "
                  f"scores=[{scores[0]:+.3f},{scores[1]:+.3f},{scores[2]:+.3f}] "
                  f"probs=[{probs[0]:.3f},{probs[1]:.3f},{probs[2]:.3f}] {match}")
        
        our_preds = np.array(our_preds)
        print(f"\nsklearn accuracy on train: {sk_acc:.4f}")
        print(f"OvR match rate vs sklearn (first {n_test}): {np.mean(our_preds == sk_preds[:n_test]):.4f}")
        
        # Check unique predictions
        all_our = []
        for i in range(len(X_scaled)):
            scores = our_predict(X_scaled[i], svm.support_vectors_, ovr_coef, ovr_intercept, gamma)
            all_our.append(svm.classes_[np.argmax(scores)])
        all_our = np.array(all_our)
        print(f"\nOvR unique predictions: {np.unique(all_our, return_counts=True)}")
        print(f"sklearn unique predictions: {np.unique(sk_preds, return_counts=True)}")
else:
    print("\nNo training data found for full test")

print(f"\nTotal mismatches on SV test: {mismatches}/{len(X_test)}")

# === Verify reordering: compare feature order ===
print("\n=== Feature order verification ===")
# Import the reorder functions
sys.path.insert(0, ".")
from deployment.code_generator_factory import get_cpp_feature_order, compute_feature_reorder_indices
cpp_order = get_cpp_feature_order(feature_names)
reorder_idx = compute_feature_reorder_indices(feature_names, cpp_order)
if reorder_idx is not None:
    print(f"Reorder needed: model[0]={feature_names[0]} -> cpp[0]={cpp_order[0]}")
    print(f"First 10 reorder indices: {reorder_idx[:10]}")
    # Show: first 5 features in model order vs cpp order
    print(f"Model order (first 5): {feature_names[:5]}")
    print(f"C++ order (first 5):   {cpp_order[:5]}")
    
    # Reorder scaler and compare with generated code
    reordered_means = [scaler.mean_[i] for i in reorder_idx]
    reordered_stds = [scaler.scale_[i] for i in reorder_idx]
    
    # Read generated scaler from har_classifier.cpp
    gen_dir = r"C:\Users\ntmhb\OneDrive\Documents\Education\Master\training-xiao-ankle\generated\svm_models\har_svm_seeed_xiao_f63_c5_balanced_int8"
    import re
    with open(os.path.join(gen_dir, "har_classifier.cpp"), "r") as f:
        content = f.read()
    
    # Extract SCALER_MEANS array
    means_match = re.search(r'SCALER_MEANS\[HAR_NUM_FEATURES\] = \{([^}]+)\}', content)
    stds_match = re.search(r'SCALER_STDS\[HAR_NUM_FEATURES\] = \{([^}]+)\}', content)
    
    if means_match:
        gen_means = [float(x.strip().rstrip('f')) for x in means_match.group(1).split(',')]
        print(f"\n=== Scaler comparison (reordered model vs generated code) ===")
        max_diff_mean = 0
        for i in range(min(5, len(gen_means))):
            diff = abs(reordered_means[i] - gen_means[i])
            max_diff_mean = max(max_diff_mean, diff)
            print(f"  [{i:2d}] {cpp_order[i]:30s}: model={reordered_means[i]:12.4f}  gen={gen_means[i]:12.4f}  diff={diff:.6f}")
        # Check all 63
        all_diffs = [abs(reordered_means[i] - gen_means[i]) for i in range(len(gen_means))]
        max_diff = max(all_diffs)
        avg_diff = sum(all_diffs) / len(all_diffs)
        print(f"  Scaler means: max_diff={max_diff:.6f}, avg_diff={avg_diff:.6f}")
    
    if stds_match:
        gen_stds = [float(x.strip().rstrip('f')) for x in stds_match.group(1).split(',')]
        all_diffs_std = [abs(reordered_stds[i] - gen_stds[i]) for i in range(len(gen_stds))]
        max_diff_std = max(all_diffs_std)
        avg_diff_std = sum(all_diffs_std) / len(all_diffs_std)
        print(f"  Scaler stds:  max_diff={max_diff_std:.6f}, avg_diff={avg_diff_std:.6f}")
    
    # Reorder SVs and compare first SV with generated code
    reordered_sv0 = svm.support_vectors_[0][reorder_idx]
    
    # Extract first SV from generated code
    sv_match = re.search(r'har_svm_sv\[HAR_SVM_N_SV\]\[HAR_NUM_FEATURES\] = \{[\s\n]*\{([^}]+)\}', content.replace('\n', ' '))
    if not sv_match:
        with open(os.path.join(gen_dir, "har_model.cpp"), "r") as f:
            model_content = f.read()
        sv_match = re.search(r'har_svm_sv\[HAR_SVM_N_SV\]\[HAR_NUM_FEATURES\] = \{[\s\n]*\{([^}]+)\}', model_content.replace('\n', ' '))
    
    if sv_match:
        gen_sv0 = [float(x.strip().rstrip('f')) for x in sv_match.group(1).split(',')]
        print(f"\n=== SV[0] comparison (reordered model vs generated code) ===")
        for i in range(min(5, len(gen_sv0))):
            diff = abs(reordered_sv0[i] - gen_sv0[i])
            print(f"  [{i:2d}] {cpp_order[i]:30s}: model={reordered_sv0[i]:8.4f}  gen={gen_sv0[i]:8.4f}  diff={diff:.6f}")
        all_diffs_sv = [abs(reordered_sv0[i] - gen_sv0[i]) for i in range(len(gen_sv0))]
        max_diff_sv = max(all_diffs_sv)
        print(f"  SV[0]: max_diff={max_diff_sv:.6f}")
    
    # Compare OvR intercepts with generated code
    int_match = re.search(r'har_svm_intercept\[HAR_NUM_CLASSES\] = \{([^}]+)\}', 
                          model_content if 'model_content' in dir() else content)
    if int_match:
        gen_int = [float(x.strip().rstrip('f')) for x in int_match.group(1).split(',')]
        print(f"\n=== OvR intercept comparison ===")
        for i in range(len(gen_int)):
            diff = abs(ovr_intercept[i] - gen_int[i])
            print(f"  class {i}: model={ovr_intercept[i]:+.6f}  gen={gen_int[i]:+.6f}  diff={diff:.6f}")
else:
    print("No reordering needed — features already in C++ order")
