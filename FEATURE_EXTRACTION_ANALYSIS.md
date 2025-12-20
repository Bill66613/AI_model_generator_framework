# Feature Extraction Analysis: Why Training Results Got Worse

## Critical Difference Found

### Old Version (1dc7576) - `time_domain` mode:

```python
# Step 1: Load all windows into combined_df
combined_df = pd.concat(all_data, ignore_index=True)
# Result: DataFrame with ALL TIMESTEPS from ALL windows
# Example: 10 windows × 150 timesteps = 1500 rows
# Columns: aX, aY, aZ, gX, gY, gZ, Time_seconds, Window_ID, Activity_Label

# Step 2: Select time_domain columns (raw sensor data)
if feature_method == 'time_domain':
    feature_cols = [col for col in combined_df.columns 
                    if col not in ['Time_seconds', 'Window_ID', 'Activity_Label'] 
                    and not any(freq in col.lower() for freq in ['freq', 'fft', 'spectrum'])]
    # Result: ['aX', 'aY', 'aZ', 'gX', 'gY', 'gZ']

# Step 3: Use ALL ROWS as training data
X = combined_df[feature_cols].values  # Shape: (1500, 6)
y = combined_df['Activity_Label'].values  # Shape: (1500,)

# EACH TIMESTEP becomes a SEPARATE TRAINING SAMPLE!
```

**Training Data:**
- **1500 samples** (all timesteps treated as independent samples)
- **6 features** per sample (raw sensor values)
- Model learns from individual timestep patterns

---

### New Version (Current) - `time_domain` mode:

```python
# Step 1: Extract features FROM EACH WINDOW
for df in all_data:  # Iterate over windows
    window_data = df[sensor_cols]  # Get one window's data
    
    if feature_method == 'time_domain':
        # Extract 90 STATISTICAL features from the window
        features_df = extract_time_domain_features(window_data, sensor_cols)
        # Result: 1 row × 90 columns
        # Features: aX_mean, aX_std, aX_min, aX_max, aX_range, ...
        #           (15 features × 6 sensors = 90 total)
    
    extracted_features_list.append(features_df)

# Step 2: Combine extracted features
features_df = pd.concat(extracted_features_list, ignore_index=True)
X = features_df.values  # Shape: (10, 90)
y = np.array(labels_list)  # Shape: (10,)

# EACH WINDOW becomes ONE TRAINING SAMPLE with 90 FEATURES!
```

**Training Data:**
- **10 samples** (one per window)
- **90 features** per sample (statistical features: mean, std, min, max, etc.)
- Model learns from aggregated window statistics

---

## The Problem

### Sample Count Reduction:
- **Old**: 1500 training samples (all timesteps)
- **New**: 10 training samples (one per window)
- **Reduction**: **99.3% fewer samples!**

### Why This Causes Worse Results:

1. **Insufficient Training Data**
   - ML models need adequate samples to learn patterns
   - Going from 1500 to 10 samples is catastrophic
   - Even with 90 features, 10 samples → severe overfitting

2. **Wrong Approach**
   - Old version: Treated each timestep as independent (incorrect but had data volume)
   - New version: Correct window-level extraction BUT assumes you have many windows
   - For HAR research: Need 100s or 1000s of windows, not 10!

3. **Data vs Features Trade-off**
   - Old: High sample count (1500), low features (6) → model can learn
   - New: Low sample count (10), high features (90) → model cannot generalize

---

## What Should Happen

### Correct Approach (Standard HAR Pipeline):

1. **Collect More Data**
   - Record long time series (minutes/hours)
   - Split into MANY windows (100s or 1000s)
   - Example: 10 minutes @ 100Hz = 60,000 samples
   - With 1.5s windows: 60,000 / 150 = **400 windows**

2. **Extract Features Per Window**
   - Each window → 90 time-domain features
   - 400 windows → 400 training samples × 90 features
   - This is sufficient for training

3. **Train/Val/Test Split**
   - 400 samples → 280 train / 60 val / 60 test
   - Model can learn meaningful patterns

---

## Current Issue

You have **very few windows** (likely 10-30), which means:

### Old Version:
- Worked because it "artificially" increased samples by treating timesteps as independent
- **Statistically incorrect** but had enough data volume to train
- Like training on individual frames instead of video clips

### New Version:
- **Statistically correct** (one sample per window)
- **But insufficient data** to train properly
- Like trying to learn from 10 video clips → impossible

---

## Solutions

### Option 1: Collect More Data (Recommended)
```python
# Record longer sessions
# Current: 10 windows × 1.5s = 15 seconds total
# Need: 400 windows × 1.5s = 10 minutes per activity
```

### Option 2: Use Sliding Windows
```python
# Generate overlapping windows from your data
# Instead of 10 non-overlapping windows
# Create 100+ windows with 50% overlap
```

### Option 3: Revert to Old "Timestep" Approach (Not Recommended)
```python
# Treat each timestep as sample (statistically questionable)
# But will work with limited data
# This is what the old version did
```

### Option 4: Use Data Augmentation
```python
# Add noise, scale, rotate sensor readings
# Generate synthetic windows from existing ones
# Can increase 10 windows → 100+ augmented windows
```

---

## Verification

Check your actual data:

```python
# How many windows do you have?
import glob
windows = glob.glob("persistent_data/dragged_window_*")
print(f"Total windows: {len(windows)}")

# How many per activity?
# You need 50-100+ windows PER ACTIVITY for decent training
```

---

## Bottom Line

**The feature extraction is CORRECT and working properly.**

**The problem is insufficient data:**
- Old version "cheated" by treating timesteps as samples (1500 samples)
- New version is correct but needs many windows (you have ~10)
- You need 100-1000x more windows for proper training

**This is why your results got worse - not a bug, but a data volume issue!**
