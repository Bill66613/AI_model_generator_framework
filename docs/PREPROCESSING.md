🔬 Normalization Methods - Impact on Training
📏 Min-Max Scaling (0-1)
How it works: Scales features to range [0, 1]

Formula: (x - min) / (max - min)
Impact on Training:

✅ Good for: Neural networks, SVM with RBF kernel
✅ Preserves relationships between data points
⚠️ Sensitive to outliers - one extreme value affects entire scale
⚠️ Poor generalization if test data exceeds training range
Best for HAR: Static activities (sitting, standing) with consistent ranges

📊 Standard Scaling (Z-score)
How it works: Centers data around mean=0, std=1

Formula: (x - mean) / std
Impact on Training:

✅ Robust to outliers compared to Min-Max
✅ Best for most ML algorithms (Random Forest, Logistic Regression)
✅ Assumes normal distribution
✅ Good generalization to unseen data
Best for HAR: Most recommended - works well with accelerometer/gyroscope data

🔄 Robust Scaling
How it works: Uses median and IQR instead of mean/std

Formula: (x - median) / IQR
Impact on Training:

✅ Highly robust to outliers
✅ Good for noisy sensor data
⚠️ May lose some signal information
Best for HAR: Noisy environments, low-quality sensors

❌ No Normalization
When to use:

Tree-based models (Random Forest, XGBoost) - they're scale-invariant
When features already have similar scales
Quick prototyping
🎯 Feature Selection - Impact on Training

🎯 All Features (138 features: Time + Frequency Domain)
Extracts: 90 time-domain features + 48 frequency-domain features
- Time: 15 statistical features per axis × 6 axes = 90 features
- Frequency: 8 FFT features per axis × 6 axes = 48 features

Impact:

✅ Maximum information preserved
✅ Good for complex activities requiring multiple sensors
✅ Captures both temporal patterns and frequency characteristics
⚠️ Risk of overfitting with small datasets
⚠️ Higher computational cost
⚠️ Frequency features cannot be deployed to Arduino
Best for: Complex activities (dancing, sports), large datasets, Python-only deployment

📈 Raw Axes Only (6 features)
Includes: Raw sensor readings only (aX, aY, aZ, gX, gY, gZ)

Impact:

✅ Extreme simplicity, fastest training
✅ Minimal computational resources
⚠️ Very limited discriminative power
⚠️ Poor performance on most activities
Best for: Testing, debugging, or extremely resource-constrained scenarios

🌊 Time-Domain Only (90 features)
Extracts: 15 statistical features per axis × 6 axes
- Features per axis: mean, std, min, max, range, median, q25, q75, iqr, skewness, kurtosis, rms, energy, zero_crossings, mean_crossing_rate

Impact:

✅ Excellent balance of performance and complexity
✅ Can be deployed to Arduino/embedded systems
✅ Good generalization with proper window size
✅ Captures temporal patterns without FFT complexity
⚠️ Missing frequency-domain information
Best for: Most real-world applications, embedded deployment, production systems

📊 Custom Selection
User-defined feature subset

Impact:

✅ Domain knowledge application
✅ Optimal balance of performance vs. complexity
🎯 Recommendations for HAR Training
For Different Activity Types:
🚶 Simple Activities (Walk, Sit, Stand)
🏃 Dynamic Activities (Run, Jump, Climb)
🤸 Complex Activities (Dance, Sports)
For Different Model Types:
🌳 Tree-Based Models (Random Forest)
🧠 Neural Networks
📏 SVM/Logistic Regression
For Different Deployment Scenarios:
📱 Mobile/Edge Devices
☁️ Cloud/Server Processing
🔋 Battery-Constrained Devices
🧪 Experimental Approach
For your HAR system, I recommend this testing sequence:

Start with Standard + All Features (baseline)
Try Robust + All Features (if noisy data)
Test Statistical Features (if overfitting occurs)
Compare with Tree model + No normalization (often surprises)