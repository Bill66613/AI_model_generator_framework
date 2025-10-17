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
🎯 All Features
Includes: All 6 sensors (aX, aY, aZ, gX, gY, gZ)

Impact:

✅ Maximum information preserved
✅ Good for complex activities requiring multiple sensors
⚠️ Risk of overfitting with small datasets
⚠️ Higher computational cost
Best for: Complex activities (dancing, sports), large datasets

📈 Statistical Features
Includes: Mean, std, min, max, range for each sensor

Impact:

✅ Dimensionality reduction while preserving key characteristics
✅ Better generalization often achieved
✅ Reduces noise from raw sensor readings
⚠️ May lose temporal patterns
Best for: Simple activities, limited computational resources

🌊 Time-Domain Only
Includes: Raw accelerometer data (aX, aY, aZ)

Impact:

✅ Lower complexity, faster training
✅ Good for basic activities (walk, run, sit)
⚠️ Limited for rotation-based activities
Best for: Simple activity classification, battery-constrained devices

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