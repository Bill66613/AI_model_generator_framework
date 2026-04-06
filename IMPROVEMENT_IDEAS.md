# Future Improvement Ideas

Collected during code review sessions. These are potential enhancements to explore when time permits.

---

## 1. TensorFlow Lite Micro Integration

Replace the manual neural network C++ implementation with TFLite Micro for better performance, quantization support, and standard inference framework. This would simplify maintenance and potentially run faster on ARM Cortex-M devices.

## 2. Model Quantization (INT8)

Implement INT8 quantization for neural network weights and activations. This would:

- Reduce model size by ~4x (float32 → int8)
- Speed up inference on microcontrollers without FPU
- Enable deployment on even smaller devices (e.g., ATtiny series)

## 3. Partial Sort for Median Calculation

Replace full insertion sort with `std::nth_element` or a partial sort algorithm for the median computation in `extract_magnitude_stats()`. Current O(n²) insertion sort can be improved to O(n) average case.

## 4. 1D-CNN Model Support

Add support for 1D Convolutional Neural Networks which are well-suited for time-series HAR tasks. CNNs can capture local temporal patterns and often outperform MLPs on raw sensor data. Would require:

- New model type in `EdgeMLModel`
- New C++ code generator for conv1d + pooling layers
- Window-level raw data input instead of hand-crafted features

## 5. Online/Incremental Learning

Allow the model to adapt to new users or environments over time using incremental/online learning techniques. Useful for personalization without full retraining.

## 6. Power-Aware Sampling

Implement adaptive sampling rate that reduces sensor polling frequency during low-activity periods (e.g., "still" detected → reduce to 25Hz). This could significantly extend battery life on wearable devices.

## 7. Activity Transition Detection

Add a transition detector that recognizes when the user is switching between activities. This could use:

- Sliding window with overlap analysis
- Confidence thresholding (low confidence → transition)
- Temporal smoothing / majority voting over recent predictions

## 8. Confusion Matrix-Guided Feature Selection

Use the confusion matrix from model evaluation to identify which classes are commonly confused, then engineer targeted features to discriminate those specific activities.

## 9. Multi-Sensor Fusion

Support additional sensors beyond accelerometer + gyroscope:

- Magnetometer for heading/orientation
- Barometer for altitude changes (stairs detection)
- Heart rate sensor for metabolic activity correlation

## 10. Model Compression via Pruning

Implement weight pruning for neural networks — remove weights below a threshold, resulting in sparse matrices that need less memory and computation. Combined with quantization, this could yield 10-20x model size reduction.

## 11. Real-Time Visualization Dashboard

Create a live dashboard showing:

- Current prediction with confidence
- Feature values in real-time
- Prediction history timeline
- Confusion patterns over time
- Battery/memory usage estimates

## 12. Automated Hyperparameter Search with Bayesian Optimization

Replace grid search with Bayesian optimization (e.g., Optuna) for more efficient hyperparameter tuning. This would:

- Reduce search time significantly
- Find better parameter combinations
- Support conditional hyperparameters
- Provide convergence plots

---

*Last updated: Session 14 of development*
