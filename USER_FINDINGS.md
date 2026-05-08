# FINDINGS

## Signal Processing

- The Signal Processing Pipeline: Are they good for processing IMU data? In Edge Impulse, they have thing called Spectral analysis. It has Filter, prior to calculating the Fast Fourier Transform (FFT). Do we have them?
- Parity: How can we make sure the preprocessing in training pipeline matches the deployment one (later generated code).

## Feature Engineering

- Data Augmentation: Does this help? What is the correctness of these methods?

## Framework UI

- The "pytorch_cnn_har_model_20260424_000449" model is showing 53 features, but code gen show "f33" ("har_pytorch_cnn_seeed_xiao_f33_c3_balanced_int8"), which is wrong.
And the "Feature Count" in "Model Parameters (from training)" shows "150 samples x 6 ch (raw windows) (orientation-robust, time + freq (DFT))" which is hard to understand, should it be something calucating to 53, right?
- "features" and "num_features_extracted" in trained_models.json: meanings/differences?

## Deployment Testing

- It is not able to detect still/running/walking with different orientation of the device mounting positions, only correct when mounted same way as data get collected for training. Which studies show the resolutions for this problem?
- Is the current data preprocessing in deployment application code is correct?
- The inference time is way too long. It took ~1 second it seems.
- Can Kalman filter be applied to get the smoother signals?

## Studies/Researches

- The application, methods used in this framework have to be based on the reliable sources of studies, papers or researches.
