# TODO.md

## Todo

- [ ] Implement code generation & deployment tab
  - [ ] Create layout for code generation UI
  - [ ] Implement compilation with Arduino CLI
  - [ ] Add flash to device functionality
  - [ ] Serial port auto-detection
- [ ] Implement device testing tab (UART)
  - [ ] Real-time serial communication
  - [ ] Live sensor data plotting
  - [ ] Real-time activity classification
- [ ] Update training tab to use engineered-dataset-store
- [ ] Add model export functionality

## In Progress

- [ ] Testing complete feature engineering workflow
  - [✓] UI implemented
  - [✓] Callbacks implemented
  - [ ] End-to-end testing with all activity labels

## Done ✓

- [✓] Framework restructured to 5 tabs (Data, Preprocessing, Feature Engineering, Training, Device Test)
- [✓] Feature engineering tab created with unified workflow
- [✓] Feature engineering callbacks fully implemented
  - [✓] populate_activity_labels() - Load labels from metadata
  - [✓] update_windows_per_label() - Display window counts
  - [✓] handle_label_selection_buttons() - Select All/Clear functionality
  - [✓] update_feature_count() - Display feature counts based on method
  - [✓] calculate_test_split() - Auto-calculate test percentage
  - [✓] execute_feature_engineering() - Unified feature engineering pipeline
- [✓] Fixed window naming display (removed "Manual"/"Window" prefix)
- [✓] Fixed "Load Previous" button to work independently
- [✓] Fixed float conversion issues in windowing callbacks
- [✓] Apply time window functionality
  - [✓] Display graph only for the input time (based on sampling rate)
  - [✓] Able to select window of data and capture in file
  - [✓] Draggable window selection on graph
- [✓] Split selected time windows
  - [✓] Understand and implement relayoutData handling
  - [✓] Save split windows to persistent storage
  - [✓] Load previous windows from metadata
- [✓] Sliding window generation
  - [✓] Generate sliding windows from current selection
  - [✓] Merge with manual windows
  - [✓] Handle overlaps and gaps
- [✓] Preprocessing tab refactored (removed feature engineering section)
