"""
Data Collection Guide for XIAO nRF52840
Collect real training data from your device to retrain the model
"""

# STEP 1: Modify Arduino sketch to output raw CSV data

# Add this to your .ino file

"""
void loop() {
    // ... existing sensor reading code ...

    // Output CSV format: timestamp,ax,ay,az,gx,gy,gz,label
    if (buffer_index >= WINDOW_SIZE) {
        buffer_index = 0;

        // Output all samples in window
        for (int i = 0; i < WINDOW_SIZE; i++) {
            Serial.print(millis());
            Serial.print(",");
            Serial.print(sensor_buffer[i][0], 4); // ax
            Serial.print(",");
            Serial.print(sensor_buffer[i][1], 4); // ay
            Serial.print(",");
            Serial.print(sensor_buffer[i][2], 4); // az
            Serial.print(",");
            Serial.print(sensor_buffer[i][3], 4); // gx
            Serial.print(",");
            Serial.print(sensor_buffer[i][4], 4); // gy
            Serial.print(",");
            Serial.print(sensor_buffer[i][5], 4); // gz
            Serial.print(",");
            Serial.println("LABEL_HERE");  // Manually label each activity
        }
    }
}
"""

# STEP 2: Collection Protocol

activities = {
    'still': {
        'description': 'Device on desk or held still',
        'duration': '2 minutes',
        'samples_needed': 120
    },
    'walking': {
        'description': 'Normal walking pace',
        'duration': '2 minutes',
        'samples_needed': 120
    },
    'running': {
        'description': 'Running or jogging',
        'duration': '2 minutes',
        'samples_needed': 120
    },
    'walking_upstairs': {
        'description': 'Walking up stairs',
        'duration': '2 minutes',
        'samples_needed': 120
    },
    'walking_downstairs': {
        'description': 'Walking down stairs',
        'duration': '2 minutes',
        'samples_needed': 120
    }
}

# STEP 3: Collect data

# For each activity

# 1. Modify code to print activity label

# 2. Upload to XIAO

# 3. Perform activity for 2 minutes

# 4. Save Serial output to CSV file

# 5. Name file: still.csv, walking.csv, running.csv, etc

# STEP 4: Prepare dataset

"""
import pandas as pd

# Combine all activity CSVs

dfs = []
for activity in ['still', 'walking', 'running', 'walking_upstairs', 'walking_downstairs']:
    df = pd.read_csv(f'{activity}.csv',
                     names=['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])
    df['label'] = activity
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)
combined.to_csv('xiao_har_dataset.csv', index=False)
print(f"Dataset created: {len(combined)} samples")
"""

# STEP 5: Retrain model

"""

# In the GUI app

# 1. Upload xiao_har_dataset.csv

# 2. Preprocess with same settings (100Hz, 1500ms windows)

# 3. Train neural network with same architecture (100, 50)

# 4. Generate new deployment code

# 5. Upload to XIAO

"""
