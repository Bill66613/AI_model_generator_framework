import os
import pandas as pd

# Define the directory where your CSV files are stored
csv_directory = '100hz'

# List all CSV files in the directory
csv_files = [f for f in os.listdir(csv_directory) if f.endswith('.csv')]

# Initialize an empty list to hold the DataFrames
data_frames = []

# Loop through each CSV file
for file in csv_files:
    # Define the full path to the file
    file_path = os.path.join(csv_directory, file)

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Add the 'label' column with the file's name (without extension)
    df['label'] = os.path.splitext(file)[0]

    # Append the DataFrame to the list
    data_frames.append(df)

# Concatenate all DataFrames in the list into one
combined_df = pd.concat(data_frames, ignore_index=True)

# Save the combined DataFrame to a new CSV file
combined_df.to_csv(f'{csv_directory}/combined_data.csv', index=False)

print("CSV files have been successfully merged!")
