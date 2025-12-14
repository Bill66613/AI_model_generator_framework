"""Split large UCI-HAR CSV files into 30-second chunks (1500 samples at 50Hz)"""
import pandas as pd
import os

UCI_DIR = "data/UCI-HAR_converted"
CHUNK_SIZE = 1500  # 30 seconds at 50Hz

# Activities to split (excluding already split walking files)
activities_to_split = [
    'walking_downstairs',
    'walking_upstairs',
    'laying',
    'sitting',
    'standing'
]

print("=" * 70)
print("SPLITTING UCI-HAR FILES INTO 30-SECOND CHUNKS")
print("=" * 70)

for activity in activities_to_split:
    input_file = os.path.join(UCI_DIR, f"{activity}.csv")
    
    if not os.path.exists(input_file):
        print(f"\n⚠️ File not found: {input_file}")
        continue
    
    print(f"\n📂 Processing: {activity}.csv")
    
    # Read the full file
    df = pd.read_csv(input_file)
    total_samples = len(df)
    num_chunks = total_samples // CHUNK_SIZE
    
    print(f"   Total samples: {total_samples:,}")
    print(f"   Creating {num_chunks} chunks of {CHUNK_SIZE} samples each")
    
    # Split into chunks
    for i in range(num_chunks):
        start_idx = i * CHUNK_SIZE
        end_idx = start_idx + CHUNK_SIZE
        
        chunk_df = df.iloc[start_idx:end_idx]
        
        # Save chunk
        output_file = os.path.join(UCI_DIR, f"{activity}_{i+1}.csv")
        chunk_df.to_csv(output_file, index=False)
        print(f"   ✅ Created: {activity}_{i+1}.csv ({len(chunk_df)} samples)")
    
    # Handle remaining samples if any
    remaining = total_samples % CHUNK_SIZE
    if remaining > 0:
        print(f"   ℹ️ Remaining {remaining} samples (not saved, < 30 seconds)")
    
    # Optionally remove the original large file (commented out for safety)
    # os.remove(input_file)
    # print(f"   🗑️ Removed original file: {activity}.csv")

print("\n" + "=" * 70)
print("✅ SPLITTING COMPLETE")
print("=" * 70)
print("\nNote: Original large files are preserved.")
print("You can manually delete them if desired:")
for activity in activities_to_split:
    print(f"  - {activity}.csv")
