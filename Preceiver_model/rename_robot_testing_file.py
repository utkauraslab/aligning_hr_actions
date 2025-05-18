import pandas as pd
from pathlib import Path

# Step 1: Load the CSV
csv_path = Path("/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/data_robot_0004_test.csv")
df = pd.read_csv(csv_path)

# Step 2: Clean and transform
# Rename timestamp to filename
df["filename"] = df.iloc[:, 0].astype(str) + ".npy"

# Create label map
label_map = {
    'nothing': 0,
    'reaching': 1,
    'grasping': 2,
    'lifting': 3,
    'transporting': 4,
    'holding': 5,
    'placing': 6,
    'releasing': 7
}

# Map labels
df["label_str"] = df.iloc[:, 1].astype(str)
df["label"] = df["label_str"].map(label_map)

# Optionally: filter out invalid labels (NaNs)
df = df[df["label"].notna()].copy()
df["label"] = df["label"].astype(int)

# Step 3: Keep only necessary columns
val_df = df[["filename", "label"]]

# Step 4: Save cleaned version
val_df.to_csv("/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/data_robot_0004_test_renamed.txt", sep="\t", index=False, header=False)

print("✅ Processed validation list saved to processed_val_list.txt")
