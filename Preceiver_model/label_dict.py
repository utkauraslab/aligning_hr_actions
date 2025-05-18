import pandas as pd
import numpy as np
from pathlib import Path
import json
import pdb

# path
# voxel_dir = Path(r"E:\UTK\Reseach\Publication\Workshop\ICRA2025_workshop_dataset_HumanRobotCorr-main\dataset_depth\dataset_depth\robot\cam_104122061850\pick\0004")
# csv_path = Path(r"E:\UTK\Reseach\Publication\Workshop\ICRA2025_workshop_dataset_HumanRobotCorr-main\dataset_depth\dataset_depth\robot\cam_104122061850\pick\0004\Labelled_Robot_004.csv")

voxel_dir = Path('/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004')
csv_path = Path('/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004/Labelled_Robot_004.csv')


# read CSV and fix Timestamp type
df = pd.read_csv(csv_path)
df['Timestamp'] = df['Timestamp'].apply(lambda x: str(int(float(x))))

# build label map
label_names = df['ID'].unique()
label_map = {name: idx for idx, name in enumerate(label_names)}

pdb.set_trace()

# generate {voxel_file_stem: label_index}
label_dict = {}
for f in voxel_dir.rglob("*.npy"):
    stem = f.stem
    match = df[df['Timestamp'] == stem]
    if not match.empty:
        action = match.iloc[0]['ID']
        label_dict[stem] = label_map[action]

# save as JSON
output_path = voxel_dir.parent / "label_dict_0004.json"
with open(output_path, "w") as f:
    json.dump(label_dict, f)

print(f"✅ label_dict saved to: {output_path}")
