import cv2
import torch

from DAV2.depth_anything_v2.dpt import DepthAnythingV2
from pcl_voxelization import find_image_by_name, get_timestamps_from_csv

import numpy as np
from pathlib import Path

import pdb


DEVICE = 'cuda:0' if torch.cuda.is_available(
) else 'mps' if torch.backends.mps.is_available() else 'cpu'


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vits'  # or 'vits', 'vitb', 'vitg'


model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(
    f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()


# ----Configuration ----
label_cvs_path = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004/Labelled_Robot_004.csv'
image_root_folder = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot'

# ---- Process ----
timestamps = get_timestamps_from_csv(label_cvs_path)

pdb.set_trace()

for timestamp in timestamps:
    image_filename = f"{timestamp}.jpg"
    depth_image_path, rgb_image_path, depth_v2_image_path = find_image_by_name(
        image_filename, image_root_folder)

    if 'user_0002_scene_0001' in str(depth_image_path):
        # pdb.set_trace()
        continue  # skip this one

    raw_img = cv2.imread(rgb_image_path)
    depth = model.infer_image(raw_img)  # HxW raw depth map in numpy

    # Assume depth is float32, HxW
    depth_min = depth.min()
    depth_max = depth.max()

    # Avoid divide by zero
    if depth_max - depth_min > 0:
        depth_normalized = (depth - depth_min) / (depth_max - depth_min)
    else:
        depth_normalized = np.zeros_like(depth)

    # # Scale to 16-bit (0–65535)
    depth_16bit = (depth_normalized * 65535).astype(np.uint16)

    depth_8bit = (depth_normalized * 255).astype(np.uint8)

    # pdb.set_trace()

    output_dir = Path(depth_image_path).parent.parent / 'depth_anything_v2'
    # Create folder if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / image_filename

    # Save as 16-bit PNG
    cv2.imwrite(str(output_path), depth_8bit)
    print(f"Depth image saved to: {output_path}")

    # pdb.set_trace()

pdb.set_trace()
