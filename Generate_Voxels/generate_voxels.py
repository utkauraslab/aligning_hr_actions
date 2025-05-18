import torch
import numpy as np
import cv2
import os
import pandas as pd
import utilis   # Assuming utilis is a custom module you have
from pathlib import Path
import shutil

import pdb

# settings
# MIN_DENOMINATOR = 1e-12
# INCLUDE_PER_VOXEL_COORD = False

CAMERAS_NUM = 1
VOXEL_SIZES = [18]  # 100x100x100 voxels
# NUM_LATENTS = 512 # PerceiverIO latents
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]

BATCH_SIZE = 1
# NUM_DEMOS = 8 # total number of training demonstrations to use while training PerAct
# NUM_TEST = 2 # episodes to evaluate on
IMAGE_H = 360
IMAGE_W = 640

# Camera intrinsic matrix
CAM_INTRINSIC = np.array([[908.49682617/2, 0, 640.63098145/2],
                          [0, 907.65515137/2, 351.33010864/2],
                          [0, 0, 1]])

# Automatically use GPU if available, otherwise fallback to CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cuda:0"
# tensorize scene bounds
bounds = torch.tensor(SCENE_BOUNDS, device=DEVICE).unsqueeze(0)


def get_timestamps_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Get all entries in first column as strings
    timestamp = df.iloc[:, 0].astype(str).tolist()
    return timestamp


def find_image_by_name(image_name, search_path):
    depth_image_path = None
    rgb_image_path = None
    depth_v2_image_path = None

    for root, dirs, files in os.walk(search_path):
        if image_name in files:
            full_path = os.path.join(root, image_name)
            if '/depth/' in full_path or os.path.normpath(full_path).split(os.sep)[-2] == 'depth':
                depth_image_path = full_path
            elif '/rgb/' in full_path or os.path.normpath(full_path).split(os.sep)[-2] == 'rgb':
                rgb_image_path = full_path

            if '/depth_anything_v2/' in full_path or os.path.normpath(full_path).split(os.sep)[-2] == 'depth_anything_v2':
                depth_v2_image_path = full_path

    return depth_image_path, rgb_image_path, depth_v2_image_path


if __name__ == "__main__":

    # ----Configuration ----
    label_cvs_path = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004/Labelled_Robot_004.csv'
    image_root_folder = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot'

    # ---- Process ----
    timestamps = get_timestamps_from_csv(label_cvs_path)

    pcl_generator = utilis.RGBDPointCloudGenerator(min_depth_m=0.4, max_depth_m=1)

    voxel_generator = utilis.VoxelGenerator(scene_bounds=SCENE_BOUNDS,
                                            voxel_sizes=VOXEL_SIZES,
                                            batch_size=BATCH_SIZE,
                                            image_size=[IMAGE_H, IMAGE_W],
                                            cam_num=CAMERAS_NUM,
                                            device=DEVICE,
                                            )

    for timestamp in timestamps:
        image_filename = f"{timestamp}.jpg"
        depth_image_path, rgb_image_path, depth_v2_image_path = find_image_by_name(image_filename, image_root_folder)
        print(f"RGB & Depth image: {image_filename} found!")

        # if depth_image_path and rgb_image_path:
        if depth_v2_image_path and rgb_image_path:

            # Read images
            rgb_path = rgb_image_path
            # depth_path = depth_image_path
            depth_path = depth_v2_image_path

            # Assume: rgb_np = (H, W, 3), point_cloud_np = (H, W, 3)
            rgb_np = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)  # (H, W, 3)
            rgb_tensor = torch.from_numpy(rgb_np).float().to(DEVICE)  # (H, W, 3)

            # 3d PCL

            pcl = pcl_generator.rgbd_to_pointcloud(is_l515=False,
                                                   color_image_path=rgb_path,
                                                   depth_image_path=depth_path,
                                                   intrinsic=CAM_INTRINSIC
                                                   )  # (N, 3), N is the number of point clouds

            pcl_tensor = torch.from_numpy(np.asarray(pcl.points)).float().to(DEVICE)
            # o3d.visualization.draw_geometries([pcl])

            pcd_img_coords, pcd_valid_mask = pcl_generator.project_pointcloud_to_image(pcl, CAM_INTRINSIC)

            voxel_grid, voxel_pcd = voxel_generator.voxel_generation(rgb_np, pcl_tensor, pcd_img_coords, pcd_valid_mask, DEVICE)

            # save file
            output_dir = Path(depth_image_path).parent.parent / 'voxel_grids_pt_size18'
            # If the directory exists, delete everything inside it
            # if output_dir.exists():
            #     shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            voxel_grid_path = output_dir / f"{timestamp}.pt"

            torch.save(voxel_grid, voxel_grid_path)

            # np.save(voxel_grid_path, voxel_grid.cpu().numpy())
            print(f"voxel_grid saved to: {voxel_grid_path}")

            # pdb.set_trace()
