import numpy as np
import open3d as o3d
import cv2
import os
import pandas as pd
import torch
import utilis
import matplotlib.pyplot as plt
from pathlib import Path

import pdb

# settings
CAMERAS_NUM = 1
VOXEL_SIZES = [100]  # 100x100x100 voxels
# NUM_LATENTS = 512 # PerceiverIO latents
# [x_min, y_min, z_min, x_max, y_max, z_max] - the metric volume to be voxelized
SCENE_BOUNDS = [-0.3, -0.5, 0.6, 0.7, 0.5, 1.6]

BATCH_SIZE = 1
# NUM_DEMOS = 8 # total number of training demonstrations to use while training PerAct
# NUM_TEST = 2 # episodes to evaluate on
IMAGE_H = 360
IMAGE_W = 640

CAM_INTRINSIC = np.array([[908.49682617/2, 0, 640.63098145/2],
                          [0, 907.65515137/2, 351.33010864/2],
                          [0, 0, 1]])

# Automatically use GPU if available, otherwise fallback to CPU
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# tensorize scene bounds
bounds = torch.tensor(SCENE_BOUNDS, device=device).unsqueeze(0)

# initialize voxelizer
vox_grid = utilis.VoxelGrid(
    coord_bounds=SCENE_BOUNDS,
    voxel_size=VOXEL_SIZES[0],
    device=device,
    batch_size=BATCH_SIZE,
    feature_size=3,
    max_num_coords=np.prod([IMAGE_H, IMAGE_W]) * CAMERAS_NUM,
)


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


def get_timestamps_from_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Get all entries in first column as strings
    timestamp = df.iloc[:, 0].astype(str).tolist()
    return timestamp


def _norm_rgb(x):
    return (x.float() / 255.0) * 2.0 - 1.0


def _preprocess_inputs(rgb, pcd):

    # pdb.set_trace()

    obs, pcds = [], []

    rgb_norm = _norm_rgb(rgb)

    # obs contains both rgb and pointcloud (used in ARM for other baselines)
    obs.append([rgb_norm, pcd])
    pcds.append(pcd)  # only pointcloud

    # pdb.set_trace()

    return obs, pcds


def get_colors_from_image(rgb_np: np.ndarray, img_coords: np.ndarray, valid_mask: np.ndarray):
    """
    Get RGB color for each valid projected point.

    Args:
        rgb_np: (H, W, 3) RGB image
        img_coords: (N, 2) pixel coordinates [u, v]
        valid_mask: (N,) valid projection mask

    Returns:
        colors: (M, 3) array of RGB colors for valid points
    """
    uvs = img_coords[valid_mask]  # shape: (M, 2)
    v = uvs[:, 1]
    u = uvs[:, 0]
    colors = rgb_np[v, u]  # (M, 3)

    return colors


def visualize_projected_points(
    rgb_np: np.ndarray,
    img_coords: np.ndarray,
    valid_mask: np.ndarray,
    dot_size: int = 0.2,
    color: str = 'red'
):
    """
    Visualize projected 3D points overlaid on the RGB image.

    Args:
        rgb_np: (H, W, 3) RGB image
        img_coords: (N, 2) array of 2D pixel coordinates (u, v)
        valid_mask: (N,) boolean mask of valid projected points
        dot_size: Size of the plotted dots
        color: Color of the overlay dots (default: red)
    """
    # Filter only valid pixel positions
    uvs = img_coords[valid_mask]

    # Visualize using matplotlib
    plt.figure(figsize=(10, 6))
    plt.imshow(rgb_np)
    plt.scatter(uvs[:, 0], uvs[:, 1], s=dot_size, c=color, marker='o')
    plt.title("Projected 3D Points on RGB Image")
    # plt.axis('off')
    # plt.tight_layout()
    plt.show()


def voxel_grid_to_pointcloud_aligned_dynamic(
    voxel_grid, bounds, dims_orig=(100, 100, 100)
):
    """
    Converts voxel grid to point cloud in world space using bounding box and voxel count.

    Args:
        voxel_grid: (1, D, H, W, C)
        bounds: (1, 6), [x_min, y_min, z_min, x_max, y_max, z_max]
        dims_orig: Tuple[int, int, int], number of voxels in each dimension

    Returns:
        open3d.geometry.PointCloud
    """
    assert voxel_grid.shape[0] == 1, "Only batch size 1 supported"
    vox_np = voxel_grid[0].cpu().numpy()
    D, H, W = vox_np.shape[:3]

    # Compute voxel size from bounding box
    if isinstance(bounds, torch.Tensor):
        bounds = bounds[0].cpu().numpy()
    else:
        bounds = bounds[0]

    bb_mins = bounds[:3]
    bb_maxs = bounds[3:]
    voxel_size = (bb_maxs - bb_mins) / np.array(dims_orig)  # (3,)

    # Get indices of occupied voxels
    occupancy = vox_np[..., -1] > 0.5
    z, y, x = np.nonzero(occupancy)
    voxel_indices = np.stack([z, y, x], axis=1)  # (N, 3)

    # Map to world coordinates
    centers_world = voxel_indices * voxel_size + bb_mins + (voxel_size / 2.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(centers_world)
    pcd.paint_uniform_color([0.0, 0.0, 0.0])  # Black
    return pcd


if __name__ == "__main__":

    # ----Configuration ----
    label_cvs_path = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004/Labelled_Robot_004.csv'
    image_root_folder = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot'

    # ---- Process ----
    timestamps = get_timestamps_from_csv(label_cvs_path)

    for timestamp in timestamps:
        image_filename = f"{timestamp}.jpg"
        depth_image_path, rgb_image_path, depth_v2_image_path = find_image_by_name(
            image_filename, image_root_folder)
        # if depth_image_path and rgb_image_path:
        if depth_v2_image_path and rgb_image_path:
            rgb_path = rgb_image_path
            # depth_path = depth_image_path
            depth_path = depth_v2_image_path
            print(f"RGB & Depth image: {image_filename} found!")

            # Assume: rgb_np = (H, W, 3), point_cloud_np = (H, W, 3)
            rgb_np = cv2.cvtColor(cv2.imread(rgb_path),
                                  cv2.COLOR_BGR2RGB)  # (H, W, 3)
            rgb_tensor = torch.from_numpy(rgb_np).float()  # (H, W, 3)

            # 3d PCL
            generator = utilis.RGBDPointCloudGenerator(
                min_depth_m=0.4, max_depth_m=1)
            pcd = generator.rgbd_to_pointcloud(
                is_l515=False,
                color_image_path=rgb_path,
                depth_image_path=depth_path,
                intrinsic=CAM_INTRINSIC
            )  # (N, 3), N is the number of point clouds

            pcd_tensor = torch.from_numpy(np.asarray(pcd.points)).float()
            # o3d.visualization.draw_geometries([pcd])

            # ================================================================================
            # following code with generate flat_imag_features for whole image into B*(K*H*W)*3
            # K is the image numbers, we keep it to 1 for now

            # obs, pcds = _preprocess_inputs(
            #     rgb_tensor.unsqueeze(0), pcd_tensor.unsqueeze(0))
            # pcd_flat = torch.cat(pcds, 1)
            # bs = obs[0][0].shape[0]
            # image_features = [o[0] for o in obs]
            # feat_size = image_features[0].shape[3]
            # flat_imag_features = torch.cat(
            #     [p.reshape(bs, -1, feat_size) for p in image_features], 1)

            # ================================================================================
            # following code with generate the 2D image points corresponding to the point cloud
            # K is the image numbers, we keep it to 1 for now
            pcd_img_coords, pcd_valid_mask = generator.project_pointcloud_to_image(
                pcd, CAM_INTRINSIC)
            # visualize_projected_points(rgb_np, pcd_img_coords, pcd_valid_mask)

            pcd_img_rgb = get_colors_from_image(
                rgb_np, pcd_img_coords, pcd_valid_mask)
            pcd_img_rgb_tensor = torch.tensor(pcd_img_rgb)
            obs, pcds = _preprocess_inputs(
                pcd_img_rgb_tensor.unsqueeze(0), pcd_tensor.unsqueeze(0))

            # flattern for point cloud
            pcd_flat = torch.cat(pcds, 1)

            # flattern for the corresponding 2D image points
            bs = obs[0][0].shape[0]
            image_features = [o[0] for o in obs]
            feat_size = image_features[0].shape[2]
            flat_imag_features = torch.cat(
                [p.reshape(bs, -1, feat_size) for p in image_features], 1)

            # pdb.set_trace()

            # voxelize!
            voxel_grid = vox_grid.coords_to_bounding_voxel_grid(pcd_flat,
                                                                coord_features=flat_imag_features,
                                                                coord_bounds=bounds)

            voxel_pcd = voxel_grid_to_pointcloud_aligned_dynamic(
                voxel_grid, bounds, dims_orig=(VOXEL_SIZES[0], VOXEL_SIZES[0], VOXEL_SIZES[0]))

            o3d.visualization.draw_geometries([pcd, voxel_pcd])

            # o3d.visualization.draw_geometries([pcd])
            # o3d.visualization.draw_geometries([voxel_pcd])

            # output_dir = Path(depth_image_path).parent.parent / 'point_clouds'
            # output_dir.mkdir(parents=True, exist_ok=True)
            # ply_path = output_dir / f"{timestamp}.ply"
            # o3d.io.write_point_cloud(ply_path, pcd, write_ascii=True)

            # output_dir = Path(depth_image_path).parent.parent / 'voxel_grids'
            # output_dir.mkdir(parents=True, exist_ok=True)
            # voxel_grid_path = output_dir / f"{timestamp}.npy"
            # np.save(voxel_grid_path, voxel_grid)

            # print(f"voxel_grid saved to: {voxel_grid_path}")

            pdb.set_trace()

        else:
            print(f"Missing one or both images for timestamp: {timestamp}!")
