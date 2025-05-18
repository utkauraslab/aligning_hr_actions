# From https://github.com/stepjam/ARM/blob/main/arm/c2farm/voxel_grid.py

from functools import reduce as funtool_reduce
from operator import mul

import torch
from torch import nn, einsum
import torch.nn.functional as F

import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

import pdb


MIN_DENOMINATOR = 1e-12
INCLUDE_PER_VOXEL_COORD = False


class VoxelGrid(nn.Module):

    def __init__(self,
                 coord_bounds,
                 voxel_size: int,
                 device,
                 batch_size,
                 feature_size,
                 max_num_coords: int,):
        super(VoxelGrid, self).__init__()
        self._device = device
        self._voxel_size = voxel_size
        self._voxel_shape = [voxel_size] * 3
        self._voxel_d = float(self._voxel_shape[-1])
        self._voxel_feature_size = 4 + feature_size
        self._voxel_shape_spec = torch.tensor(self._voxel_shape,
                                              device=device).unsqueeze(
            0) + 2  # +2 because we crop the edges.
        self._coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
                                          device=device).unsqueeze(0)
        max_dims = self._voxel_shape_spec[0]
        self._total_dims_list = torch.cat(
            [torch.tensor([batch_size], device=device), max_dims,
             torch.tensor([4 + feature_size], device=device)], -1).tolist()
        self._ones_max_coords = torch.ones((batch_size, max_num_coords, 1),
                                           device=device)
        self._num_coords = max_num_coords

        shape = self._total_dims_list

        self._result_dim_sizes = torch.tensor(
            [funtool_reduce(mul, shape[i + 1:], 1) for i in range(len(shape) - 1)] + [
                1], device=device)
        flat_result_size = funtool_reduce(mul, shape, 1)

        self._initial_val = torch.tensor(0, dtype=torch.float,
                                         device=device)
        self._flat_output = torch.ones(flat_result_size, dtype=torch.float,
                                       device=device) * self._initial_val
        self._arange_to_max_coords = torch.arange(4 + feature_size,
                                                  device=device)
        self._flat_zeros = torch.zeros(flat_result_size, dtype=torch.float,
                                       device=device)

        self._const_1 = torch.tensor(1.0, device=device)
        self._batch_size = batch_size

        # Coordinate Bounds:
        self._bb_mins = self._coord_bounds[..., 0:3]
        bb_maxs = self._coord_bounds[..., 3:6]
        bb_ranges = bb_maxs - self._bb_mins
        # get voxel dimensions. 'DIMS' mode
        self._dims = dims = self._voxel_shape_spec.int()
        self._dims_orig = dims_orig = self._voxel_shape_spec.int() - 2
        self._dims_m_one = (dims - 1).int()
        # BS x 1 x 3
        self._res = bb_ranges / (dims_orig.float() + MIN_DENOMINATOR)
        self._res_minis_2 = bb_ranges / (dims.float() - 2 + MIN_DENOMINATOR)

        self._voxel_indicy_denmominator = self._res + MIN_DENOMINATOR
        self._dims_m_one_zeros = torch.zeros_like(self._dims_m_one)

        batch_indices = torch.arange(self._batch_size, dtype=torch.int,
                                     device=device).view(self._batch_size, 1, 1)
        self._tiled_batch_indices = batch_indices.repeat(
            [1, self._num_coords, 1])

        w = self._voxel_shape[0] + 2

        arange = torch.arange(0, w, dtype=torch.float, device=device)
        self._index_grid = torch.cat([
            arange.view(w, 1, 1, 1).repeat([1, w, w, 1]),
            arange.view(1, w, 1, 1).repeat([w, 1, w, 1]),
            arange.view(1, 1, w, 1).repeat([w, w, 1, 1])], dim=-1).unsqueeze(
            0).repeat([self._batch_size, 1, 1, 1, 1])

    def _broadcast(self, src: torch.Tensor, other: torch.Tensor, dim: int):
        if dim < 0:
            dim = other.dim() + dim
        if src.dim() == 1:
            for _ in range(0, dim):
                src = src.unsqueeze(0)
        for _ in range(src.dim(), other.dim()):
            src = src.unsqueeze(-1)
        src = src.expand_as(other)
        return src

    def _scatter_mean(self, src: torch.Tensor, index: torch.Tensor, out: torch.Tensor,
                      dim: int = -1):
        out = out.scatter_add_(dim, index, src)

        index_dim = dim
        if index_dim < 0:
            index_dim = index_dim + src.dim()
        if index.dim() <= index_dim:
            index_dim = index.dim() - 1

        ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
        out_count = torch.zeros(out.size(), dtype=out.dtype, device=out.device)
        out_count = out_count.scatter_add_(index_dim, index, ones)
        out_count.clamp_(1)
        count = self._broadcast(out_count, out, dim)
        if torch.is_floating_point(out):
            out.true_divide_(count)
        else:
            out.floor_divide_(count)
        return out

    def _scatter_nd(self, indices, updates):
        indices_shape = indices.shape
        num_index_dims = indices_shape[-1]
        flat_updates = updates.view((-1,))
        indices_scales = self._result_dim_sizes[0:num_index_dims].view(
            [1] * (len(indices_shape) - 1) + [num_index_dims])
        indices_for_flat_tiled = ((indices * indices_scales).sum(
            dim=-1, keepdims=True)).view(-1, 1).repeat(
            *[1, self._voxel_feature_size])

        implicit_indices = self._arange_to_max_coords[
            :self._voxel_feature_size].unsqueeze(0).repeat(
            *[indices_for_flat_tiled.shape[0], 1])
        indices_for_flat = indices_for_flat_tiled + implicit_indices
        flat_indices_for_flat = indices_for_flat.view((-1,)).long()

        flat_scatter = self._scatter_mean(
            flat_updates, flat_indices_for_flat,
            out=torch.zeros_like(self._flat_output))
        return flat_scatter.view(self._total_dims_list)

    def coords_to_bounding_voxel_grid(self, coords, coord_features=None):
        voxel_indicy_denmominator = self._voxel_indicy_denmominator
        res, bb_mins = self._res, self._bb_mins

        if self._coord_bounds is not None:

            # coord_bounds = torch.tensor(coord_bounds, dtype=torch.float,
            #                             device=self._device).unsqueeze(0)

            bb_mins = self._coord_bounds[..., 0:3]
            bb_maxs = self._coord_bounds[..., 3:6]
            bb_ranges = bb_maxs - bb_mins
            res = bb_ranges / (self._dims_orig.float() + MIN_DENOMINATOR)
            voxel_indicy_denmominator = res + MIN_DENOMINATOR

        # pdb.set_trace()

        bb_mins_shifted = bb_mins - res  # shift back by one
        floor = torch.floor(
            (coords - bb_mins_shifted.unsqueeze(1)) / voxel_indicy_denmominator.unsqueeze(1)).int()
        voxel_indices = torch.min(floor, self._dims_m_one)
        voxel_indices = torch.max(voxel_indices, self._dims_m_one_zeros)

        # global-coordinate point cloud (x, y, z)
        voxel_values = coords

        # rgb values (R, G, B)
        if coord_features is not None:
            # concat rgb values (B, 128, 128, 3)
            voxel_values = torch.cat([voxel_values, coord_features], -1)

        # coordinates to aggregate over
        _, num_coords, _ = voxel_indices.shape
        all_indices = torch.cat([
            self._tiled_batch_indices[:, :num_coords], voxel_indices], -1)

        # max coordinates
        voxel_values_pruned_flat = torch.cat(
            [voxel_values, self._ones_max_coords[:, :num_coords]], -1)

        # aggregate across camera views
        scattered = self._scatter_nd(
            all_indices.view([-1, 1 + 3]),
            voxel_values_pruned_flat.view(-1, self._voxel_feature_size))

        vox = scattered[:, 1:-1, 1:-1, 1:-1]
        if INCLUDE_PER_VOXEL_COORD:
            res_expanded = res.unsqueeze(1).unsqueeze(1).unsqueeze(1)
            res_centre = (res_expanded * self._index_grid) + res_expanded / 2.0
            coord_positions = (res_centre + bb_mins_shifted.unsqueeze(
                1).unsqueeze(1).unsqueeze(1))[:, 1:-1, 1:-1, 1:-1]
            vox = torch.cat(
                [vox[..., :-1], coord_positions, vox[..., -1:]], -1)

        # occupied value
        occupied = (vox[..., -1:] > 0).float()
        vox = torch.cat([
            vox[..., :-1], occupied], -1)

        # hard voxel-location position encoding
        return torch.cat(
            [vox[..., :-1], self._index_grid[:, :-2, :-2, :-2] / self._voxel_d,
             vox[..., -1:]], -1)


class RGBDPointCloudGenerator:
    def __init__(self, min_depth_m=0.03, max_depth_m=0.8, width=640, height=360):
        self.min_depth_m = min_depth_m
        self.max_depth_m = max_depth_m
        self.width = width
        self.height = height

    def rgbd_to_pointcloud(
        self,
        is_l515: bool,
        color_image_path: str,
        depth_image_path: str,
        intrinsic: np.ndarray,
        extrinsic: np.ndarray = np.eye(4),
        downsample_factor: float = 1.0
    ):
        # Load images
        color = cv2.cvtColor(cv2.imread(color_image_path), cv2.COLOR_BGR2RGB)
        # depth = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED).astype(
        #     np.float32)[0:color.shape[0], :, :]

        # Load depth (grayscale 8-bit), match RGB height
        # depth_raw = cv2.imread(
        #     depth_image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        depth_raw = cv2.imread(
            depth_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
        depth = depth_raw[0:color.shape[0], :]

        # pdb.set_trace()

        # Downsample
        new_width = int(self.width / downsample_factor)
        new_height = int(self.height / downsample_factor)
        color = cv2.resize(color, (new_width, new_height)).astype(np.uint8)
        depth = cv2.resize(depth, (new_width, new_height))

        min_depth_m = 0.4
        max_depth_m = 1.2
        depth = depth / 255.0 * (max_depth_m - min_depth_m) + min_depth_m

        # Convert depth to meters
        # depth /= 4000.0 if is_l515 else 1000.0

        # pdb.set_trace()

        # depth[(depth < self.min_depth_m) | (depth > self.max_depth_m)] = 0

        # Convert to Open3D image formats
        color_o3d = o3d.geometry.Image(color)
        depth_o3d = o3d.geometry.Image(depth)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color_o3d, depth_o3d, depth_scale=1.0, convert_rgb_to_intensity=False)

        # Build intrinsics
        fx = intrinsic[0, 0] / downsample_factor
        fy = intrinsic[1, 1] / downsample_factor
        cx = intrinsic[0, 2] / downsample_factor
        cy = intrinsic[1, 2] / downsample_factor
        intrinsic_o3d = o3d.camera.PinholeCameraIntrinsic(
            new_width, new_height, fx, fy, cx, cy)

        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic_o3d, extrinsic)

        return pcd

    def project_pointcloud_to_image(
            self,
            pcd: o3d.geometry.PointCloud,
            intrinsic: np.ndarray,
            extrinsic: np.ndarray = np.eye(4),
            downsample_factor: float = 1.0
    ):
        """
        Projects 3D point cloud into 2D image pixel coordinates using intrinsics & extrinsics.

        Returns:
            img_coords: (N, 2) array of [u, v] pixel coordinates
            valid_mask: (N,) boolean mask for valid, in-image projections
        """
        # Inverse of camera pose: world → camera
        camera_pose_inv = np.linalg.inv(extrinsic)

        # Convert 3D points to camera frame
        points = np.asarray(pcd.points)  # (N, 3)
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])  # (N, 4)
        points_cam = (camera_pose_inv @ points_h.T).T[:, :3]  # (N, 3)

        # Extract intrinsics
        fx = intrinsic[0, 0] / downsample_factor
        fy = intrinsic[1, 1] / downsample_factor
        cx = intrinsic[0, 2] / downsample_factor
        cy = intrinsic[1, 2] / downsample_factor

        # Projection
        z = points_cam[:, 2]
        valid_mask = z > 0
        x = points_cam[valid_mask, 0]
        y = points_cam[valid_mask, 1]
        z = z[valid_mask]

        u = (fx * x / z + cx).astype(np.int32)
        v = (fy * y / z + cy).astype(np.int32)

        # Image bounds
        h = int(self.height / downsample_factor)
        w = int(self.width / downsample_factor)
        in_bounds = (u >= 0) & (u < w) & (v >= 0) & (v < h)

        # Final valid [u, v] pairs
        final_u = u[in_bounds]
        final_v = v[in_bounds]

        # Create full img_coords array and mask
        img_coords = np.zeros((points.shape[0], 2), dtype=np.int32)
        full_valid = np.zeros(points.shape[0], dtype=bool)
        full_valid[valid_mask] = in_bounds
        img_coords[full_valid] = np.stack([final_u, final_v], axis=1)

        return img_coords, full_valid


class VoxelGenerator:
    def __init__(self, scene_bounds, voxel_sizes, batch_size, image_size, cam_num, device):
        self.scene_bounds = scene_bounds
        self.voxel_sizes = voxel_sizes[0]
        self.batch_size = batch_size
        self.image_h = image_size[0]
        self.image_w = image_size[1]
        self.cam_num = cam_num
        self.device = device

        # pdb.set_trace()

        # Initialize VoxelGrid for each voxel size
        self.vox_grid = VoxelGrid(
            coord_bounds=self.scene_bounds,
            voxel_size=self.voxel_sizes,
            device=device,
            batch_size=self.batch_size,
            feature_size=3,
            max_num_coords=np.prod([self.image_h, self.image_w]) * self.cam_num,
        )

        self.generator = RGBDPointCloudGenerator(
            min_depth_m=0.4, max_depth_m=1)

    def _norm_rgb(self, x):
        return (x.float() / 255.0) * 2.0 - 1.0

    def _preprocess_inputs(self, rgb, pcd):

        # pdb.set_trace()

        obs, pcds = [], []

        rgb_norm = self._norm_rgb(rgb)

        # obs contains both rgb and pointcloud (used in ARM for other baselines)
        obs.append([rgb_norm, pcd])
        pcds.append(pcd)  # only pointcloud

        # pdb.set_trace()

        return obs, pcds

    def get_colors_from_image(self, rgb_np: np.ndarray, img_coords: np.ndarray, valid_mask: np.ndarray):
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

    def visualize_projected_points(self,
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

    def voxel_grid_to_pointcloud_aligned_dynamic(self,
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

        # pdb.set_trace()

        # Compute voxel size from bounding box
        if isinstance(bounds, torch.Tensor):
            bounds = bounds[0].cpu().numpy()
        else:
            bounds = np.array(bounds)

        bb_mins = bounds[:3]
        bb_maxs = bounds[3:]

        voxel_size = (bb_maxs - bb_mins) / np.array(dims_orig)  # (3,)

        # Get indices of occupied voxels
        occupancy = vox_np[..., -1] > 0.5
        z, y, x = np.nonzero(occupancy)
        voxel_indices = np.stack([z, y, x], axis=1)  # (N, 3)

        # Map to world coordinates
        centers_world = voxel_indices * \
            voxel_size + bb_mins + (voxel_size / 2.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(centers_world)
        pcd.paint_uniform_color([0.0, 0.0, 0.0])  # Black
        return pcd

    def voxel_generation(self, rgb_np, pcl_tensor, pcd_img_coords, pcd_valid_mask, device):

        pcd_img_rgb = self.get_colors_from_image(rgb_np, pcd_img_coords, pcd_valid_mask)
        pcd_img_rgb_tensor = torch.tensor(pcd_img_rgb, dtype=torch.float, device=device)
        obs, pcds = self._preprocess_inputs(pcd_img_rgb_tensor.unsqueeze(0), pcl_tensor.unsqueeze(0))

        # pdb.set_trace()

        # flattern for point cloud
        pcd_flat = torch.cat(pcds, 1)

        # flattern for the corresponding 2D image points
        bs = obs[0][0].shape[0]
        image_features = [o[0] for o in obs]
        feat_size = image_features[0].shape[2]
        flat_imag_features = torch.cat([p.reshape(bs, -1, feat_size) for p in image_features], 1)

        # flat
        # voxelize!
        voxel_grid = self.vox_grid.coords_to_bounding_voxel_grid(pcd_flat, coord_features=flat_imag_features)

        voxel_pcd = self.voxel_grid_to_pointcloud_aligned_dynamic(voxel_grid,
                                                                  self.scene_bounds,
                                                                  dims_orig=(self.voxel_sizes, self.voxel_sizes, self.voxel_sizes)
                                                                  )

        # o3d.visualization.draw_geometries([pcd_tensor, voxel_pcd])

        return voxel_grid, voxel_pcd
