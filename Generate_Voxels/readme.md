## 🛠 How to Modify

- Check the required packages in `requirements.txt`
- Modify the following lines in `pcl_voxelization.py`:
  - **Line 189**: Set your `label_cvs_path`
  - **Line 190**: Set your `image_root_folder`

---


## 🚀 How to Run

```
python3 pcl_voxelization.py
```

- This script will compute a voxel grid and store it in a variable named `voxel_grid` (defined around **line 267**).
- The shape of `voxel_grid` is: `torch.Size([1, 100, 100, 100, 10])`, representing:
  - `B x D x H x W x C`, where:
    - `B`: batch size
    - `D/H/W`: voxel grid resolution (e.g., 100³)
    - `C`: feature dimension (e.g., 10)
- To visualize the resulting voxel grid along with the original point cloud, run:

```
o3d.visualization.draw_geometries([pcd, voxel_pcd])
```
- Refer to `example_voxels_pcl.png` for a visual reference: colored points represent the original point cloud, while black points indicate voxel centers.


---

## 🌊 How to Generate New Depth Images

This code works **ONLY FOR NOW** with high-quality depth images. The original ones **CANNOT** yield good voxelizations, so we recommend using **Depth Anything V2** to generate improved depth maps.

### Steps:

1. **Clone Depth Anything V2**:

```
git clone https://github.com/DepthAnything/Depth-Anything-V2
```

2. **Download their pretrained model**  
   - Use the [small model](https://github.com/DepthAnything/Depth-Anything-V2?tab=readme-ov-file#pre-trained-models) — it already gives good results.

3. Place everything inside your `Generate_Voxels/` directory.

4. Run the following to generate new depth images:

```
python3 get_depth_images.py
```

## 📁 Required Files and Directory Structure

Make sure your directory structure under `Generate_Voxels/` looks like this:

```
Generate_Voxels/
├── DAV2/                # The cloned Depth Anything V2 repo (rename the folder to 'DAV2')
├── checkpoints/         # The folder containing the pretrained model files
├── get_depth_images.py  # Script to generate new depth images
├── pcl_voxelization.py  # Main voxelization script
├── utilis.py            # Custom utility functions used by the scripts
```

