import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def convert_npy_to_pt(voxel_dirs, max_points=50000, pt_folder_name="voxel_pt"):
    """
    Convert .npy voxel files to .pt format and save them in corresponding mirrored directories.
    """
    for voxel_dir in voxel_dirs:
        voxel_dir = Path(voxel_dir)
        output_dir = voxel_dir.parent / pt_folder_name
        output_dir.mkdir(parents=True, exist_ok=True)

        npy_files = list(voxel_dir.glob("*.npy"))
        for npy_file in tqdm(npy_files, desc=f"Converting {voxel_dir}"):
            try:
                voxel = np.load(npy_file)
                if voxel.ndim == 5:
                    voxel = voxel[0]
                voxel = voxel.reshape(-1, voxel.shape[-1])
                if voxel.shape[0] > max_points:
                    indices = np.random.choice(voxel.shape[0], max_points, replace=False)
                    voxel = voxel[indices]
                voxel_tensor = torch.tensor(voxel, dtype=torch.float32)

                output_path = output_dir / (npy_file.stem + ".pt")
                torch.save(voxel_tensor, output_path)
            except Exception as e:
                print(f"Failed to process {npy_file.name}: {e}")


voxel_root = Path(r"/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004")
voxel_dirs = [str(p / "voxel_grids") for p in voxel_root.glob("user_*/") if (p / "voxel_grids").exists()]


convert_npy_to_pt(voxel_dirs)
