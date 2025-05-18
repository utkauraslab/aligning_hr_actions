import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import random
import pandas as pd
from collections import Counter
from tqdm import tqdm
import time
import pdb

try:
    import seaborn as sns
except ImportError:
    print("⚠️ seaborn not found. Confusion matrix will be displayed without seaborn.")
    sns = None

try:
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
except ImportError:
    print("❗ scikit-learn is not installed. Please install it with `pip install scikit-learn`. Confusion matrix will not be shown.")
    confusion_matrix = ConfusionMatrixDisplay = None

# ------------ Perceiver Classifier ------------


class PerceiverClassifier(nn.Module):
    def __init__(self, input_dim=10, latent_dim=512, num_latents=128, num_classes=8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.cross_attn = nn.MultiheadAttention(latent_dim, num_heads=8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.ReLU(),
            # nn.Dropout(p=0.2),  # try 0.2 to 0.5
            nn.Linear(latent_dim // 2, num_classes)
        )

    def forward(self, x):

        # pdb.set_trace()
        B, N, C = x.shape

        x_proj = self.input_proj(x)
        latents = self.latents.unsqueeze(0).repeat(B, 1, 1)
        latents, _ = self.cross_attn(latents, x_proj, x_proj)
        latents = latents.mean(dim=1)
        out = self.mlp(latents)
        return out

# # ------------ Dataset ------------
class VoxelDataset(Dataset):
    def __init__(self, voxel_dirs, label_dict):
        self.voxel_files = []
        for path in voxel_dirs:
            self.voxel_files.extend(Path(path).glob("*.npy"))
        self.voxel_files = self.voxel_files  # Use full set
        self.label_dict = label_dict

        # Precompute labels for fast access
        self.labels = [self.label_dict[file.stem] for file in self.voxel_files]

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        file = self.voxel_files[idx]
        voxel = np.load(file)
        if voxel.ndim == 5:
            voxel = voxel[0]
        voxel = voxel.reshape(-1, voxel.shape[-1])
        max_points = 50000
        if voxel.shape[0] > max_points:
            indices = np.random.choice(voxel.shape[0], max_points, replace=False)
            voxel = voxel[indices]
        voxel = torch.tensor(voxel, dtype=torch.float32)
        label = self.labels[idx]  # ✅ Faster access from precomputed list

        pdb.set_trace()

        return voxel, torch.tensor(label, dtype=torch.long)

    def get_filename_and_label(self, idx):
        file = self.voxel_files[idx]
        label = self.labels[idx]  # ✅ Use precomputed label
        return file.name, label

    def get_filename(self, idx):
        return self.voxel_files[idx].name


# # ------------ Dataset PtFile ------------
class VoxelDatasetPtFile(Dataset):
    def __init__(self, pt_dirs, label_dict, max_points=50000):
        self.voxel_files = []
        for path in pt_dirs:
            self.voxel_files.extend(Path(path).glob("*.pt"))
        self.label_dict = label_dict
        self.max_points = max_points

        # Precompute labels for fast access
        self.labels = [self.label_dict[file.stem] for file in self.voxel_files]

    def __len__(self):
        return len(self.voxel_files)

    def __getitem__(self, idx):
        file = self.voxel_files[idx]
        voxel = torch.load(file)  # shape: [N, C] or something similar

        # ⚠️ Make sure it's [N, C], flatten batch if needed
        if voxel.ndim > 2:
            voxel = voxel.view(-1, voxel.shape[-1])

        # Limit number of points
        if voxel.shape[0] > self.max_points:
            indices = torch.randperm(voxel.shape[0])[:self.max_points]
            voxel = voxel[indices]

        label = self.labels[idx]

        # pdb.set_trace()

        return voxel, torch.tensor(label, dtype=torch.long)

    def get_filename_and_label(self, idx):
        file = self.voxel_files[idx]
        label = self.labels[idx]
        return file.name, label

    def get_filename(self, idx):
        return self.voxel_files[idx].name


# ------------ Training Loop ------------

def train(model, train_loader, val_loader, y_train, device, output_dir, epochs=1000, lr=1e-4):

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Learning rate scheduler: reduce LR when val loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=10,
        verbose=True
    )

    # Class weighting from y_train
    train_labels = y_train.tolist()
    class_counts = Counter(train_labels)
    weights = torch.tensor(
        [1 / (class_counts.get(i, 1e-5)) for i in range(8)],
        dtype=torch.float32,
        device=device
    )
    weights = weights / weights.sum()
    criterion = nn.CrossEntropyLoss(weight=weights)

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        start_time = time.time()

        model.train()
        total_loss = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            pred = model(x)
            if y.min() < 0 or y.max() >= 8:
                print("❗ Invalid label(s) found:", y)
                raise ValueError("Label out of bounds for CrossEntropyLoss")

            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # pdb.set_trace()

        train_losses.append(total_loss / len(train_loader))

        # Evaluation
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item()
                correct += (pred.argmax(dim=1) == y).sum().item()
                total += y.size(0)
                all_preds.extend(pred.argmax(dim=1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_losses.append(val_loss / len(val_loader))
        acc = correct / total
        print(f"Epoch {epoch + 1}: Train Loss = {train_losses[-1]:.4f}, Val Loss = {val_losses[-1]:.4f}, Val Acc = {acc:.2%}")

        with torch.no_grad():
            probs = torch.softmax(pred, dim=1)
            print("Sample prediction probs:", probs[0].cpu().numpy())

        # Save the model every 100 epochs
        if (epoch + 1) % 10 == 0:
            output_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = output_dir / f"model_epoch_{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model saved to {checkpoint_path}")

            train_loss_fig_path = output_dir / f"perceiver_loss_{epoch + 1}.png"
            plot_losses(train_losses, val_losses, train_loss_fig_path)

            np.save(output_dir / f"train_losses_{epoch + 1}.npy", train_losses)
            np.save(output_dir / f"val_losses_{epoch + 1}.npy", val_losses)

        # Update learning rate scheduler
        scheduler.step(val_losses[-1])

        elapsed = time.time() - start_time
        print(f"⏱️ Epoch {epoch + 1} took {elapsed:.2f} seconds")

    # Confusion matrix at the end
    if confusion_matrix:
        cm = confusion_matrix(all_labels, all_preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title("Validation Confusion Matrix")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

    return train_losses, val_losses


# ------------ Plot Loss ------------
def plot_losses(train_losses, val_losses, train_loss_fig_path="perceiver_loss.png"):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(train_loss_fig_path)
    # plt.show()
    plt.close()  # Optional: close the figure to free memory


def save_dataset_traning_info(dataset, train_set, val_set, save_path="dataset_info.txt"):

    # Get indices
    train_indices = train_set.indices
    val_indices = val_set.indices

    # Extract filename and label pairs
    train_data = [dataset.get_filename_and_label(i) for i in train_indices]
    val_data = [dataset.get_filename_and_label(i) for i in val_indices]

    save_path.mkdir(parents=True, exist_ok=True)  # Make sure the directory exists

    # Save train file
    train_txt_path = save_path / "train_data.txt"
    with open(train_txt_path, "w") as f:
        for fname, label in train_data:
            f.write(f"{fname}\t{label}\n")

    # Save val file
    val_txt_path = save_path / "val_data.txt"
    with open(val_txt_path, "w") as f:
        for fname, label in val_data:
            f.write(f"{fname}\t{label}\n")

    print(f"Train/val file lists saved to {save_path}")


def custom_train_val_split(dataset, val_txt_path, val_ratio_of_remaining=0.2, seed=42):
    """
    dataset: a VoxelDataset instance
    val_txt_path: path to the txt file containing filenames (with .npy) and labels (tab-separated)
    val_ratio_of_remaining: percentage of remaining samples (not in txt) to use for validation
    """
    # Step 1: Load fixed validation filenames from txt
    val_txt = Path(val_txt_path)
    with open(val_txt, "r") as f:
        fixed_val_filenames = set(line.strip().split("\t")[0] for line in f)

    # Step 2: Separate indices into fixed val and remaining
    fixed_val_indices = []
    other_indices = []

    for i in range(len(dataset)):
        fname = dataset.get_filename_and_label(i)[0]  # expects file.name like '123456.npy'
        # pdb.set_trace()

        if fname in fixed_val_filenames:
            fixed_val_indices.append(i)
        else:
            other_indices.append(i)

    # Step 3: Random split of the remaining
    random.seed(seed)
    random.shuffle(other_indices)

    split = int((1 - val_ratio_of_remaining) * len(other_indices))
    train_indices = other_indices[:split]
    val_indices = fixed_val_indices + other_indices[split:]

    # Step 4: Create subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    print(f"📊 Dataset split complete: {len(train_indices)} train, {len(val_indices)} val")
    return train_set, val_set


def load_full_voxel_dataset(dataset):
    x_list = []
    y_list = []

    for i in tqdm(range(len(dataset)), desc="📦 Preloading dataset"):
        x, y = dataset[i]
        x_list.append(x)
        y_list.append(y)

    x_tensor = torch.stack(x_list)  # Shape: [K, N, C]
    y_tensor = torch.tensor(y_list, dtype=torch.long)

    return x_tensor, y_tensor


def preload_subset(subset):
    x_list, y_list = [], []
    for i in tqdm(range(len(subset))):
        x, y = subset[i]
        # ⛔ Do NOT call x.to(device) here
        x_list.append(x.cpu())  # Explicitly ensure CPU
        y_list.append(y)
    return torch.stack(x_list), torch.tensor(y_list, dtype=torch.long)


# ------------ Main  ------------
if __name__ == '__main__':

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    print("👀 Preparing to load data...")
    # voxel_root = Path(r"E:\UTK\Reseach\Publication\Workshop\ICRA2025_workshop_dataset_HumanRobotCorr-main\dataset_depth\dataset_depth\robot\cam_104122061850\pick\0004")
    # voxel_dirs = [str(p / "voxel_grids") for p in voxel_root.glob("user_0002_scene_*/") if (p / "voxel_grids").exists()]
    # json_path = Path(r"E:\UTK\Reseach\Publication\Workshop\ICRA2025_workshop_dataset_HumanRobotCorr-main\dataset_depth\dataset_depth\robot\cam_104122061850\pick\label_dict.json")

    voxel_root = Path(r"/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/0004")
    json_path = Path(r"/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/label_dict_0004.json")
    testing_data_txt_path = "/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/data_robot_0004_test_renamed.txt"

    # voxel_dirs = [str(p / "voxel_grids") for p in voxel_root.glob("user_*/") if (p / "voxel_grids").exists()]
    # voxel_dirs = [str(p / "voxel_pt") for p in voxel_root.glob("user_*/") if (p / "voxel_pt").exists()]
    # voxel_dirs = [str(p / "voxel_grids_pt_size36") for p in voxel_root.glob("user_*/") if (p / "voxel_grids_pt_size36").exists()]
    voxel_dirs = [str(p / "voxel_grids_pt_size21") for p in voxel_root.glob("user_*/") if (p / "voxel_grids_pt_size21").exists()]
    # voxel_dirs = [str(p / "voxel_grids_pt_size18") for p in voxel_root.glob("user_*/") if (p / "voxel_grids_pt_size18").exists()]

    # Loading dataset
    with open(json_path, "r") as f:
        label_dict = json.load(f)

    # dataset = VoxelDataset(voxel_dirs, label_dict)
    dataset = VoxelDatasetPtFile(voxel_dirs, label_dict)

    train_set, val_set = custom_train_val_split(dataset, testing_data_txt_path, val_ratio_of_remaining=0.2)

    # x_tensor, y_tensor = load_full_voxel_dataset(dataset)
    x_train, y_train = preload_subset(train_set)
    x_val, y_val = preload_subset(val_set)
    # pdb.set_trace()

    # train_size = int(0.8 * len(dataset))
    # val_size = len(dataset) - train_size
    # train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    output_dir = Path('/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/weights_004/run2_grids_size21')
    save_dataset_traning_info(dataset, train_set, val_set, output_dir)

    print('loading train/validation data!')
    # train_loader = DataLoader(train_set, batch_size=10, shuffle=True)
    # val_loader = DataLoader(val_set, batch_size=10, shuffle=False)

    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=60,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=10,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Init model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = x_train.shape[-1]
    model = PerceiverClassifier(input_dim=input_dim).to(device)

    pdb.set_trace()

    print('going to train!')

    train_losses, val_losses = train(
        model,
        train_loader,
        val_loader,
        y_train,
        device,
        # input_dim=x_train.shape[-1],
        output_dir=output_dir,
        epochs=200,
        lr=1e-4
    )

    plot_losses(train_losses, val_losses)

    torch.save(model.state_dict(), "perceiver_model.pth")
    print("✅ Training complete and model saved.")
