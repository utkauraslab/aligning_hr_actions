import numpy as np
import matplotlib.pyplot as plt

result_path = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/weights_004/run1_grids_size21'

# Load saved losses
train_losses = np.load(result_path + "/train_losses_200.npy")[0:150]
val_losses = np.load(result_path + "/val_losses_200.npy")[0:150]
epochs = np.arange(1, len(train_losses) + 1)

# Marker interval
interval = 10

# Plot setup
plt.figure(figsize=(4, 2))  # square plot

# Plot full line, but add markers only every `interval` steps
plt.plot(epochs, train_losses, color='gray', linewidth=0.8, label='train')
plt.plot(epochs[::interval], train_losses[::interval], 'o', color='gray', markersize=4)

plt.plot(epochs, val_losses, color='darkred', linestyle='--', linewidth=0.8, label='validation')
plt.plot(epochs[::interval], val_losses[::interval], 's', color='darkred', markersize=4)


# Axis labels and ticks
plt.xlabel("Training epochs (robot)", fontsize=10)
plt.ylabel("Loss (Units)", fontsize=10)
plt.xticks(fontsize=5)
plt.yticks(fontsize=5)

# Legend
plt.legend(frameon=False, fontsize=9)

# Save to PDF
plt.tight_layout(pad=0.2)
plt.savefig(result_path + "/loss_curve_compact.pdf", format='pdf')
plt.close()
