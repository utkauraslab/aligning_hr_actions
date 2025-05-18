import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

result_path = '/home/fei/Documents/Dataset/icra25_align_human_robot/labeled/robot/cam_104122061850/pick/weights_004/run1_grids_size21'

# Shared class names (target order)
class_names = ["reaching", "grasping", "lifting", "holding", "transporting", "placing", "releasing", "nothing"]

# Human confusion matrix (already in correct order)
conf_human = np.array([
    [24, 3,  0,  0, 0, 0, 0, 10],  # reaching
    [7,  3,  3,  0, 0, 0, 0,  0],  # grasping
    [2, 12, 4,  4, 1, 0, 0,  1],  # lifting
    [0,  2, 5, 10, 0, 0, 0,  0],  # holding
    [0,  0, 0,  0, 5, 2, 0,  1],  # transporting
    [0,  0, 0,  0, 0, 5, 1,  1],  # placing
    [0,  0, 0,  0, 0, 0, 1,  2],  # releasing
    [27, 0, 0,  0, 0, 0, 0, 97],  # nothing
])

# Robot confusion matrix (in different label order)
# Robot label_map: 0=nothing, 1=reaching, 2=grasping, ..., 7=releasing
conf_robot_raw = np.array([
    [543,  76,   5,   4,  42,  17,  17,  11],  # nothing
    [74, 497,  50,  37,  41,  17,  13,   5],  # reaching
    [2,  15,  67,   5,   1,   0,   2,   0],  # grasping
    [0,  11,  12,  34,   4,   2,   2,   3],  # lifting
    [11,  26,   3,   5, 142,  18,   9,   6],  # transporting
    [14,   8,   0,   7,  22, 110,   3,   3],  # holding
    [2,   5,   0,   1,   4,   1,  52,   5],  # placing
    [3,   1,   0,   0,   1,   2,   6,   5],  # releasing
])

# Mapping robot indices to correct order
robot_idx_map = {
    'reaching': 1,
    'grasping': 2,
    'lifting': 3,
    'holding': 5,
    'transporting': 4,
    'placing': 6,
    'releasing': 7,
    'nothing': 0
}

# Build reorder index list
robot_order = [robot_idx_map[cls] for cls in class_names]
conf_robot = conf_robot_raw[np.ix_(robot_order, robot_order)]  # reorder rows and cols

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

sns.heatmap(conf_human, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
axes[0].set_title("Human")
axes[0].set_xlabel("Predicted label")
axes[0].set_ylabel("True label")
axes[0].tick_params(axis='x', rotation=45)

sns.heatmap(conf_robot, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=False, ax=axes[1], cbar=True)
axes[1].set_title("Robot")
axes[1].set_xlabel("Predicted label")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(result_path + "/human_robot_confusion_matrix.pdf", format="pdf")
plt.show()
