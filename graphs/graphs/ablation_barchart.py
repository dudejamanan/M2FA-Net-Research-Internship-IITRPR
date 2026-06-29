import numpy as np
import matplotlib.pyplot as plt

# Methods
methods = [
    "Baseline\n(Early Fusion)",
    "No Multi-\nLevel Fusion",
    "No CBAM\nAttention",
    "No Deep\nSupervision",
    "No Warmup\nScheduler",
    "No\nAugmentation",
    "M-Net\n(Proposed)"
]

# Mean values
accuracy = [92.07, 92.66, 93.09, 92.98, 92.90, 93.28, 93.81]
precision = [89.07, 89.49, 90.56, 89.44, 89.86, 89.50, 90.37]
recall = [78.24, 80.14, 81.02, 80.88, 81.08, 82.55, 84.63]
iou = [72.16, 74.08, 75.15, 74.47, 74.71, 75.87, 78.69]
dice = [83.16, 84.31, 85.29, 84.77, 85.08, 85.91, 87.61]

# Standard deviations
acc_std = [0.14, 0.15, 0.12, 0.14, 0.10, 0.08, 0.06]
prec_std = [0.52, 1.09, 0.62, 0.48, 2.27, 0.66, 0.42]
rec_std = [0.57, 1.92, 1.02, 0.82, 2.64, 0.73, 0.68]
iou_std = [0.42, 0.77, 0.45, 0.47, 0.60, 0.38, 0.21]
dice_std = [0.28, 0.63, 0.30, 0.35, 0.44, 0.25, 0.15]

metrics = [accuracy, precision, recall, iou, dice]
errors = [acc_std, prec_std, rec_std, iou_std, dice_std]
metric_names = ["Accuracy", "Precision", "Recall", "IoU", "Dice"]

# Plot
x = np.arange(len(methods))
width = 0.15

fig, ax = plt.subplots(figsize=(14, 6))

for i, (metric, err) in enumerate(zip(metrics, errors)):
    ax.bar(
        x + i * width,
        metric,
        width,
        label=metric_names[i],
        yerr=err,
        capsize=3
    )

ax.set_ylabel("Score (%)", fontsize=12)
ax.set_xlabel("Ablation Variants", fontsize=12)
ax.set_title("Ablation Study Results", fontsize=14, fontweight='bold')

ax.set_xticks(x + 2 * width)
ax.set_xticklabels(methods)

ax.legend(ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.12))
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("ablation_study_barplot.png", dpi=600, bbox_inches='tight')
plt.show()