import matplotlib.pyplot as plt

methods = [
    "Baseline",
    "No MLF",
    "No CBAM",
    "No DS",
    "No Warmup",
    "No Aug",
    "M-Net"
]

iou = [76.48, 76.68, 77.62, 77.24, 76.90, 78.18, 80.79]
dice = [86.41, 86.55, 87.18, 86.92, 86.72, 87.55, 89.21]

plt.figure(figsize=(10,5))

plt.plot(
    methods, iou,
    marker='o',
    linewidth=2,
    markersize=7,
    label='IoU'
)

plt.plot(
    methods, dice,
    marker='s',
    linewidth=2,
    markersize=7,
    label='Dice'
)

plt.ylabel('Score (%)', fontsize=12)
plt.xlabel('Ablation Variant', fontsize=12)
plt.title('Impact of Each Component on Segmentation Performance', fontsize=14)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# Optional: display values on points
for i, val in enumerate(iou):
    plt.text(i, val + 0.15, f"{val:.2f}", ha='center', fontsize=8)

for i, val in enumerate(dice):
    plt.text(i, val + 0.15, f"{val:.2f}", ha='center', fontsize=8)

plt.tight_layout()
plt.savefig("ablation_iou_dice.png", dpi=600, bbox_inches='tight')
plt.show()