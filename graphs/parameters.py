import matplotlib.pyplot as plt

models = [
    "DenseUNet",
    "HRNet",
    "PSPNet",
    "UNet",
    "CRF-UNet",
    "UNet++",
    "DeepLabV3",
    "ViT",
    "SegFormer",
    "SwinCNN",
    "CNN+Transformer",
    "MNet (Proposed)"
]

params = [
    17.6,
    42.8,
    48.0,
    48.8,
    48.8,
    53.0,
    79.2,
    1.24,
    3.7,
    62.0,
    92.0,
    64.3
]

# Colors
colors = ["skyblue"] * (len(models)-1) + ["crimson"]

fig, ax = plt.subplots(figsize=(12, 7))

bars = ax.barh(models, params, color=colors)

# Value labels
for bar in bars:
    width = bar.get_width()
    ax.text(
        width + 1,
        bar.get_y() + bar.get_height()/2,
        f"{width:.2f}M",
        va="center",
        fontsize=9
    )

# Category separators
ax.axhline(6.5, color='gray', linestyle='--', alpha=0.6)
ax.axhline(8.5, color='gray', linestyle='--', alpha=0.6)
ax.axhline(10.5, color='gray', linestyle='--', alpha=0.6)

# Category labels outside plot
ax.text(
    1.02, 0.22, "CNN Models",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold"
)

ax.text(
    1.02, 0.53, "Transformer Models",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold"
)

ax.text(
    1.02, 0.68, "Hybrid Models",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold"
)

ax.text(
    1.02, 0.92, "Proposed",
    transform=ax.transAxes,
    fontsize=12,
    fontweight="bold",
    color="crimson"
)

ax.set_xlabel("Parameters (Millions)")
ax.set_ylabel("Models")
ax.set_title("Comparison of Model Complexity Based on Parameter Count")

ax.grid(axis='x', linestyle='--', alpha=0.5)

# Extra room on right
plt.subplots_adjust(right=0.82)

plt.savefig(
    "parameter_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()