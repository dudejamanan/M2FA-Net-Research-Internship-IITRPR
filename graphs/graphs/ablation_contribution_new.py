import matplotlib.pyplot as plt
import numpy as np

components = [
    "Multi-Level\nFusion",
    "CBAM\nAttention",
    "Deep\nSupervision",
    "Warmup\nScheduler",
    "Data\nAugmentation"
]

# IoU contribution = IoU(M-Net) - IoU(without component)
contributions = [4.11, 3.17, 3.55, 3.89, 2.61]

# Sort descending
idx = np.argsort(contributions)[::1]
components = np.array(components)[idx]
contributions = np.array(contributions)[idx]

plt.figure(figsize=(8,5))

bars = plt.bar(
    components,
    contributions,
    edgecolor='black'
)

plt.ylabel("IoU Contribution (%)", fontsize=12)
plt.xlabel("Component", fontsize=12)
plt.title("Contribution of Each Component", fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.4)

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 0.05,
        f"{height:.2f}",
        ha='center'
    )

plt.tight_layout()
plt.savefig("component_contribution.png", dpi=600)
plt.show()