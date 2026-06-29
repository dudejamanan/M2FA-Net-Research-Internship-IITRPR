import matplotlib.pyplot as plt
import numpy as np

components = [
    "Multi-Level\nFusion",
    "CBAM\nAttention",
    "Deep\nSupervision",
    "Warmup\nScheduler",
    "Data\nAugmentation"
]

contributions = [4.61, 3.54, 4.22, 3.98, 2.82]

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
plt.title("Contribution of Each Component to M-Net", fontsize=14)
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