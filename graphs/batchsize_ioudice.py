import matplotlib.pyplot as plt

# Batch Sizes
batch_sizes = [2, 4, 8]

# Test Metrics
test_iou = [
    0.7636766708,
    0.7872275826,
    0.8064905179
]

test_dice = [
    0.8591387383,
    0.8783727277,
    0.8913140153
]

# Plot
plt.figure(figsize=(8, 5))

plt.plot(
    batch_sizes,
    test_iou,
    marker='o',
    linewidth=2,
    markersize=8,
    label='Test IoU'
)

plt.plot(
    batch_sizes,
    test_dice,
    marker='s',
    linewidth=2,
    markersize=8,
    label='Test Dice'
)

plt.xlabel('Batch Size')
plt.ylabel('Score')
plt.title('Effect of Batch Size on Test IoU and Dice')
plt.xticks(batch_sizes)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()