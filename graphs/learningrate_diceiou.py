import matplotlib.pyplot as plt
import numpy as np

# Learning Rates
learning_rates = np.array([
    1e-5, 5e-5, 1e-4, 1.5e-4, 2e-4,
    3e-4, 4e-4, 5e-4, 1e-3, 5e-3
])

# Test Metrics
test_iou = [
    0.7382874211, 0.7806951683, 0.7889076859, 0.7894539528, 0.7902707286,
    0.7894729975, 0.7885341727, 0.7886382363, 0.7899846623, 0.7706699379
]

test_dice = [
    0.8439905872, 0.8739742353, 0.8795285173, 0.8799117622, 0.8803951118,
    0.8799616636, 0.8792765783, 0.8795977264, 0.8804675241, 0.8674022821
]

# Plot
plt.figure(figsize=(8, 5))

plt.plot(
    learning_rates,
    test_iou,
    marker='o',
    linewidth=2,
    markersize=7,
    label='Test IoU'
)

plt.plot(
    learning_rates,
    test_dice,
    marker='s',
    linewidth=2,
    markersize=7,
    label='Test Dice'
)

plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Score')
plt.title('Effect of Learning Rate on Test IoU and Dice')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.tight_layout()

plt.show()