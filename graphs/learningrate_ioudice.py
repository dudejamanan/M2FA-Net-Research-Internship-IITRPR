import matplotlib.pyplot as plt
import numpy as np

# Learning Rates
learning_rates = np.array([
    1e-5, 5e-5, 1e-4, 1.5e-4, 2e-4,
    3e-4, 4e-4, 5e-4, 1e-3, 5e-3
])

# IoU Scores
val_iou = [
    0.7449870387, 0.7888232547, 0.7935693707, 0.7919097706, 0.7927889138,
    0.7940634486, 0.7922592960, 0.7915803858, 0.7936857656, 0.7689883677
]

test_iou = [
    0.7382874211, 0.7806951683, 0.7889076859, 0.7894539528, 0.7902707286,
    0.7894729975, 0.7885341727, 0.7886382363, 0.7899846623, 0.7706699379
]

# Dice Scores
val_dice = [
    0.8497477328, 0.8792563675, 0.8823898247, 0.8811858642, 0.8819100585,
    0.8825963644, 0.8816855368, 0.8811324619, 0.8829768947, 0.8662208696
]

test_dice = [
    0.8439905872, 0.8739742353, 0.8795285173, 0.8799117622, 0.8803951118,
    0.8799616636, 0.8792765783, 0.8795977264, 0.8804675241, 0.8674022821
]

# -------------------------
# Plot IoU
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, val_iou, marker='o', linewidth=2, label='Validation IoU')
plt.plot(learning_rates, test_iou, marker='s', linewidth=2, label='Test IoU')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('IoU Score')
plt.title('Effect of Learning Rate on IoU')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Plot Dice
# -------------------------
plt.figure(figsize=(8, 5))
plt.plot(learning_rates, val_dice, marker='o', linewidth=2, label='Validation Dice')
plt.plot(learning_rates, test_dice, marker='s', linewidth=2, label='Test Dice')
plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Dice Score')
plt.title('Effect of Learning Rate on Dice Score')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------
# Combined Plot
# -------------------------
plt.figure(figsize=(10, 6))
plt.plot(learning_rates, val_iou, marker='o', linewidth=2, label='Val IoU')
plt.plot(learning_rates, test_iou, marker='s', linewidth=2, label='Test IoU')
plt.plot(learning_rates, val_dice, marker='^', linewidth=2, label='Val Dice')
plt.plot(learning_rates, test_dice, marker='d', linewidth=2, label='Test Dice')

plt.xscale('log')
plt.xlabel('Learning Rate')
plt.ylabel('Metric Value')
plt.title('Learning Rate Tuning: IoU and Dice')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()