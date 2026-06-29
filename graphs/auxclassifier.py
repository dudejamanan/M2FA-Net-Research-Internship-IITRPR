import matplotlib.pyplot as plt

aux_weight = [0, 0.1, 0.3, 0.5, 0.7]

test_iou = [
    0.7983188096,
    0.8021578889,
    0.8057443885,
    0.8070886689,
    0.8089919339
]

test_dice = [
    0.8860231178,
    0.8885638873,
    0.8908316155,
    0.8917902150,
    0.8929356409
]

plt.figure(figsize=(8, 5))

plt.plot(
    aux_weight,
    test_iou,
    marker='o',
    linewidth=2.5,
    markersize=8,
    label='Test IoU'
)

plt.plot(
    aux_weight,
    test_dice,
    marker='s',
    linewidth=2.5,
    markersize=8,
    label='Test Dice'
)

# Highlight best point
best_idx = test_iou.index(max(test_iou))

plt.scatter(
    aux_weight[best_idx],
    test_iou[best_idx],
    s=120,
    zorder=5
)

plt.scatter(
    aux_weight[best_idx],
    test_dice[best_idx],
    s=120,
    zorder=5
)

plt.xlabel('Auxiliary Loss Weight')
plt.ylabel('Score')
plt.title('Effect of Auxiliary Loss Weight on Test Performance')

plt.xticks(aux_weight)
plt.ylim(0.79, 0.90)

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    'auxiliary_loss_weight_ablation.png',
    dpi=300,
    bbox_inches='tight'
)

plt.show()