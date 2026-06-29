import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("FULL MODEL (Proposed)_Run_1_history.csv")

epochs = df["Epoch"]

iou = df["Val_IoU"]
dice = df["Val_Dice"]

best_idx = iou.idxmax()

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    iou,
    marker='o',
    linewidth=2,
    label="IoU"
)

plt.plot(
    epochs,
    dice,
    marker='s',
    linewidth=2,
    label="Dice"
)



plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation IoU and Dice Across Training")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(
    "iou_dice_curves.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()