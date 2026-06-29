import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("FULL MODEL (Proposed)_Run_1_history.csv")

epochs = df["Epoch"]

plt.figure(figsize=(8,5))

plt.plot(
    epochs,
    df["Train_Loss"],
    linewidth=2,
    label="Training Loss"
)

plt.plot(
    epochs,
    df["Val_Loss"],
    linewidth=2,
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()

plt.savefig(
    "loss_curves.png",
    dpi=600,
    bbox_inches="tight"
)

plt.show()