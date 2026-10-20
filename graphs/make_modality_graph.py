"""
Input-modality ablation bar graph for M2FA-Net.

Usage:
    python make_modality_graph.py                 # writes modality_ablation.png and .pdf
    python make_modality_graph.py --out Images/modality_ablation

Edit the numbers in DATA below if your results change.
Requires: matplotlib, numpy
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (label, mean, std) per input setting, in percent
LABELS = ["RGB only", "RGB + MS4", "RGB + MS4\n+ NDVI", "RGB + MS4\n+ NDVI + NDRE"]
DATA = {
    "IoU (%)":  {"mean": [83.11, 83.75, 83.68, 83.81], "std": [1.24, 0.10, 0.19, 0.18], "ylim": (80, 86)},
    "Dice (%)": {"mean": [90.77, 91.15, 91.12, 91.19], "std": [0.75, 0.06, 0.11, 0.11], "ylim": (88, 93)},
}
COLORS = ["#9aa5b1", "#3b7ea1", "#3b7ea1", "#1f5f85"]


def main(out_stem: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
    x = np.arange(len(LABELS))

    for ax, (metric, d) in zip(axes, DATA.items()):
        ax.bar(
            x, d["mean"], yerr=d["std"], capsize=4,
            color=COLORS, edgecolor="black", linewidth=0.6,
            error_kw=dict(lw=1),
        )
        for xi, m, s in zip(x, d["mean"], d["std"]):
            ax.text(xi, m + s + 0.08, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=8.5)
        ax.set_ylim(*d["ylim"])  # truncated axis: say so in the caption
        ax.set_ylabel(metric, fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{out_stem}.png")
    fig.savefig(f"{out_stem}.pdf")  # vector version for the journal
    print(f"Saved {out_stem}.png and {out_stem}.pdf")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="modality_ablation", help="output path without extension")
    main(parser.parse_args().out)
