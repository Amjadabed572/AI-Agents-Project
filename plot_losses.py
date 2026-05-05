"""
plot_losses.py - Loss curve and model comparison visualizations for HW1.
"""

import matplotlib.pyplot as plt
from typing import Dict, List

COLORS = {"MLP": "#2196F3", "RNN": "#FF5722", "LSTM": "#4CAF50"}
EPOCHS = 50


def plot_loss_curves(histories: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Plot train/val MSE loss curves for all models side by side.

    Parameters
    ----------
    histories : dict mapping model name to {'train_loss': [...], 'val_loss': [...]}
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (name, hist) in zip(axes, histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(
            epochs,
            hist["train_loss"],
            label="Train MSE",
            color=COLORS[name],
            linewidth=2,
        )
        ax.plot(
            epochs,
            hist["val_loss"],
            label="Val MSE",
            color=COLORS[name],
            linewidth=2,
            linestyle="--",
        )
        ax.set_title(name, fontsize=14, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        final = hist["val_loss"][-1]
        ax.annotate(
            f"Final: {final:.4f}",
            xy=(EPOCHS, final),
            xytext=(EPOCHS * 0.55, final * 1.3),
            fontsize=9,
            color=COLORS[name],
        )

    fig.suptitle(
        "Training vs Validation Loss — Frequency Extraction Task",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig("loss_curves.png", dpi=150, bbox_inches="tight")
    print("Saved: loss_curves.png", flush=True)
    plt.show()


def plot_final_comparison(histories: Dict[str, Dict[str, List[float]]]) -> None:
    """
    Bar chart comparing final validation MSE across all models.

    Parameters
    ----------
    histories : dict mapping model name to loss history
    """
    names = list(histories.keys())
    values = [h["val_loss"][-1] for h in histories.values()]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, values, color=colors, width=0.5, edgecolor="black")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    ax.set_title("Final Validation MSE Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("MSE Loss")
    ax.set_ylim(0, max(values) * 1.2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
    print("Saved: model_comparison.png", flush=True)
    plt.show()
