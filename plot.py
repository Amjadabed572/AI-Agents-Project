"""
plot.py - Visualization for HW1
Generates:
1. Training vs validation loss curves for all 3 models
2. Signal visualization: mixed input vs extracted output vs ground truth
3. Bar chart comparing final MSE across models
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
from dataset import (
    get_dataloaders,
    generate_combined,
    extract_windows,
    FREQUENCIES,
    SAMPLE_RATE,
    WINDOW_LEN,
    NUM_CLASSES,
    NOISE_SIGMA,
)
from models import MLP, RNNModel, LSTMModel
from train import train_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
LR = 1e-3
FREQ_LIST: List[float] = list(FREQUENCIES)


def train_all() -> tuple:  # type: ignore[type-arg]
    """Train all models and return (models dict, histories dict)."""
    train_loader, val_loader = get_dataloaders(batch_size=64, samples_per_freq=500)
    models: Dict[str, torch.nn.Module] = {
        "MLP": MLP(),
        "RNN": RNNModel(),
        "LSTM": LSTMModel(),
    }
    histories: Dict[str, Dict[str, List[float]]] = {}
    for name, model in models.items():
        print(f"Training {name}...", flush=True)
        hist = train_model(
            model,
            train_loader,
            val_loader,
            epochs=EPOCHS,
            lr=LR,
            device=DEVICE,
            verbose=False,
        )
        histories[name] = hist
        print(f"  Done. Final Val MSE: {hist['val_loss'][-1]:.6f}", flush=True)
    return models, histories


def plot_loss_curves(histories: Dict[str, Dict[str, List[float]]]) -> None:
    """Plot train/val loss curves for all models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {"MLP": "#2196F3", "RNN": "#FF5722", "LSTM": "#4CAF50"}

    for ax, (name, hist) in zip(axes, histories.items()):
        epochs = range(1, len(hist["train_loss"]) + 1)
        ax.plot(
            epochs,
            hist["train_loss"],
            label="Train MSE",
            color=colors[name],
            linewidth=2,
        )
        ax.plot(
            epochs,
            hist["val_loss"],
            label="Val MSE",
            color=colors[name],
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
            color=colors[name],
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


def plot_signal_extraction(models: Dict[str, torch.nn.Module]) -> None:
    """3x4 grid: each model x each frequency showing extraction quality."""
    np.random.seed(99)
    freq_names = [f"{int(f)} Hz" for f in FREQ_LIST]
    model_names = list(models.keys())

    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.3)

    for row, name in enumerate(model_names):
        model = models[name].to(DEVICE)
        model.eval()

        for col, freq_idx in enumerate(range(NUM_CLASSES)):
            ax = fig.add_subplot(gs[row, col])

            mixed, components = generate_combined(
                FREQ_LIST,
                duration=10.0,
                sample_rate=SAMPLE_RATE,
                noise_std=NOISE_SIGMA,
            )
            clean = components[freq_idx]

            mixed_win = (
                torch.tensor(extract_windows(mixed, WINDOW_LEN)[0])
                .unsqueeze(0)
                .to(DEVICE)
            )
            clean_win = extract_windows(clean, WINDOW_LEN)[0]

            label = torch.zeros(1, NUM_CLASSES, device=DEVICE)
            label[0, freq_idx] = 1.0

            with torch.no_grad():
                pred = model(mixed_win, label).squeeze().cpu().numpy()

            x = np.arange(WINDOW_LEN)
            ax.plot(
                x,
                mixed_win.squeeze().cpu().numpy(),
                color="gray",
                alpha=0.5,
                linewidth=1.5,
                label="Mixed",
            )
            ax.plot(
                x, clean_win, color="green", linewidth=2, linestyle="--", label="Truth"
            )
            ax.plot(x, pred, color="red", linewidth=2, label="Predicted")

            mse = float(np.mean((pred - clean_win) ** 2))
            if row == 0:
                ax.set_title(freq_names[col], fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(name, fontsize=12, fontweight="bold")
            ax.set_xlabel("Sample")
            ax.grid(True, alpha=0.3)
            ax.annotate(
                f"MSE={mse:.4f}",
                xy=(0.05, 0.05),
                xycoords="axes fraction",
                fontsize=8,
                color="red",
            )
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="upper right")

    fig.suptitle(
        "Frequency Extraction: Mixed Input → Predicted vs Ground Truth\n"
        "(Gray=Mixed, Green=Truth, Red=Predicted)",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig("signal_extraction.png", dpi=150, bbox_inches="tight")
    print("Saved: signal_extraction.png", flush=True)
    plt.show()


def plot_final_comparison(histories: Dict[str, Dict[str, List[float]]]) -> None:
    """Bar chart comparing final val MSE across all models."""
    names = list(histories.keys())
    values = [h["val_loss"][-1] for h in histories.values()]
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

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


if __name__ == "__main__":
    print("Training all models for plotting...", flush=True)
    models, histories = train_all()
    print("\nGenerating plots...", flush=True)
    plot_loss_curves(histories)
    plot_signal_extraction(models)
    plot_final_comparison(histories)
    print("\nAll plots saved!", flush=True)
