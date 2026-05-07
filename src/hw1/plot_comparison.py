"""
plot_comparison.py - 6-panel signal comparison visualization for hw1.
Shows: noisy input, ground truth, MLP/RNN/LSTM predictions, overlay.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from hw1.constants import (
    FREQUENCIES,
    NOISE_SIGMA,
    NUM_CLASSES,
    SAMPLE_RATE,
    WINDOW_LEN,
)
from hw1.signals import extract_windows, generate_combined


def _get_predictions(
    models: dict[str, nn.Module],
    mixed: np.ndarray,
    freq_idx: int,
    device: torch.device,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Run all models on one window and return (predictions dict, mixed window)."""
    mixed_tensor = (
        torch.tensor(extract_windows(mixed, WINDOW_LEN)[0]).unsqueeze(0).to(device)
    )
    label = torch.zeros(1, NUM_CLASSES, device=device)
    label[0, freq_idx] = 1.0

    preds: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            preds[name] = model(mixed_tensor, label).squeeze().cpu().numpy()

    return preds, mixed_tensor.squeeze().cpu().numpy()


def plot_comparison(
    models: dict[str, nn.Module],
    device: torch.device,
    freq_idx: int = 2,
    sample_idx: int = 0,
    save_path: str = "assets/comparison.png",
) -> None:
    """
    Create a 6-panel comparison figure for one frequency and sample.

    Panels: noisy input, ground truth, MLP, RNN, LSTM, overlay.
    """
    np.random.seed(sample_idx)
    mixed, components = generate_combined(
        list(FREQUENCIES),
        duration=10.0,
        sample_rate=SAMPLE_RATE,
        noise_std=NOISE_SIGMA,
    )
    clean_win = extract_windows(components[freq_idx], WINDOW_LEN)[0]
    preds, mixed_win = _get_predictions(models, mixed, freq_idx, device)

    mlp_pred = preds["MLP"]
    rnn_pred = preds["RNN"]
    lstm_pred = preds["LSTM"]

    freq_label = f"{int(FREQUENCIES[freq_idx])} Hz"
    fig, axes = plt.subplots(6, 1, figsize=(10, 14))

    panels = [
        (axes[0], mixed_win, "Noisy Mixed Input"),
        (axes[1], clean_win, f"Ground Truth Pure Sine ({freq_label})"),
        (axes[2], mlp_pred, "MLP Prediction"),
        (axes[3], rnn_pred, "RNN Prediction"),
        (axes[4], lstm_pred, "LSTM Prediction"),
    ]
    for ax, signal, title in panels:
        ax.plot(signal)
        ax.set_title(title)
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    axes[5].plot(clean_win, label="Ground Truth", linewidth=2, color="black")
    axes[5].plot(mlp_pred, label="MLP", linestyle="--", alpha=0.8)
    axes[5].plot(rnn_pred, label="RNN", linestyle="-.", alpha=0.8)
    axes[5].plot(lstm_pred, label="LSTM", linestyle=":", alpha=0.8)
    axes[5].set_title("Overlay: All Predictions vs Ground Truth")
    axes[5].set_ylabel("Amplitude")
    axes[5].legend(loc="upper right")
    axes[5].grid(True, alpha=0.3)
    axes[-1].set_xlabel("Sample Index")

    fig.suptitle(f"Signal Comparison — Sample {sample_idx} ({freq_label})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}", flush=True)
    plt.show()
