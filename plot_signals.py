"""
plot_signals.py - Signal extraction visualization for HW1.
Shows mixed input vs predicted output vs ground truth for each model/frequency.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List
from constants import FREQUENCIES, SAMPLE_RATE, WINDOW_LEN, NUM_CLASSES, NOISE_SIGMA
from signals import generate_combined, extract_windows

FREQ_NAMES: List[str] = [f"{int(f)} Hz" for f in FREQUENCIES]


def _predict_window(
    model: torch.nn.Module,
    mixed: np.ndarray,
    freq_idx: int,
    device: torch.device,
) -> tuple:  # type: ignore[type-arg]
    """Run one window through a model and return (mixed_win, clean_win, pred)."""
    mixed_win = (
        torch.tensor(extract_windows(mixed, WINDOW_LEN)[0]).unsqueeze(0).to(device)
    )
    label = torch.zeros(1, NUM_CLASSES, device=device)
    label[0, freq_idx] = 1.0
    with torch.no_grad():
        pred = model(mixed_win, label).squeeze().cpu().numpy()
    return mixed_win.squeeze().cpu().numpy(), pred


def plot_signal_extraction(
    models: Dict[str, torch.nn.Module],
    device: torch.device,
) -> None:
    """
    3x4 grid showing frequency extraction quality per model and frequency.

    Parameters
    ----------
    models : dict mapping model name to trained model instance
    device : torch device to run inference on
    """
    np.random.seed(99)
    fig = plt.figure(figsize=(18, 12))
    gs = gridspec.GridSpec(3, 4, hspace=0.5, wspace=0.3)

    for row, name in enumerate(models.keys()):
        model = models[name].to(device)
        model.eval()

        for col, freq_idx in enumerate(range(NUM_CLASSES)):
            ax = fig.add_subplot(gs[row, col])
            mixed, components = generate_combined(
                list(FREQUENCIES),
                duration=10.0,
                sample_rate=SAMPLE_RATE,
                noise_std=NOISE_SIGMA,
            )
            clean_win = extract_windows(components[freq_idx], WINDOW_LEN)[0]
            mixed_arr, pred = _predict_window(model, mixed, freq_idx, device)

            x = np.arange(WINDOW_LEN)
            ax.plot(x, mixed_arr, color="gray", alpha=0.5, linewidth=1.5, label="Mixed")
            ax.plot(
                x, clean_win, color="green", linewidth=2, linestyle="--", label="Truth"
            )
            ax.plot(x, pred, color="red", linewidth=2, label="Predicted")

            mse = float(np.mean((pred - clean_win) ** 2))
            if row == 0:
                ax.set_title(FREQ_NAMES[col], fontsize=12, fontweight="bold")
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
        "Frequency Extraction: Mixed Input -> Predicted vs Ground Truth\n"
        "(Gray=Mixed, Green=Truth, Red=Predicted)",
        fontsize=13,
        fontweight="bold",
    )
    plt.savefig("signal_extraction.png", dpi=150, bbox_inches="tight")
    print("Saved: signal_extraction.png", flush=True)
    plt.show()
