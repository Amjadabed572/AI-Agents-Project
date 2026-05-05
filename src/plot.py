"""
plot.py - Entry point for all hw1 visualizations.
Trains all models and generates loss curves, signal extraction, comparison chart.
"""

import torch
from typing import Dict, List
from hw1.dataset import get_dataloaders
from hw1.models import MLP, RNNModel, LSTMModel
from hw1.train import train_model
from hw1.plot_losses import plot_loss_curves, plot_final_comparison
from hw1.plot_signals import plot_signal_extraction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
LR = 1e-3


def train_all() -> tuple:  # type: ignore[type-arg]
    """
    Train MLP, RNN, and LSTM models.

    Returns (models dict, histories dict).
    """
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


if __name__ == "__main__":
    print("Training all models for plotting...", flush=True)
    models, histories = train_all()
    print("\nGenerating plots...", flush=True)
    plot_loss_curves(histories)
    plot_signal_extraction(models, DEVICE)
    plot_final_comparison(histories)
    print("\nAll plots saved to assets/!", flush=True)
