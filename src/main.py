"""
main.py - Entry point for HW1.
Trains MLP, RNN, and LSTM on frequency extraction and prints results.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

import torch

print("Starting...", flush=True)

from hw1.constants import FREQUENCIES
from hw1.dataset import get_dataloaders
from hw1.models import MLP, LSTMModel, RNNModel
from hw1.shared.version import get_version
from hw1.train import compare_models, count_parameters

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 50
BATCH = 64
LR = 1e-3

print(f"Version : {get_version()}", flush=True)
print(f"Device  : {DEVICE}", flush=True)
print(f"Frequencies (Hz): {FREQUENCIES}", flush=True)

train_loader, val_loader = get_dataloaders(batch_size=BATCH, samples_per_freq=500)
print(
    f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}", flush=True
)

models = {"MLP": MLP(), "RNN": RNNModel(), "LSTM": LSTMModel()}
for name, model in models.items():
    print(f"{name}: {count_parameters(model):,} trainable parameters", flush=True)

results = compare_models(
    models, train_loader, val_loader, epochs=EPOCHS, lr=LR, device=DEVICE
)

print("\n\n=== Final Results ===", flush=True)
print(f"{'Model':<8} {'Final Val MSE':>15}", flush=True)
print("-" * 25, flush=True)
for name, hist in results.items():
    print(f"{name:<8} {hist['val_loss'][-1]:>15.6f}", flush=True)
