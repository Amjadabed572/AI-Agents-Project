"""
metrics.py - Regression metrics for hw1 evaluation.
Provides MSE, MAE, and R² computation over model predictions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def compute_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean squared error between pred and target arrays."""
    return float(np.mean((pred - target) ** 2))


def compute_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute mean absolute error between pred and target arrays."""
    return float(np.mean(np.abs(pred - target)))


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute coefficient of determination R².

    R² = 1 - SS_res / SS_tot.
    Returns 1.0 for perfect prediction, can be negative for bad models.
    """
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def evaluate_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """
    Run inference on loader and compute MSE, MAE, and R² metrics.

    Parameters
    ----------
    model  : trained model with forward(mixed_window, label) interface
    loader : DataLoader over the evaluation split
    device : torch device for inference

    Returns
    -------
    dict with keys 'mse', 'mae', 'r2', each a Python float
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            mixed = batch["mixed_window"].to(device)
            label = batch["label"].to(device)
            clean = batch["clean_window"].cpu().numpy()
            pred = model(mixed, label).cpu().numpy()
            all_preds.append(pred)
            all_targets.append(clean)

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return {
        "mse": compute_mse(preds, targets),
        "mae": compute_mae(preds, targets),
        "r2": compute_r2(preds, targets),
    }
