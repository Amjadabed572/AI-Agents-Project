"""
train.py - Training loop and evaluation utilities for HW1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple
import time


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch. Returns mean MSE loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        noisy  = batch["noisy_window"].to(device)
        clean  = batch["clean_window"].to(device)
        label  = batch["label"].to(device)

        optimizer.zero_grad()
        pred = model(noisy, label)
        loss = criterion(pred, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)

    return total_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on a DataLoader. Returns mean MSE loss."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            noisy  = batch["noisy_window"].to(device)
            clean  = batch["clean_window"].to(device)
            label  = batch["label"].to(device)
            pred   = model(noisy, label)
            loss   = criterion(pred, clean)
            total_loss += loss.item() * noisy.size(0)
    return total_loss / len(loader.dataset)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Full training run.

    Returns
    -------
    dict with keys 'train_loss' and 'val_loss' (lists, one value per epoch)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss  = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"[{model.__class__.__name__}] "
                f"Epoch {epoch:3d}/{epochs} | "
                f"Train MSE: {tr_loss:.6f} | "
                f"Val MSE: {va_loss:.6f} | "
                f"{time.time()-t0:.1f}s"
            )

    return history


def compare_models(
    models: Dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Dict[str, List[float]]]:
    """
    Train all models and return their histories.

    Parameters
    ----------
    models : dict  name → model instance

    Returns
    -------
    dict  name → {'train_loss': [...], 'val_loss': [...]}
    """
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"  Training {name}")
        print(f"{'='*50}")
        history = train_model(model, train_loader, val_loader,
                               epochs=epochs, lr=lr, device=device)
        results[name] = history
        final_val = history["val_loss"][-1]
        print(f"  -> Final Val MSE: {final_val:.6f}")
    return results


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)