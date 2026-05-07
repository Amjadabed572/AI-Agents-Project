"""
train.py - Training loop and evaluation utilities for hw1.
Provides train_epoch, evaluate, train_model, compare_models, count_parameters.
"""

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

GRAD_CLIP_MAX_NORM = 1.0  # max gradient norm for clipping
EARLY_STOPPING_PATIENCE = 15  # epochs without improvement before stopping


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Run one training epoch with gradient clipping. Returns mean MSE loss."""
    model.train()
    total_loss = 0.0
    for batch in loader:
        mixed = batch["mixed_window"].to(device)
        clean = batch["clean_window"].to(device)
        label = batch["label"].to(device)
        optimizer.zero_grad()
        loss = criterion(model(mixed, label), clean)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_MAX_NORM)
        optimizer.step()
        total_loss += loss.item() * mixed.size(0)
    return total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model on DataLoader. Returns mean MSE loss, no gradients."""
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            mixed = batch["mixed_window"].to(device)
            clean = batch["clean_window"].to(device)
            label = batch["label"].to(device)
            total_loss += criterion(model(mixed, label), clean).item() * mixed.size(0)
    return total_loss / max(len(loader.dataset), 1)  # type: ignore[arg-type]


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Full training run with early stopping and gradient clipping.

    Stops early if val loss does not improve for EARLY_STOPPING_PATIENCE epochs.
    Returns dict with 'train_loss' and 'val_loss' lists (one value per epoch).
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        va = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tr)
        history["val_loss"].append(va)

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"[{model.__class__.__name__}] Epoch {epoch:3d}/{epochs} | "
                f"Train MSE: {tr:.6f} | Val MSE: {va:.6f} | "
                f"{time.time()-t0:.1f}s",
                flush=True,
            )

        if va < best_val:
            best_val = va
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(
                f"  Early stopping at epoch {epoch} "
                f"(no improvement for {EARLY_STOPPING_PATIENCE} epochs)",
                flush=True,
            )
            break

    return history


def compare_models(
    models: dict[str, nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> dict[str, dict[str, list[float]]]:
    """Train all models and return their loss histories."""
    results = {}
    for name, model in models.items():
        print(f"\n{'='*50}\n  Training {name}\n{'='*50}", flush=True)
        results[name] = train_model(
            model, train_loader, val_loader, epochs=epochs, lr=lr, device=device
        )
        print(f"  -> Final Val MSE: {results[name]['val_loss'][-1]:.6f}", flush=True)
    return results


def count_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
