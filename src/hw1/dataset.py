"""
dataset.py - SineDataset and DataLoader factory for hw1.
Task: combined noisy signal window + 1-hot label -> clean target frequency window.
"""


import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from hw1.constants import (
    AMPLITUDE,
    FREQUENCIES,
    NOISE_SIGMA,
    SAMPLE_RATE,
    SIGNAL_DURATION,
    WINDOW_LEN,
)
from hw1.signals import extract_windows, generate_combined, one_hot


class SineDataset(Dataset):
    """
    Dataset for frequency extraction from combined signals.

    Each item:
        mixed_window  : Tensor [WINDOW_LEN]  - combined noisy signal window
        clean_window  : Tensor [WINDOW_LEN]  - target frequency only (no noise)
        label         : Tensor [NUM_CLASSES] - 1-hot frequency label
        freq_idx      : int                  - class index (0-3)
    """

    def __init__(
        self,
        frequencies: list[float] = FREQUENCIES,
        duration: float = SIGNAL_DURATION,
        sample_rate: int = SAMPLE_RATE,
        window_len: int = WINDOW_LEN,
        noise_std: float = NOISE_SIGMA,
        samples_per_freq: int = 500,
        seed: int = 42,
    ):
        """Initialise dataset and generate all windows."""
        super().__init__()
        np.random.seed(seed)
        self.window_len = window_len
        self.frequencies = frequencies
        self.mixed_windows: list[np.ndarray] = []
        self.clean_windows: list[np.ndarray] = []
        self.labels: list[np.ndarray] = []
        self.freq_indices: list[int] = []
        self._generate(frequencies, duration, sample_rate, noise_std, samples_per_freq)

    def _generate(
        self,
        frequencies: list[float],
        duration: float,
        sample_rate: int,
        noise_std: float,
        samples_per_freq: int,
    ) -> None:
        """Generate windows for all frequency classes."""
        for freq_idx in range(len(frequencies)):
            label = one_hot(freq_idx)
            collected = 0
            while collected < samples_per_freq:
                mixed, components = generate_combined(
                    frequencies,
                    duration,
                    sample_rate,
                    amplitude=AMPLITUDE,
                    noise_std=noise_std,
                )
                for mw, cw in zip(
                    extract_windows(mixed, self.window_len),
                    extract_windows(components[freq_idx], self.window_len),
                ):
                    if collected >= samples_per_freq:
                        break
                    self.mixed_windows.append(mw)
                    self.clean_windows.append(cw)
                    self.labels.append(label)
                    self.freq_indices.append(freq_idx)
                    collected += 1

    def __len__(self) -> int:
        """Return total number of windows."""
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Return one dataset item by index."""
        return {
            "mixed_window": torch.tensor(self.mixed_windows[idx]),
            "clean_window": torch.tensor(self.clean_windows[idx]),
            "label": torch.tensor(self.labels[idx]),
            "freq_idx": self.freq_indices[idx],
        }


def get_dataloaders(
    batch_size: int = 64,
    train_split: float = 0.8,
    samples_per_freq: int = 500,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader]:
    """Build train/val DataLoaders with reproducible split."""
    dataset = SineDataset(samples_per_freq=samples_per_freq, seed=seed)
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )
