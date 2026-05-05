"""
dataset.py - Signal Dataset Generator for HW1
Generates sine wave signals at 4 known frequencies with/without noise.
Frequencies chosen: 1 Hz, 5 Hz, 10 Hz, 20 Hz
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# ── Constants ──────────────────────────────────────────────────────────────────
FREQUENCIES = [1, 5, 10, 20]          # Hz  (chosen to be well-separated)
SAMPLE_RATE  = 200                      # samples/sec  (≥ 2 × max_freq = 40)
WINDOW_LEN   = 10                       # context window in samples
SIGNAL_DURATION = 10.0                  # seconds
AMPLITUDE    = 1.0                      # base amplitude A
NOISE_SIGMA  = 0.10                     # noise as fraction of A  (10 %)
NUM_CLASSES  = len(FREQUENCIES)         # 4


def generate_sine(
    freq: float,
    duration: float = SIGNAL_DURATION,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    phase: float = 0.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Generate a (possibly noisy) sine wave.

    Signal model:  y(t) = (A ± σ_A) · sin(2π f t + φ) + σ_noise
    where σ_A  = amplitude * noise_std  (amplitude jitter)
          σ_noise is additive Gaussian noise with the same std.

    Parameters
    ----------
    freq        : frequency in Hz
    duration    : total length in seconds
    sample_rate : samples per second
    amplitude   : base amplitude A
    phase       : initial phase φ  (radians)
    noise_std   : noise level as a fraction of A  (0 = pure signal)

    Returns
    -------
    np.ndarray of shape (num_samples,)
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    A_actual = amplitude * (1.0 + np.random.randn() * noise_std)   # amplitude jitter
    signal = A_actual * np.sin(2 * np.pi * freq * t + phase)
    if noise_std > 0:
        signal += np.random.randn(len(t)) * amplitude * noise_std  # additive noise
    return signal.astype(np.float32)


def one_hot(class_idx: int, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Return a 1-hot encoded vector for class_idx."""
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[class_idx] = 1.0
    return vec


def extract_windows(signal: np.ndarray, window_len: int = WINDOW_LEN) -> np.ndarray:
    """
    Slice a 1-D signal into consecutive non-overlapping windows.

    Returns array of shape  (num_windows, window_len).
    """
    n_windows = len(signal) // window_len
    return signal[: n_windows * window_len].reshape(n_windows, window_len)


class SineDataset(Dataset):
    """
    PyTorch Dataset of sliding-window sine-wave samples.

    Each item is a dict:
        'noisy_window'  : Tensor [WINDOW_LEN]  – samples with noise
        'clean_window'  : Tensor [WINDOW_LEN]  – samples without noise
        'label'         : Tensor [NUM_CLASSES] – 1-hot frequency label
        'freq_idx'      : int                  – class index (0–3)
    """

    def __init__(
        self,
        frequencies: List[float] = FREQUENCIES,
        duration: float = SIGNAL_DURATION,
        sample_rate: int = SAMPLE_RATE,
        window_len: int = WINDOW_LEN,
        noise_std: float = NOISE_SIGMA,
        samples_per_freq: int = 500,
        seed: int = 42,
    ):
        super().__init__()
        np.random.seed(seed)

        self.window_len = window_len
        self.frequencies = frequencies

        self.noisy_windows: List[np.ndarray] = []
        self.clean_windows: List[np.ndarray] = []
        self.labels:        List[np.ndarray] = []
        self.freq_indices:  List[int]        = []

        for freq_idx, freq in enumerate(frequencies):
            label = one_hot(freq_idx)
            windows_collected = 0

            while windows_collected < samples_per_freq:
                # random phase each signal to increase variety
                phase = np.random.uniform(0, 2 * np.pi)

                clean  = generate_sine(freq, duration, sample_rate,
                                        noise_std=0.0, phase=phase)
                noisy  = generate_sine(freq, duration, sample_rate,
                                        noise_std=noise_std, phase=phase)

                clean_wins = extract_windows(clean, window_len)
                noisy_wins = extract_windows(noisy, window_len)

                for cw, nw in zip(clean_wins, noisy_wins):
                    if windows_collected >= samples_per_freq:
                        break
                    self.clean_windows.append(cw)
                    self.noisy_windows.append(nw)
                    self.labels.append(label)
                    self.freq_indices.append(freq_idx)
                    windows_collected += 1

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "noisy_window": torch.tensor(self.noisy_windows[idx]),
            "clean_window": torch.tensor(self.clean_windows[idx]),
            "label":        torch.tensor(self.labels[idx]),
            "freq_idx":     self.freq_indices[idx],
        }


def get_dataloaders(
    batch_size: int = 64,
    train_split: float = 0.8,
    samples_per_freq: int = 500,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train / validation DataLoaders.

    Returns (train_loader, val_loader).
    """
    dataset = SineDataset(samples_per_freq=samples_per_freq, seed=seed)
    n_train = int(len(dataset) * train_split)
    n_val   = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader