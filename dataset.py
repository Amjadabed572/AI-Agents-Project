"""
dataset.py - Signal Dataset Generator for HW1
Generates COMBINED sine wave signals (all 4 frequencies mixed together).
The network must extract one target frequency from the combined noisy signal.
Frequencies chosen: 1 Hz, 5 Hz, 10 Hz, 20 Hz
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List

# ── Constants ──────────────────────────────────────────────────────────────────
FREQUENCIES: List[float] = [1.0, 5.0, 10.0, 20.0]  # Hz (chosen: well-separated)
SAMPLE_RATE = 200  # samples/sec (≥ 2 × 20 Hz = 40 Hz, Nyquist)
WINDOW_LEN = 10  # context window in samples
SIGNAL_DURATION = 10.0  # seconds
AMPLITUDE = 1.0  # base amplitude A per frequency component
NOISE_SIGMA = 0.10  # noise as fraction of A (10%)
NUM_CLASSES = len(FREQUENCIES)  # 4


def generate_sine(
    freq: float,
    duration: float = SIGNAL_DURATION,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    phase: float = 0.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Generate one sine wave component: y(t) = (A +- sigma_A) * sin(2pi*f*t + phi + sigma_2)

    Parameters
    ----------
    freq      : frequency in Hz
    noise_std : noise level as fraction of A (0 = pure)

    Returns np.ndarray of shape (num_samples,)
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    A_actual = amplitude * (1.0 + np.random.randn() * noise_std)
    phase_noise = np.random.randn() * noise_std if noise_std > 0 else 0.0
    signal = A_actual * np.sin(2 * np.pi * freq * t + phase + phase_noise)
    if noise_std > 0:
        signal += np.random.randn(len(t)) * amplitude * noise_std
    return signal.astype(np.float32)


def generate_combined(
    frequencies: List[float] = FREQUENCIES,
    duration: float = SIGNAL_DURATION,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    noise_std: float = 0.0,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Generate a combined signal = sum of all frequency components.

    Returns
    -------
    combined   : np.ndarray (num_samples,) - mixed signal (model input)
    components : list of np.ndarray        - each individual clean sine wave
    """
    components = []
    for freq in frequencies:
        phase = np.random.uniform(0, 2 * np.pi)
        component = generate_sine(
            freq, duration, sample_rate, amplitude, phase, noise_std=0.0
        )
        components.append(component)

    combined = np.sum(components, axis=0).astype(np.float32)
    if noise_std > 0:
        n_samples = int(duration * sample_rate)
        combined += (np.random.randn(n_samples) * amplitude * noise_std).astype(
            np.float32
        )
    return combined, components


def one_hot(class_idx: int, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Return a 1-hot encoded vector for class_idx."""
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[class_idx] = 1.0
    return vec


def extract_windows(signal: np.ndarray, window_len: int = WINDOW_LEN) -> np.ndarray:
    """Slice a 1-D signal into non-overlapping windows -> (n_windows, window_len)."""
    n_windows = len(signal) // window_len
    return signal[: n_windows * window_len].reshape(n_windows, window_len)


class SineDataset(Dataset):
    """
    Dataset for frequency extraction from combined signals.

    Each item:
        'mixed_window'  : Tensor [WINDOW_LEN]  - window of combined noisy signal
        'clean_window'  : Tensor [WINDOW_LEN]  - window of target frequency only
        'label'         : Tensor [NUM_CLASSES] - 1-hot of target frequency
        'freq_idx'      : int                  - class index (0-3)

    Task: given mixed_window + label -> predict clean_window of target frequency.
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

        self.mixed_windows: List[np.ndarray] = []
        self.clean_windows: List[np.ndarray] = []
        self.labels: List[np.ndarray] = []
        self.freq_indices: List[int] = []

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
                clean = components[freq_idx]

                mixed_wins = extract_windows(mixed, window_len)
                clean_wins = extract_windows(clean, window_len)

                for mw, cw in zip(mixed_wins, clean_wins):
                    if collected >= samples_per_freq:
                        break
                    self.mixed_windows.append(mw)
                    self.clean_windows.append(cw)
                    self.labels.append(label)
                    self.freq_indices.append(freq_idx)
                    collected += 1

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
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
) -> Tuple[DataLoader, DataLoader]:
    """Build train / validation DataLoaders. Returns (train_loader, val_loader)."""
    dataset = SineDataset(samples_per_freq=samples_per_freq, seed=seed)
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader
