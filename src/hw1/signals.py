"""
signals.py - Signal generation utilities for hw1.
Generates individual sine waves and combined multi-frequency signals.
"""

from __future__ import annotations

import numpy as np

from hw1.constants import (
    AMPLITUDE,
    FREQUENCIES,
    NUM_CLASSES,
    SAMPLE_RATE,
    SIGNAL_DURATION,
    WINDOW_LEN,
)


def generate_sine(
    freq: float,
    duration: float = SIGNAL_DURATION,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    phase: float = 0.0,
    noise_std: float = 0.0,
) -> np.ndarray:
    """
    Generate one sine wave: y(t) = (A +- sigma_A) * sin(2pi*f*t + phi + sigma_2).

    Parameters
    ----------
    freq      : frequency in Hz
    noise_std : noise as fraction of A (0 = pure signal)

    Returns np.ndarray of shape (num_samples,) dtype float32.
    """
    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)
    a_jitter = amplitude * (1.0 + np.random.randn() * noise_std)
    phase_noise = np.random.randn() * noise_std if noise_std > 0 else 0.0
    signal = a_jitter * np.sin(2 * np.pi * freq * t + phase + phase_noise)
    if noise_std > 0:
        signal += np.random.randn(len(t)) * amplitude * noise_std
    return signal.astype(np.float32)


def generate_combined(
    frequencies: list[float] = FREQUENCIES,
    duration: float = SIGNAL_DURATION,
    sample_rate: int = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    noise_std: float = 0.0,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Generate combined signal = sum of all frequency components + noise.

    Returns
    -------
    combined   : np.ndarray (num_samples,) - mixed signal (model input)
    components : list of np.ndarray        - individual clean sine waves
    """
    components = []
    for freq in frequencies:
        phase = np.random.uniform(0, 2 * np.pi)
        components.append(
            generate_sine(freq, duration, sample_rate, amplitude, phase, 0.0)
        )
    combined = np.sum(components, axis=0).astype(np.float32)
    if noise_std > 0:
        n = int(duration * sample_rate)
        combined += (np.random.randn(n) * amplitude * noise_std).astype(np.float32)
    return combined, components


def one_hot(class_idx: int, num_classes: int = NUM_CLASSES) -> np.ndarray:
    """Return a 1-hot encoded vector of length num_classes for class_idx."""
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[class_idx] = 1.0
    return vec


def extract_windows(signal: np.ndarray, window_len: int = WINDOW_LEN) -> np.ndarray:
    """Slice 1-D signal into non-overlapping windows -> (n_windows, window_len)."""
    n = len(signal) // window_len
    return signal[: n * window_len].reshape(n, window_len)
