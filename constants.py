"""
constants.py - Shared constants for HW1.
All project-wide constants are defined here and imported by other modules.
"""

from typing import List

# Frequencies chosen: well-separated on log scale, covers low/mid/high range
FREQUENCIES: List[float] = [1.0, 5.0, 10.0, 20.0]  # Hz

# Sampling rate: 200 Hz satisfies Nyquist for max freq 20 Hz (min 40 Hz needed)
SAMPLE_RATE: int = 200  # samples per second

WINDOW_LEN: int = 10  # context window size in samples
SIGNAL_DURATION: float = 10.0  # seconds per generated signal
AMPLITUDE: float = 1.0  # base amplitude A per frequency component
NOISE_SIGMA: float = 0.10  # noise level as fraction of A (10%)
NUM_CLASSES: int = len(FREQUENCIES)  # 4 frequency classes
