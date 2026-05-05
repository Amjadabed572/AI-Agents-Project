"""
constants.py - Shared constants for hw1.
All project-wide constants are defined here and imported by other modules.
Single source of truth — never hardcode these values elsewhere.
"""


# Frequencies: logarithmically spaced, covers low/mid/high range
# Max 20 Hz is well below Nyquist limit of 100 Hz (sample rate 200 Hz)
FREQUENCIES: list[float] = [1.0, 5.0, 10.0, 20.0]  # Hz

# 200 Hz satisfies Nyquist for max frequency 20 Hz (min needed: 40 Hz)
SAMPLE_RATE: int = 200  # samples per second

WINDOW_LEN: int = 10  # context window size in samples
SIGNAL_DURATION: float = 10.0  # seconds per generated signal
AMPLITUDE: float = 1.0  # base amplitude A per frequency component
NOISE_SIGMA: float = 0.10  # noise level as fraction of A (10%)
NUM_CLASSES: int = len(FREQUENCIES)  # 4 frequency classes
