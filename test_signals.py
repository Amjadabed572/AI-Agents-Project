"""
test_signals.py - Unit tests for signal generation, helpers, and dataset.
Run with: pytest test_signals.py -v
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from constants import (
    FREQUENCIES,
    SAMPLE_RATE,
    WINDOW_LEN,
    NUM_CLASSES,
    SIGNAL_DURATION,
    AMPLITUDE,
    NOISE_SIGMA,
)
from signals import generate_sine, generate_combined, one_hot, extract_windows
from dataset import SineDataset, get_dataloaders


class TestGenerateSine:
    """Tests for generate_sine() function."""

    def test_output_length(self):
        """Signal length equals duration x sample_rate."""
        assert len(generate_sine(freq=1, duration=10, sample_rate=200)) == 2000

    def test_output_dtype(self):
        """Signal dtype is float32 for PyTorch compatibility."""
        assert generate_sine(freq=1).dtype == np.float32

    def test_pure_signal_amplitude(self):
        """Pure sine (no noise) stays within [-A, A]."""
        sig = generate_sine(freq=5, noise_std=0.0)
        assert np.max(np.abs(sig)) <= AMPLITUDE + 1e-6

    def test_noise_increases_variance(self):
        """Noisy signal has higher variance than pure signal."""
        np.random.seed(0)
        assert np.var(generate_sine(10, noise_std=0.20)) > np.var(
            generate_sine(10, noise_std=0.0)
        )

    def test_frequency_content(self):
        """Dominant FFT frequency matches requested frequency."""
        sig = generate_sine(freq=5, duration=10, sample_rate=200, noise_std=0.0)
        fft_mag = np.abs(np.fft.rfft(sig))
        fft_freq = np.fft.rfftfreq(len(sig), d=1.0 / 200)
        assert abs(fft_freq[np.argmax(fft_mag)] - 5) < 1.0

    def test_different_frequencies_differ(self):
        """Signals at different frequencies are not identical."""
        np.random.seed(1)
        assert not np.allclose(
            generate_sine(1, noise_std=0.0, phase=0.0),
            generate_sine(20, noise_std=0.0, phase=0.0),
        )

    def test_custom_duration(self):
        """Custom duration produces correct sample count."""
        assert len(generate_sine(freq=1, duration=5, sample_rate=100)) == 500


class TestGenerateCombined:
    """Tests for generate_combined() function."""

    def test_output_length(self):
        """Combined signal length equals duration x sample_rate."""
        combined, _ = generate_combined()
        assert len(combined) == int(SIGNAL_DURATION * SAMPLE_RATE)

    def test_num_components(self):
        """Number of components equals number of frequencies."""
        _, components = generate_combined()
        assert len(components) == NUM_CLASSES

    def test_combined_is_sum_of_components(self):
        """Combined signal equals sum of components when no noise."""
        np.random.seed(5)
        combined, components = generate_combined(noise_std=0.0)
        assert np.allclose(combined, np.sum(components, axis=0), atol=1e-5)

    def test_noise_changes_combined(self):
        """Adding noise changes the combined signal."""
        np.random.seed(7)
        c1, _ = generate_combined(noise_std=0.0)
        np.random.seed(7)
        c2, _ = generate_combined(noise_std=0.1)
        assert not np.allclose(c1, c2)

    def test_components_dtype(self):
        """All components are float32."""
        _, components = generate_combined()
        for c in components:
            assert c.dtype == np.float32


class TestOneHot:
    """Tests for one_hot() helper."""

    def test_correct_length(self):
        assert len(one_hot(0)) == NUM_CLASSES

    def test_single_one(self):
        vec = one_hot(2)
        assert np.sum(vec) == 1.0 and vec[2] == 1.0

    def test_all_indices(self):
        for i in range(NUM_CLASSES):
            vec = one_hot(i)
            assert vec[i] == 1.0 and np.sum(vec == 0) == NUM_CLASSES - 1

    def test_dtype(self):
        assert one_hot(0).dtype == np.float32


class TestExtractWindows:
    """Tests for extract_windows() helper."""

    def test_window_shape(self):
        wins = extract_windows(np.ones(100, dtype=np.float32), window_len=10)
        assert wins.shape == (10, 10)

    def test_no_overlap(self):
        sig = np.arange(100, dtype=np.float32)
        wins = extract_windows(sig, window_len=10)
        assert wins[0, -1] != wins[1, 0]

    def test_partial_window_dropped(self):
        assert extract_windows(np.ones(105, dtype=np.float32), 10).shape[0] == 10


class TestSineDataset:
    """Tests for SineDataset class."""

    @pytest.fixture(scope="class")
    def dataset(self):
        """Shared small dataset fixture."""
        return SineDataset(samples_per_freq=50, seed=0)

    def test_total_length(self, dataset):
        assert len(dataset) == NUM_CLASSES * 50

    def test_item_keys(self, dataset):
        assert set(dataset[0].keys()) == {
            "mixed_window",
            "clean_window",
            "label",
            "freq_idx",
        }

    def test_mixed_window_shape(self, dataset):
        assert dataset[0]["mixed_window"].shape == (WINDOW_LEN,)

    def test_clean_window_shape(self, dataset):
        assert dataset[0]["clean_window"].shape == (WINDOW_LEN,)

    def test_label_shape(self, dataset):
        assert dataset[0]["label"].shape == (NUM_CLASSES,)

    def test_label_is_one_hot(self, dataset):
        for i in range(len(dataset)):
            assert dataset[i]["label"].sum().item() == pytest.approx(1.0)

    def test_freq_idx_range(self, dataset):
        for i in range(len(dataset)):
            assert 0 <= dataset[i]["freq_idx"] < NUM_CLASSES

    def test_mixed_differs_from_clean(self, dataset):
        """Mixed window contains all frequencies so differs from single-freq clean."""
        assert all(
            not torch.allclose(dataset[i]["mixed_window"], dataset[i]["clean_window"])
            for i in range(20)
        )


class TestGetDataloaders:
    """Tests for get_dataloaders() factory."""

    def test_returns_two_loaders(self):
        tr, va = get_dataloaders(batch_size=32, samples_per_freq=50)
        assert isinstance(tr, DataLoader) and isinstance(va, DataLoader)

    def test_sizes_sum_to_total(self):
        tr, va = get_dataloaders(batch_size=32, samples_per_freq=50)
        assert len(tr.dataset) + len(va.dataset) == NUM_CLASSES * 50  # type: ignore[arg-type]
