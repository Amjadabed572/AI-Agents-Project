"""
test_hw1.py - Unit tests for HW1
Covers: dataset generation, model shapes, training sanity checks.
Run with:  python -m pytest test_hw1.py -v
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    generate_sine,
    one_hot,
    extract_windows,
    SineDataset,
    get_dataloaders,
    FREQUENCIES,
    SAMPLE_RATE,
    WINDOW_LEN,
    NUM_CLASSES,
    SIGNAL_DURATION,
    AMPLITUDE,
    NOISE_SIGMA,
)
from models import MLP, RNNModel, LSTMModel, INPUT_SIZE, OUTPUT_SIZE
from train import train_epoch, evaluate, count_parameters


# ══════════════════════════════════════════════════════════════════════════════
# Dataset / Signal Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerateSine:
    """Tests for the generate_sine() function."""

    def test_output_length(self):
        """Signal length equals duration × sample_rate."""
        sig = generate_sine(freq=1, duration=10, sample_rate=200)
        assert len(sig) == 2000

    def test_output_dtype(self):
        """Signal is float32 (compatible with PyTorch)."""
        sig = generate_sine(freq=1)
        assert sig.dtype == np.float32

    def test_pure_signal_amplitude(self):
        """Pure sine (no noise) stays within [-A, A]."""
        sig = generate_sine(freq=5, noise_std=0.0)
        assert np.max(np.abs(sig)) <= AMPLITUDE + 1e-6

    def test_noise_increases_variance(self):
        """Noisy signal has higher variance than the pure signal."""
        np.random.seed(0)
        pure  = generate_sine(freq=10, noise_std=0.0)
        noisy = generate_sine(freq=10, noise_std=0.20)
        assert np.var(noisy) > np.var(pure)

    def test_frequency_content(self):
        """Dominant FFT frequency matches the requested frequency."""
        freq = 5
        sig = generate_sine(freq=freq, duration=10, sample_rate=200, noise_std=0.0)
        fft_mag  = np.abs(np.fft.rfft(sig))
        fft_freq = np.fft.rfftfreq(len(sig), d=1.0 / 200)
        dominant = fft_freq[np.argmax(fft_mag)]
        assert abs(dominant - freq) < 1.0   # within 1 Hz

    def test_different_frequencies_differ(self):
        """Signals at different frequencies are not identical."""
        np.random.seed(1)
        s1 = generate_sine(freq=1,  noise_std=0.0, phase=0.0)
        s2 = generate_sine(freq=20, noise_std=0.0, phase=0.0)
        assert not np.allclose(s1, s2)

    def test_custom_duration(self):
        """Custom duration produces correct sample count."""
        sig = generate_sine(freq=1, duration=5, sample_rate=100)
        assert len(sig) == 500


class TestOneHot:
    """Tests for the one_hot() helper."""

    def test_correct_length(self):
        assert len(one_hot(0)) == NUM_CLASSES

    def test_single_one(self):
        vec = one_hot(2)
        assert np.sum(vec) == 1.0
        assert vec[2] == 1.0

    def test_all_indices(self):
        for i in range(NUM_CLASSES):
            vec = one_hot(i)
            assert vec[i] == 1.0
            assert np.sum(vec == 0) == NUM_CLASSES - 1

    def test_dtype(self):
        assert one_hot(0).dtype == np.float32


class TestExtractWindows:
    """Tests for the extract_windows() helper."""

    def test_window_shape(self):
        sig = np.ones(100, dtype=np.float32)
        wins = extract_windows(sig, window_len=10)
        assert wins.shape == (10, 10)

    def test_no_overlap(self):
        """Consecutive windows contain different indices."""
        sig  = np.arange(100, dtype=np.float32)
        wins = extract_windows(sig, window_len=10)
        assert wins[0, -1] != wins[1, 0]   # last of w0 ≠ first of w1

    def test_partial_window_dropped(self):
        """Remainder samples that don't fill a window are discarded."""
        sig  = np.ones(105, dtype=np.float32)
        wins = extract_windows(sig, window_len=10)
        assert wins.shape[0] == 10   # 105 // 10 = 10


class TestSineDataset:
    """Tests for SineDataset."""

    @pytest.fixture(scope="class")
    def dataset(self):
        return SineDataset(samples_per_freq=50, seed=0)

    def test_total_length(self, dataset):
        assert len(dataset) == NUM_CLASSES * 50

    def test_item_keys(self, dataset):
        item = dataset[0]
        assert set(item.keys()) == {"noisy_window", "clean_window", "label", "freq_idx"}

    def test_window_shapes(self, dataset):
        item = dataset[0]
        assert item["noisy_window"].shape == (WINDOW_LEN,)
        assert item["clean_window"].shape == (WINDOW_LEN,)

    def test_label_shape(self, dataset):
        item = dataset[0]
        assert item["label"].shape == (NUM_CLASSES,)

    def test_label_is_one_hot(self, dataset):
        for i in range(len(dataset)):
            label = dataset[i]["label"]
            assert label.sum().item() == pytest.approx(1.0)

    def test_freq_idx_range(self, dataset):
        for i in range(len(dataset)):
            assert 0 <= dataset[i]["freq_idx"] < NUM_CLASSES

    def test_noisy_differs_from_clean(self, dataset):
        """At least some noisy windows differ from their clean counterparts."""
        diffs = [not torch.allclose(dataset[i]["noisy_window"],
                                     dataset[i]["clean_window"])
                 for i in range(20)]
        assert any(diffs)


class TestGetDataloaders:
    """Tests for the get_dataloaders() factory."""

    def test_returns_two_loaders(self):
        tr, va = get_dataloaders(batch_size=32, samples_per_freq=50)
        assert isinstance(tr, DataLoader) and isinstance(va, DataLoader)

    def test_sizes_sum_to_total(self):
        tr, va = get_dataloaders(batch_size=32, samples_per_freq=50)
        total = len(tr.dataset) + len(va.dataset)
        assert total == NUM_CLASSES * 50


# ══════════════════════════════════════════════════════════════════════════════
# Model Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestMLP:
    def _batch(self, B=4):
        noisy = torch.randn(B, WINDOW_LEN)
        label = torch.zeros(B, NUM_CLASSES)
        label[:, 0] = 1.0
        return noisy, label

    def test_output_shape(self):
        model = MLP()
        noisy, label = self._batch()
        out = model(noisy, label)
        assert out.shape == (4, OUTPUT_SIZE)

    def test_parameter_count_positive(self):
        assert count_parameters(MLP()) > 0

    def test_grad_flows(self):
        model = MLP()
        noisy, label = self._batch()
        loss = model(noisy, label).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None


class TestRNNModel:
    def _batch(self, B=4):
        noisy = torch.randn(B, WINDOW_LEN)
        label = torch.zeros(B, NUM_CLASSES); label[:, 1] = 1.0
        return noisy, label

    def test_output_shape(self):
        model = RNNModel()
        noisy, label = self._batch()
        assert model(noisy, label).shape == (4, OUTPUT_SIZE)

    def test_grad_flows(self):
        model = RNNModel()
        noisy, label = self._batch()
        model(noisy, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None


class TestLSTMModel:
    def _batch(self, B=4):
        noisy = torch.randn(B, WINDOW_LEN)
        label = torch.zeros(B, NUM_CLASSES); label[:, 2] = 1.0
        return noisy, label

    def test_output_shape(self):
        model = LSTMModel()
        noisy, label = self._batch()
        assert model(noisy, label).shape == (4, OUTPUT_SIZE)

    def test_grad_flows(self):
        model = LSTMModel()
        noisy, label = self._batch()
        model(noisy, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_lstm_more_params_than_rnn(self):
        """LSTM has more parameters than a comparable RNN (due to 4 gates)."""
        assert count_parameters(LSTMModel()) > count_parameters(RNNModel())


# ══════════════════════════════════════════════════════════════════════════════
# Training Tests
# ══════════════════════════════════════════════════════════════════════════════

class TestTraining:
    @pytest.fixture(scope="class")
    def loaders(self):
        return get_dataloaders(batch_size=32, samples_per_freq=50, seed=7)

    @pytest.mark.parametrize("ModelClass", [MLP, RNNModel, LSTMModel])
    def test_loss_decreases(self, loaders, ModelClass):
        """Train loss after 5 epochs should be lower than after epoch 1."""
        tr, va = loaders
        model = ModelClass()
        device = torch.device("cpu")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        losses = []
        for _ in range(5):
            l = train_epoch(model, tr, optimizer, criterion, device)
            losses.append(l)

        assert losses[-1] < losses[0], (
            f"{ModelClass.__name__}: loss did not decrease "
            f"({losses[0]:.4f} → {losses[-1]:.4f})"
        )

    @pytest.mark.parametrize("ModelClass", [MLP, RNNModel, LSTMModel])
    def test_evaluate_returns_float(self, loaders, ModelClass):
        _, va = loaders
        model = ModelClass()
        device = torch.device("cpu")
        criterion = torch.nn.MSELoss()
        val_loss = evaluate(model, va, criterion, device)
        assert isinstance(val_loss, float)
        assert val_loss >= 0.0