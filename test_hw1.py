"""
test_hw1.py - Unit tests for HW1
Covers: dataset generation, model shapes, training sanity checks.
Run with: python -m pytest test_hw1.py -v
"""

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import (
    generate_sine,
    generate_combined,
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
from models import MLP, RNNModel, LSTMModel, OUTPUT_SIZE
from train import train_epoch, evaluate, count_parameters

# ══════════════════════════════════════════════════════════════════════════════
# Signal Generation Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestGenerateSine:
    def test_output_length(self):
        sig = generate_sine(freq=1, duration=10, sample_rate=200)
        assert len(sig) == 2000

    def test_output_dtype(self):
        assert generate_sine(freq=1).dtype == np.float32

    def test_pure_signal_amplitude(self):
        sig = generate_sine(freq=5, noise_std=0.0)
        assert np.max(np.abs(sig)) <= AMPLITUDE + 1e-6

    def test_noise_increases_variance(self):
        np.random.seed(0)
        pure = generate_sine(freq=10, noise_std=0.0)
        noisy = generate_sine(freq=10, noise_std=0.20)
        assert np.var(noisy) > np.var(pure)

    def test_frequency_content(self):
        sig = generate_sine(freq=5, duration=10, sample_rate=200, noise_std=0.0)
        fft_mag = np.abs(np.fft.rfft(sig))
        fft_freq = np.fft.rfftfreq(len(sig), d=1.0 / 200)
        dominant = fft_freq[np.argmax(fft_mag)]
        assert abs(dominant - 5) < 1.0

    def test_different_frequencies_differ(self):
        np.random.seed(1)
        s1 = generate_sine(freq=1, noise_std=0.0, phase=0.0)
        s2 = generate_sine(freq=20, noise_std=0.0, phase=0.0)
        assert not np.allclose(s1, s2)

    def test_custom_duration(self):
        sig = generate_sine(freq=1, duration=5, sample_rate=100)
        assert len(sig) == 500


class TestGenerateCombined:
    def test_output_length(self):
        combined, components = generate_combined()
        assert len(combined) == int(SIGNAL_DURATION * SAMPLE_RATE)

    def test_num_components(self):
        _, components = generate_combined()
        assert len(components) == NUM_CLASSES

    def test_combined_is_sum_of_components(self):
        np.random.seed(5)
        combined, components = generate_combined(noise_std=0.0)
        expected = np.sum(components, axis=0)
        assert np.allclose(combined, expected, atol=1e-5)

    def test_noise_changes_combined(self):
        np.random.seed(7)
        c1, _ = generate_combined(noise_std=0.0)
        np.random.seed(7)
        c2, _ = generate_combined(noise_std=0.1)
        assert not np.allclose(c1, c2)

    def test_components_dtype(self):
        _, components = generate_combined()
        for c in components:
            assert c.dtype == np.float32


# ══════════════════════════════════════════════════════════════════════════════
# Helper Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestOneHot:
    def test_correct_length(self):
        assert len(one_hot(0)) == NUM_CLASSES

    def test_single_one(self):
        vec = one_hot(2)
        assert np.sum(vec) == 1.0 and vec[2] == 1.0

    def test_all_indices(self):
        for i in range(NUM_CLASSES):
            vec = one_hot(i)
            assert vec[i] == 1.0
            assert np.sum(vec == 0) == NUM_CLASSES - 1

    def test_dtype(self):
        assert one_hot(0).dtype == np.float32


class TestExtractWindows:
    def test_window_shape(self):
        wins = extract_windows(np.ones(100, dtype=np.float32), window_len=10)
        assert wins.shape == (10, 10)

    def test_no_overlap(self):
        sig = np.arange(100, dtype=np.float32)
        wins = extract_windows(sig, window_len=10)
        assert wins[0, -1] != wins[1, 0]

    def test_partial_window_dropped(self):
        wins = extract_windows(np.ones(105, dtype=np.float32), window_len=10)
        assert wins.shape[0] == 10


# ══════════════════════════════════════════════════════════════════════════════
# Dataset Tests
# ══════════════════════════════════════════════════════════════════════════════


class TestSineDataset:
    @pytest.fixture(scope="class")
    def dataset(self):
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
        diffs = [
            not torch.allclose(dataset[i]["mixed_window"], dataset[i]["clean_window"])
            for i in range(20)
        ]
        assert all(diffs)


class TestGetDataloaders:
    def test_returns_two_loaders(self):
        tr, va = get_dataloaders(batch_size=32, samples_per_freq=50)
        assert isinstance(tr, DataLoader) and isinstance(va, DataLoader)

    def test_sizes_sum_to_total(self):
        tr, va = get_dataloaders(batch_size=32, samples_per_freq=50)
        assert len(tr.dataset) + len(va.dataset) == NUM_CLASSES * 50


# ══════════════════════════════════════════════════════════════════════════════
# Model Tests
# ══════════════════════════════════════════════════════════════════════════════


def _batch(B=4):
    mixed = torch.randn(B, WINDOW_LEN)
    label = torch.zeros(B, NUM_CLASSES)
    label[:, 0] = 1.0
    return mixed, label


class TestMLP:
    def test_output_shape(self):
        mixed, label = _batch()
        assert MLP()(mixed, label).shape == (4, OUTPUT_SIZE)

    def test_parameter_count_positive(self):
        assert count_parameters(MLP()) > 0

    def test_grad_flows(self):
        model = MLP()
        mixed, label = _batch()
        model(mixed, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None


class TestRNNModel:
    def test_output_shape(self):
        mixed, label = _batch()
        assert RNNModel()(mixed, label).shape == (4, OUTPUT_SIZE)

    def test_grad_flows(self):
        model = RNNModel()
        mixed, label = _batch()
        model(mixed, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None


class TestLSTMModel:
    def test_output_shape(self):
        mixed, label = _batch()
        assert LSTMModel()(mixed, label).shape == (4, OUTPUT_SIZE)

    def test_grad_flows(self):
        model = LSTMModel()
        mixed, label = _batch()
        model(mixed, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_lstm_more_params_than_rnn(self):
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
        tr, va = loaders
        model = ModelClass().to("cpu")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        losses = [
            train_epoch(model, tr, optimizer, criterion, torch.device("cpu"))
            for _ in range(5)
        ]
        assert losses[-1] < losses[0], (
            f"{ModelClass.__name__}: loss did not decrease "
            f"({losses[0]:.4f} -> {losses[-1]:.4f})"
        )

    @pytest.mark.parametrize("ModelClass", [MLP, RNNModel, LSTMModel])
    def test_evaluate_returns_float(self, loaders, ModelClass):
        _, va = loaders
        val_loss = evaluate(ModelClass(), va, torch.nn.MSELoss(), torch.device("cpu"))
        assert isinstance(val_loss, float) and val_loss >= 0.0
