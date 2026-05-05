"""
test_models.py - Unit tests for model architectures and training loop.
Run with: pytest tests/ -v
"""

import pytest
import torch

from hw1.constants import NUM_CLASSES, WINDOW_LEN
from hw1.dataset import get_dataloaders
from hw1.models import MLP, OUTPUT_SIZE, LSTMModel, RNNModel
from hw1.train import count_parameters, evaluate, train_epoch


def _batch(B: int = 4) -> tuple:  # type: ignore[type-arg]
    """Create a dummy batch of mixed windows and labels."""
    mixed = torch.randn(B, WINDOW_LEN)
    label = torch.zeros(B, NUM_CLASSES)
    label[:, 0] = 1.0
    return mixed, label


class TestMLP:
    """Tests for MLP model."""

    def test_output_shape(self):
        """MLP output shape must be (B, OUTPUT_SIZE)."""
        mixed, label = _batch()
        assert MLP()(mixed, label).shape == (4, OUTPUT_SIZE)

    def test_parameter_count_positive(self):
        """MLP must have trainable parameters."""
        assert count_parameters(MLP()) > 0

    def test_grad_flows(self):
        """Gradients must reach all MLP parameters."""
        model = MLP()
        mixed, label = _batch()
        model(mixed, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None


class TestRNNModel:
    """Tests for RNN model."""

    def test_output_shape(self):
        """RNN output shape must be (B, OUTPUT_SIZE)."""
        mixed, label = _batch()
        assert RNNModel()(mixed, label).shape == (4, OUTPUT_SIZE)

    def test_grad_flows(self):
        """Gradients must reach all RNN parameters."""
        model = RNNModel()
        mixed, label = _batch()
        model(mixed, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None


class TestLSTMModel:
    """Tests for LSTM model."""

    def test_output_shape(self):
        """LSTM output shape must be (B, OUTPUT_SIZE)."""
        mixed, label = _batch()
        assert LSTMModel()(mixed, label).shape == (4, OUTPUT_SIZE)

    def test_grad_flows(self):
        """Gradients must reach all LSTM parameters."""
        model = LSTMModel()
        mixed, label = _batch()
        model(mixed, label).sum().backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_lstm_more_params_than_rnn(self):
        """LSTM has more parameters than RNN due to 4 gates vs 1."""
        assert count_parameters(LSTMModel()) > count_parameters(RNNModel())


class TestTraining:
    """Tests for training loop and evaluation."""

    @pytest.fixture(scope="class")
    def loaders(self):  # type: ignore[override]
        """Shared small DataLoaders fixture."""
        return get_dataloaders(batch_size=32, samples_per_freq=50, seed=7)

    @pytest.mark.parametrize("ModelClass", [MLP, RNNModel, LSTMModel])
    def test_loss_decreases(self, loaders, ModelClass) -> None:
        """Training loss must decrease over 5 epochs for all models."""
        tr, _ = loaders
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
    def test_evaluate_returns_float(self, loaders, ModelClass) -> None:
        """Evaluate must return a non-negative float."""
        _, va = loaders
        val_loss = evaluate(ModelClass(), va, torch.nn.MSELoss(), torch.device("cpu"))
        assert isinstance(val_loss, float) and val_loss >= 0.0


def test_version():
    """Version string must be non-empty."""
    from hw1.shared.version import get_version

    assert get_version() == "1.00"
