"""
models.py - Neural network model definitions for hw1.
Three architectures: MLP, RNN, LSTM for frequency extraction task.
Input: combined mixed signal window + 1-hot label -> clean target window.
"""

import torch
import torch.nn as nn
from hw1.constants import WINDOW_LEN, NUM_CLASSES

INPUT_SIZE = WINDOW_LEN + NUM_CLASSES  # 14: 10 samples + 4 label
OUTPUT_SIZE = WINDOW_LEN  # 10: predicted clean window


class MLP(nn.Module):
    """
    Fully connected network for frequency extraction.

    Architecture: Input(14) -> Linear(64) -> ReLU -> Linear(128)
                  -> ReLU -> Linear(64) -> ReLU -> Linear(10)

    Label is concatenated to the window so the network always knows
    which frequency to extract. Funnel-expand-funnel shape (64-128-64)
    balances capacity without over-parameterising.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_sizes: tuple = (64, 128, 64),
        output_size: int = OUTPUT_SIZE,
    ):
        """Initialise MLP with configurable hidden layer sizes."""
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, mixed_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Forward pass: concatenate window + label, run through MLP."""
        return self.net(torch.cat([mixed_window, label], dim=-1))


class RNNModel(nn.Module):
    """
    Elman RNN for frequency extraction.

    Architecture: per-step input (1+4=5) -> RNN(hidden=64) -> Linear(10)

    Label is broadcast to every timestep so the hidden state is always
    conditioned on the target frequency. tanh suits bounded sine values.
    """

    def __init__(
        self,
        input_size: int = 1 + NUM_CLASSES,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = OUTPUT_SIZE,
    ):
        """Initialise RNN with configurable hidden size and layers."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, mixed_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Forward pass: feed window sequentially with label at each step."""
        B = mixed_window.size(0)
        label_exp = label.unsqueeze(1).expand(B, WINDOW_LEN, -1)
        x = torch.cat([mixed_window.unsqueeze(-1), label_exp], dim=-1)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


class LSTMModel(nn.Module):
    """
    Two-layer LSTM for frequency extraction.

    Architecture: per-step input (1+4=5) -> LSTM(hidden=64, layers=2) -> Linear(10)

    Two layers: layer 1 captures sample transitions, layer 2 models waveform shape.
    Dropout (0.2) regularises the deeper network. Gating prevents vanishing gradients.
    """

    def __init__(
        self,
        input_size: int = 1 + NUM_CLASSES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = OUTPUT_SIZE,
    ):
        """Initialise LSTM with configurable hidden size, layers, and dropout."""
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, mixed_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """Forward pass: feed window sequentially with label at each step."""
        B = mixed_window.size(0)
        label_exp = label.unsqueeze(1).expand(B, WINDOW_LEN, -1)
        x = torch.cat([mixed_window.unsqueeze(-1), label_exp], dim=-1)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
