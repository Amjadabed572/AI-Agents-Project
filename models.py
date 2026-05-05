"""
models.py - Model Definitions for HW1
Three architectures: MLP, RNN, LSTM.
Input: combined mixed signal window (10 samples) + 1-hot frequency label (4).
Output: clean window of the target frequency only (10 samples).
"""

import torch
import torch.nn as nn
from dataset import WINDOW_LEN, NUM_CLASSES

INPUT_SIZE = WINDOW_LEN + NUM_CLASSES  # 10 + 4 = 14
OUTPUT_SIZE = WINDOW_LEN  # 10


# ── 1. MLP ─────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """
    Fully connected network for frequency extraction.

    Architecture: Input(14) -> 64 -> 128 -> 64 -> Output(10)

    The mixed window and label are concatenated into a flat vector.
    The network learns to use the label to suppress all frequencies
    except the target one.

    Choice: ReLU activations, funnel-expand-funnel shape (64-128-64)
    gives enough capacity without over-parameterising.
    """

    def __init__(
        self,
        input_size: int = INPUT_SIZE,
        hidden_sizes: tuple = (64, 128, 64),
        output_size: int = OUTPUT_SIZE,
    ):
        super().__init__()
        layers = []
        in_dim = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, mixed_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mixed_window : (B, WINDOW_LEN) - combined signal window
        label        : (B, NUM_CLASSES) - 1-hot target frequency

        Returns (B, OUTPUT_SIZE) - predicted clean window
        """
        x = torch.cat([mixed_window, label], dim=-1)
        return self.net(x)


# ── 2. RNN ─────────────────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    """
    Elman RNN for frequency extraction.

    Architecture: per-step input (1+4=5) -> RNN(hidden=64) -> Linear(10)

    The mixed signal is fed sample-by-sample. The label is appended at
    every timestep so the hidden state is always conditioned on the
    target frequency.

    Choice: tanh activation (default) suits bounded sine values in [-A, A].
    Single layer keeps it comparable to MLP parameter count.
    """

    def __init__(
        self,
        input_size: int = 1 + NUM_CLASSES,
        hidden_size: int = 64,
        num_layers: int = 1,
        output_size: int = OUTPUT_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, mixed_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mixed_window : (B, WINDOW_LEN)
        label        : (B, NUM_CLASSES)
        """
        B = mixed_window.size(0)
        label_exp = label.unsqueeze(1).expand(B, WINDOW_LEN, -1)
        x = torch.cat([mixed_window.unsqueeze(-1), label_exp], dim=-1)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)
        return self.fc(out[:, -1, :])


# ── 3. LSTM ────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    Two-layer LSTM for frequency extraction.

    Architecture: per-step input (1+4=5) -> LSTM(hidden=64, layers=2) -> Linear(10)

    Two layers allow layer 1 to capture sample-to-sample transitions and
    layer 2 to model the waveform shape across the full window.
    Dropout(0.2) between layers regularises the deeper network.

    Choice: LSTM gates protect the cell state from vanishing gradients,
    making it better than plain RNN at retaining the periodic structure
    of low-frequency signals across the 10-sample window.
    """

    def __init__(
        self,
        input_size: int = 1 + NUM_CLASSES,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = OUTPUT_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, mixed_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        mixed_window : (B, WINDOW_LEN)
        label        : (B, NUM_CLASSES)
        """
        B = mixed_window.size(0)
        label_exp = label.unsqueeze(1).expand(B, WINDOW_LEN, -1)
        x = torch.cat([mixed_window.unsqueeze(-1), label_exp], dim=-1)
        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])
