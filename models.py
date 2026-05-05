"""
models.py - Model Definitions for HW1
Three architectures: Fully Connected MLP, RNN, LSTM
All take a window of 10 samples + 1-hot freq label as input,
and output a predicted clean window of 10 samples (regression / denoising task).
"""

import torch
import torch.nn as nn
from dataset import WINDOW_LEN, NUM_CLASSES


# ── Shared input size ──────────────────────────────────────────────────────────
# Input = [noisy_window (10) + one_hot_label (4)] = 14
INPUT_SIZE  = WINDOW_LEN + NUM_CLASSES   # 14
OUTPUT_SIZE = WINDOW_LEN                  # 10  (predict clean window)


# ── 1. Fully Connected MLP ─────────────────────────────────────────────────────
class MLP(nn.Module):
    """
    Three-hidden-layer fully-connected network.

    Architecture
    ------------
    Input(14) → Linear(64) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(10)

    Choice rationale
    ----------------
    Hidden sizes 64-128-64 give enough capacity for a 10-sample regression
    without over-parameterising.  ReLU is standard for regression heads.
    No BatchNorm here to keep the architecture transparent for comparison.
    """

    def __init__(
        self,
        input_size: int  = INPUT_SIZE,
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

    def forward(self, noisy_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_window : (B, WINDOW_LEN)
        label        : (B, NUM_CLASSES)

        Returns
        -------
        (B, OUTPUT_SIZE)  – predicted clean window
        """
        x = torch.cat([noisy_window, label], dim=-1)   # (B, 14)
        return self.net(x)


# ── 2. RNN ─────────────────────────────────────────────────────────────────────
class RNNModel(nn.Module):
    """
    Single-layer Elman RNN followed by a linear projection.

    The window is treated as a time-series of individual samples:
    sequence length = WINDOW_LEN, feature dim = 1 + NUM_CLASSES (label appended
    to every time-step so the network always knows which frequency to extract).

    Architecture
    ------------
    Input(1+4=5) per step → RNN(hidden=64) → last hidden → Linear(10)

    Choice rationale
    ----------------
    Appending the label at every step helps the RNN gate its memory
    appropriately for the target frequency.  hidden_size=64 matches
    the MLP for a fair comparison.  tanh activation (default) suits
    bounded sine-wave values.
    """

    def __init__(
        self,
        input_size: int  = 1 + NUM_CLASSES,   # per-step: 1 sample + label
        hidden_size: int = 64,
        num_layers: int  = 1,
        output_size: int = OUTPUT_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.rnn = nn.RNN(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, noisy_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_window : (B, WINDOW_LEN)
        label        : (B, NUM_CLASSES)
        """
        B = noisy_window.size(0)
        # Expand label to every time-step → (B, WINDOW_LEN, NUM_CLASSES)
        label_expanded = label.unsqueeze(1).expand(B, WINDOW_LEN, -1)
        # Reshape window → (B, WINDOW_LEN, 1)
        x = noisy_window.unsqueeze(-1)
        # Concatenate → (B, WINDOW_LEN, 1+NUM_CLASSES)
        x = torch.cat([x, label_expanded], dim=-1)

        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.rnn(x, h0)          # out: (B, WINDOW_LEN, hidden_size)
        last   = out[:, -1, :]            # (B, hidden_size)
        return self.fc(last)              # (B, OUTPUT_SIZE)


# ── 3. LSTM ────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    Two-layer LSTM followed by a linear projection.

    Architecture
    ------------
    Input(1+4=5) per step → LSTM(hidden=64, layers=2) → last hidden → Linear(10)

    Choice rationale
    ----------------
    Two layers allow the LSTM to capture both local sample-to-sample dynamics
    (layer 1) and global shape of the waveform within the window (layer 2).
    Dropout(0.2) between layers regularises the deeper network.
    Same hidden_size as RNN keeps the comparison fair while the gating
    mechanism provides the extra capacity.
    """

    def __init__(
        self,
        input_size: int  = 1 + NUM_CLASSES,
        hidden_size: int = 64,
        num_layers: int  = 2,
        dropout: float   = 0.2,
        output_size: int = OUTPUT_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, noisy_window: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        noisy_window : (B, WINDOW_LEN)
        label        : (B, NUM_CLASSES)
        """
        B = noisy_window.size(0)
        label_expanded = label.unsqueeze(1).expand(B, WINDOW_LEN, -1)
        x = noisy_window.unsqueeze(-1)
        x = torch.cat([x, label_expanded], dim=-1)

        h0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, B, self.hidden_size, device=x.device)
        out, _ = self.lstm(x, (h0, c0))  # (B, WINDOW_LEN, hidden_size)
        last   = out[:, -1, :]
        return self.fc(last)             # (B, OUTPUT_SIZE)