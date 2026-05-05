# HW1 — Signal Frequency Extraction with MLP, RNN, and LSTM

**Group:** amj-naji  
**GitHub:** https://github.com/Amjadabed572/hw1-sine-rnn  
**Version:** 1.00

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Amjadabed572/hw1-sine-rnn.git
cd hw1-sine-rnn

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
uv pip install torch numpy matplotlib pytest
```

## Usage

```bash
# Train all models and print results
python src/main.py

# Generate all plots (saved to assets/)
python src/plot.py

# Run all tests
pytest tests/ -v
```

---

## 1. Introduction

This lab explores three neural network architectures — MLP, RNN, and LSTM — on a
frequency extraction task: given a **combined signal** made of multiple sine waves
mixed together plus a 1-hot label, predict the clean version of the target frequency.

---

## 2. Signal Model

```
y(t) = (A ± σ_A) · sin(2π f t + φ + σ_2) + ε(t)
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| A | Base amplitude | 1.0 |
| σ_A | Amplitude jitter | 10% of A |
| f | Frequency | ∈ {1, 5, 10, 20} Hz |
| φ | Random phase | uniform ∈ [0, 2π) |
| σ_2 | Phase noise | N(0, 0.10²) |
| ε(t) | Additive noise | N(0, (0.10·A)²) |

**Frequencies (1, 5, 10, 20 Hz):** Logarithmically spaced, diverse range.
**Sample rate (200 Hz):** Satisfies Nyquist for all frequencies (min 40 Hz needed).
**Noise (10%):** Perceptible but recoverable.

---

## 3. Project Structure

```
hw1-sine-rnn/
├── src/hw1/            ← Python package
│   ├── constants.py    ← Shared constants
│   ├── signals.py      ← Signal generation
│   ├── dataset.py      ← SineDataset, DataLoaders
│   ├── models.py       ← MLP, RNN, LSTM
│   ├── train.py        ← Training loop
│   ├── plot_losses.py  ← Loss visualizations
│   ├── plot_signals.py ← Signal visualizations
│   └── shared/version.py
├── tests/unit/         ← 43 unit tests
├── docs/               ← PRD, PLAN, TODO, PROMPT_LOG, per-model PRDs
├── config/             ← setup.json, rate_limits.json
├── assets/             ← Generated plots
└── src/main.py         ← Entry point
```

All Python files comply with the **150 code-line limit**.

---

## 4. Model Architectures

### MLP — 18,186 parameters
```
Input(14) -> Linear(64)->ReLU -> Linear(128)->ReLU -> Linear(64)->ReLU -> Linear(10)
```

### RNN — 5,194 parameters
```
Per-step(5) -> RNN(hidden=64, tanh) -> last hidden -> Linear(10)
```

### LSTM — 52,106 parameters
```
Per-step(5) -> LSTM(hidden=64, layers=2, dropout=0.2) -> last hidden -> Linear(10)
```

---

## 5. Results

### Final Validation MSE

| Model | Parameters | Final Val MSE | Rank |
|-------|-----------|-------------|------|
| **MLP** | 18,186 | **0.0094** | 🥇 |
| **LSTM** | 52,106 | 0.0202 | 🥈 |
| **RNN** | 5,194 | 0.0744 | 🥉 |

![alt text](assets/model_comparison.png)

### Loss Curves

![alt text](assets/loss_curves.png)

### Signal Extraction

![alt text](assets/signal_extraction.png)

| Frequency | MLP MSE | RNN MSE | LSTM MSE | Winner |
|-----------|---------|---------|----------|--------|
| 1 Hz | 1.3532 | **0.0315** | 1.2451 | RNN |
| 5 Hz | 0.2125 | 0.1296 | **0.0636** | LSTM |
| 10 Hz | 0.8718 | 0.5940 | **0.2679** | LSTM |
| 20 Hz | 0.1623 | **0.0140** | 0.1703 | RNN |

---

## 6. Analysis

**MLP wins overall** (best average MSE) but fails at 1 Hz — treats samples
as independent features, ignoring temporal order.

**RNN wins at 1 Hz** — slow signals look like straight lines; RNN's sequential
memory tracks the trend. Confirms lecture: RNN good at short-term patterns.

**LSTM wins at 5 and 10 Hz** — gating retains curvature information across
all 10 steps, outperforming both MLP and RNN at mid-range frequencies.

---

## 7. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Frequencies | 1, 5, 10, 20 Hz | Logarithmically spaced |
| Sample rate | 200 Hz | Safe Nyquist margin |
| Noise | 10% | Perceptible but recoverable |
| Task | Extract from combined signal | Per homework spec |
| Label injection | Every RNN/LSTM timestep | Conditions hidden state on target |
| Epochs | 50 | Sufficient; LSTM still improving |

---

## 8. References

1. Elman (1990). Finding structure in time. *Cognitive Science*, 14(2).
2. Hochreiter & Schmidhuber (1997). Long short-term memory. *Neural Computation*, 9(8).
3. Goodfellow et al. (2016). *Deep Learning*. MIT Press.
4. PyTorch Documentation. https://pytorch.org/docs