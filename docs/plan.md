# PRD — Product Requirements Document
## HW1: Signal Frequency Extraction with MLP, RNN, and LSTM

---

## 1. Overview
Build and compare three neural network architectures (MLP, RNN, LSTM) on a
frequency extraction task: given a combined noisy sine wave window and a 1-hot
frequency label, predict the clean version of the target frequency component.

---

## 2. Background
Recurrent Neural Networks maintain memory of previous inputs, making them suited
for sequential data. This exercise demonstrates the strengths and limitations of
MLP, RNN, and LSTM on a controlled signal processing task, applying concepts from
the lecture on sequence modelling and LSTM architecture.

---

## 3. Goals
- Implement a combined sine wave dataset generator with noise
- Train and compare MLP, RNN, and LSTM on frequency extraction
- Evaluate all models using MSE loss on a held-out validation set
- Document findings in a detailed lab report (README.md)

---

## 4. Signal Model
```
y(t) = (A ± σ_A) · sin(2π f t + φ + σ_2) + ε(t)
```
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Amplitude A | 1.0 | Unit amplitude for normalised comparison |
| Noise σ | 10% of A | Perceptible but recoverable |
| Frequencies | 1, 5, 10, 20 Hz | Logarithmically spaced |
| Sample rate | 200 Hz | Satisfies Nyquist (min 40 Hz needed) |
| Duration | 10 seconds | Sufficient windows per signal |
| Window size | 10 samples | As specified in homework |

---

## 5. Dataset Requirements
- Combined signal = sum of all 4 frequency components + noise
- Each dataset item: `(mixed_window, clean_window, label, freq_idx)`
- 500 windows per frequency → 2,000 total
- 80% train / 20% validation split
- Reproducible via fixed random seed

---

## 6. Model Requirements

### MLP
- Input: `[mixed_window (10) || label (4)]` = 14
- Architecture: Linear(64) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(10)
- Output: predicted clean window (10 samples)

### RNN
- Per-step input: `[sample_t (1) || label (4)]` = 5
- Single layer, hidden size 64, tanh activation
- Output: last hidden state → Linear(10)

### LSTM
- Per-step input: `[sample_t (1) || label (4)]` = 5
- 2 layers, hidden size 64, dropout 0.2
- Output: last hidden state → Linear(10)

---

## 7. Training Requirements
- Loss function: MSE
- Optimiser: Adam, lr = 0.001
- Batch size: 64, Epochs: 50
- Identical hyperparameters for all models (fair comparison)

---

## 8. Code Requirements
- Maximum 150 code lines per Python file
- Unit tests with pytest (43 tests across 2 test files)
- All constants in `constants.py`
- Signal generation in `signals.py`
- Virtual environment managed with UV

---

## 9. Deliverables
| File | Purpose |
|------|---------|
| `constants.py` | Shared project constants |
| `signals.py` | Signal generation functions |
| `dataset.py` | SineDataset and DataLoader factory |
| `models.py` | MLP, RNN, LSTM definitions |
| `train.py` | Training loop and evaluation |
| `main.py` | Entry point |
| `plot.py` | Plot entry point |
| `plot_losses.py` | Loss curve visualizations |
| `plot_signals.py` | Signal extraction visualization |
| `test_signals.py` | Signal and dataset tests (29 tests) |
| `test_models.py` | Model and training tests (14 tests) |
| `README.md` | Lab report with results |
| `docs/PRD.md` | This document |
| `docs/PLAN.md` | Implementation plan |
| `docs/TODO.md` | Task checklist |

---

## 10. Success Criteria
- All 43 unit tests pass
- All three models converge (loss decreases over epochs)
- Final Val MSE documented and analysed
- Code is clean, documented, under 150 code lines per file
- GitHub repo is public with meaningful commit history