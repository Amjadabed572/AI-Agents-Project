# PRD - Program Requirements Document
## HW1: Signal Frequency Extraction with MLP, RNN, and LSTM

---

## 1. Overview
Build and compare three neural network architectures (MLP, RNN, LSTM) on a signal
processing task: given a noisy sine wave window and a frequency label, predict the
clean signal window.

---

## 2. Background
Recurrent Neural Networks (RNNs) are designed to handle sequential data by
maintaining a memory of previous inputs. This exercise demonstrates the strengths
and limitations of MLP, RNN, and LSTM on a controlled signal processing task,
directly applying concepts from the lecture on sequence modelling.

---

## 3. Goals
- Implement a sine wave dataset generator with noise
- Train and compare MLP, RNN, and LSTM on signal denoising
- Evaluate using MSE loss on a held-out validation set
- Document findings in a lab report (README.md)

---

## 4. Dataset Requirements
- **Frequencies:** 4 known frequencies — 1 Hz, 5 Hz, 10 Hz, 20 Hz
- **Signal model:** y(t) = (A ± σ_A) · sin(2π f t + φ) + ε(t)
  - A = 1.0 (amplitude)
  - σ = 0.10 (noise as fraction of A)
  - φ = random phase per signal
- **Sample rate:** 200 Hz (satisfies Nyquist for 20 Hz max)
- **Signal duration:** 10 seconds per generated signal
- **Context window:** 10 samples
- **Per frequency:** 500 windows (2000 total)
- **Train/Val split:** 80% / 20%
- Each dataset entry contains:
  - 1-hot frequency label (size 4)
  - 10 noisy samples (model input)
  - 10 clean samples (regression target)

---

## 5. Model Requirements

### 5.1 MLP (Fully Connected)
- Input: concatenation of noisy window (10) + label (4) = 14
- Hidden layers: 64 → 128 → 64 with ReLU activations
- Output: 10 (predicted clean window)

### 5.2 RNN
- Input per step: 1 sample + label (4) = 5
- Single layer, hidden size 64, tanh activation
- Output from last hidden state → Linear(10)

### 5.3 LSTM
- Input per step: 1 sample + label (4) = 5
- 2 layers, hidden size 64, dropout 0.2
- Output from last hidden state → Linear(10)

---

## 6. Training Requirements
- Loss function: MSE (Mean Squared Error)
- Optimiser: Adam, lr = 0.001
- Batch size: 64
- Epochs: 50
- All models trained with identical hyperparameters for fair comparison

---

## 7. Code Requirements
- Maximum 150 lines per Python file
- Unit tests required (pytest)
- Virtual environment managed with UV
- All work done via terminal (no Cursor, no Copilot)
- GitHub repository with commit history

---

## 8. Deliverables
- `dataset.py` — signal generation and dataset
- `models.py` — MLP, RNN, LSTM definitions
- `train.py` — training loop and evaluation
- `main.py` — entry point
- `test_hw1.py` — unit tests (37 tests)
- `README.md` — lab report with results and GitHub link
- `prd.md` — this document
- `plan.md` — implementation plan
- `todo.md` — task list
- PDF submission with GitHub link

---

## 9. Success Criteria
- All unit tests pass (37/37)
- All three models converge (loss decreases over epochs)
- Final Val MSE < 0.01 for all models
- Code is clean, documented, and under 150 lines per file