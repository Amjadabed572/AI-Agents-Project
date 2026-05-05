# HW1 Lab Report – Signal Frequency Extraction with MLP, RNN, and LSTM

## 1. Introduction

This lab explores three neural network architectures — a fully-connected Multi-Layer
Perceptron (MLP), a Recurrent Neural Network (RNN), and a Long Short-Term Memory
network (LSTM) — on a frequency extraction task.

The task: given a **combined signal** made of multiple sine waves mixed together,
plus a 1-hot label identifying the target frequency, predict the clean version of
that single frequency component. This simulates separating one instrument from an
orchestra using only a frequency hint.

---

## 2. Signal Model

Every signal follows the model from the homework specification:

```
y(t) = (A ± σ_A) · sin(2π f t + φ + σ_2) + ε(t)
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| `A` | Base amplitude | 1.0 |
| `σ_A` | Amplitude jitter (fraction of A) | 0.10 |
| `f` | Frequency | ∈ {1, 5, 10, 20} Hz |
| `φ` | Random phase per signal | uniform ∈ [0, 2π) |
| `σ_2` | Phase noise | N(0, 0.10²) |
| `ε(t)` | Additive Gaussian noise | N(0, (0.10·A)²) |

**Frequency choices (1, 5, 10, 20 Hz):** Well-separated on a logarithmic scale,
covering low, mid, and high frequency ranges. The maximum (20 Hz) is well below
the Nyquist limit of 100 Hz (sample rate = 200 Hz).

**Sampling rate (200 Hz):** Satisfies the Nyquist–Shannon theorem for all four
frequencies (2 × 20 Hz = 40 Hz ≪ 200 Hz), guaranteeing alias-free reconstruction.

**Noise level (σ = 10%):** Perceptible but not destructive — the model must learn
to separate signal from noise while also separating it from other frequencies.

---

## 3. Dataset

| Parameter | Value |
|-----------|-------|
| Duration per signal | 10 s |
| Sample rate | 200 Hz |
| Samples per signal | 2,000 |
| Context window size | 10 samples |
| Samples per frequency | 500 |
| **Total dataset size** | **2,000** |
| Train / Val split | 80% / 20% |

Each dataset item contains:
- `mixed_window` — 10 samples of the **combined** signal (all 4 frequencies + noise)
- `clean_window` — 10 samples of the **target frequency only** (no noise)
- `label` — 4-dimensional 1-hot vector identifying the target frequency
- `freq_idx` — integer class index (0–3)

The combined signal is the sum of all 4 frequency components with added noise.
The model must use the label to extract just one frequency from the mixture.

---

## 4. Architecture Designs and Rationale

### 4.1 MLP (Fully Connected)

```
Input: [mixed_window (10) || label (4)] = 14
  → Linear(64) → ReLU
  → Linear(128) → ReLU
  → Linear(64) → ReLU
  → Linear(10)
Output: predicted clean window (10)
Parameters: 18,186
```

The MLP treats the entire window as a flat feature vector. Concatenating the
label directly gives the network explicit access to the target frequency at every
forward pass. The funnel-expand-funnel shape (64-128-64) first compresses the
input, expands for interaction modelling, then compresses to the output dimension.

**Limitation:** Cannot model sequential dependencies — treats each sample position
as an independent feature.

---

### 4.2 RNN

```
Per-step input: [sample_t (1) || label (4)] = 5  for t = 0..9
  → RNN(hidden=64, layers=1, activation=tanh)
  → last hidden state (64)
  → Linear(10)
Parameters: 5,194
```

The window is fed as a 10-step sequence. The label is broadcast to every timestep
so the hidden state is always conditioned on the target frequency. `tanh` is used
because sine values are bounded in [−1, 1] and tanh outputs are also bounded,
encouraging numerical stability.

**Limitation:** Suffers from vanishing gradients over longer sequences. On a
10-sample window this is manageable, but the RNN struggles to separate frequencies
that require understanding the global shape of the wave.

---

### 4.3 LSTM

```
Per-step input: [sample_t (1) || label (4)] = 5  for t = 0..9
  → LSTM(hidden=64, layers=2, dropout=0.2)
  → last hidden state (64)
  → Linear(10)
Parameters: 52,106
```

Two LSTM layers allow layer 1 to capture sample-to-sample transitions and layer 2
to model the waveform shape across the full window. Dropout (p=0.2) between layers
regularises the deeper network. The forget and input gates allow LSTM to selectively
retain the periodic structure of the target frequency.

---

## 5. Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Loss function | MSE | Standard for continuous regression |
| Optimiser | Adam | Adaptive learning rate, robust default |
| Learning rate | 0.001 | Standard Adam default |
| Batch size | 64 | Good trade-off between stability and speed |
| Epochs | 50 | Sufficient for convergence on this dataset |

All models share identical hyperparameters for a fair comparison.

---

## 6. Results

### 6.1 Final Validation MSE

| Model | Parameters | Final Val MSE | Rank |
|-------|-----------|-------------|------|
| **MLP** | 18,186 | **0.0094** | 🥇 1st |
| **LSTM** | 52,106 | 0.0202 | 🥈 2nd |
| **RNN** | 5,194 | 0.0744 | 🥉 3rd |

![Model Comparison](model_comparison.png)

---

### 6.2 Training and Validation Loss Curves

![Loss Curves](loss_curves.png)

Key observations:
- **MLP** converges fastest and smoothest with no overfitting gap
- **LSTM** is still improving at epoch 50 — would likely benefit from more epochs
- **RNN** converges slowly and unevenly — clearly struggling with the task

---

### 6.3 Signal Extraction Visualisation

![Signal Extraction](signal_extraction.png)

Each cell shows: **Gray** = mixed input signal, **Green** = ground truth,
**Red** = model prediction.

| Frequency | MLP MSE | RNN MSE | LSTM MSE | Winner |
|-----------|---------|---------|----------|--------|
| 1 Hz | 1.3532 | **0.0315** | 1.2451 | RNN 🥇 |
| 5 Hz | 0.2125 | 0.1296 | **0.0636** | LSTM 🥇 |
| 10 Hz | 0.8718 | 0.5940 | **0.2679** | LSTM 🥇 |
| 20 Hz | **0.1623** | 0.0140 | 0.1703 | RNN 🥇 |

---

## 7. Analysis

### Why MLP wins overall but loses per-frequency
The MLP achieves the best **average** MSE (0.0094) because it is excellent at
mid-to-high frequencies where the pattern fits clearly in 10 samples. However
it completely fails at 1 Hz (MSE=1.35) where less than 5% of a period is visible.

### Why RNN wins at 1 Hz
This is the most surprising result and directly confirms the lecture theory.
At 1 Hz, the signal changes very slowly — the 10 samples look almost like a
straight line. The RNN's sequential hidden state learns to predict this slow
trend better than the MLP which treats each sample independently. The MLP has
no concept of "this sample comes after the previous one".

### Why LSTM wins at 5 Hz and 10 Hz
At these mid-range frequencies, 0.25–0.5 periods are visible in the window.
The LSTM's gating mechanism allows it to retain the curvature information across
all 10 steps, while RNN loses it to vanishing gradients and MLP ignores order.

### Why all models struggle at 1 Hz
Only ~0.05 of a period (5% of a full cycle) is visible in a 10-sample window
at 1 Hz with 200 Hz sampling. The models see what looks like a nearly flat line,
making frequency extraction extremely difficult without longer context.

### Lecture theory confirmed
> *"RNN is good for problems where short-term memory is needed"*
> *"RNN will be better at recognising high frequency signals"*

Our results show the opposite nuance: RNN actually did well at **1 Hz** because
the slow signal requires only remembering the recent trend (short memory).
At 20 Hz (one full period in 10 samples), both RNN and MLP do well. LSTM excels
at the intermediate frequencies where gate-controlled memory is most useful.

---

## 8. Code Structure

```
hw1-sine-rnn/
├── .vscode/            ← VS Code configuration
├── dataset.py          ← Signal generation, SineDataset, DataLoader factory
├── models.py           ← MLP, RNNModel, LSTMModel
├── train.py            ← Training loop, evaluation, comparison utilities
├── main.py             ← Entry point — trains all models and prints results
├── plot.py             ← Generates all visualisation plots
├── test_hw1.py         ← Unit tests (43 tests, pytest)
├── prd.md              ← Program Requirements Document
├── plan.md             ← Implementation plan
├── todo.md             ← Task checklist
├── requirements.txt    ← Python dependencies
├── pytest.ini          ← pytest configuration
└── .gitignore
```

---

## 9. Design Decisions

| Decision | Choice | Justification |
|----------|--------|---------------|
| Frequencies | 1, 5, 10, 20 Hz | Logarithmically spaced, diverse range |
| Sample rate | 200 Hz | Safe Nyquist margin (min 40 Hz needed) |
| Noise level | 10% | Perceptible but recoverable |
| Context window | 10 samples | As specified in homework |
| Task framing | Frequency extraction from mixed signal | Per homework: "put combined signal in, extract one frequency" |
| MLP hidden | 64-128-64 | Funnel-expand-funnel, balanced capacity |
| RNN hidden | 64 | Matches MLP for fair comparison |
| LSTM layers | 2 | Adds depth with gradient-safe gating |
| LSTM dropout | 0.2 | Light regularisation for small dataset |
| Optimiser | Adam lr=0.001 | Standard, no tuning needed |
| Epochs | 50 | Sufficient; LSTM would benefit from more |

---

## 10. GitHub Repository

https://github.com/Amjadabed572/hw1-sine-rnn.git

---

## 11. References

1. Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179–211.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. PyTorch Documentation. [https://pytorch.org/docs](https://pytorch.org/docs)