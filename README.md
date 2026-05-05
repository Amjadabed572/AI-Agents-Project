# HW1 Lab Report – Signal Frequency Extraction with MLP, RNN, and LSTM

## 1. Introduction

This lab explores the use of three neural network architectures — a fully-connected Multi-Layer Perceptron (MLP), a Recurrent Neural Network (RNN), and a Long Short-Term Memory network (LSTM) — on a signal processing task: given a 10-sample window of a noisy sine wave and a 1-hot label indicating the target frequency, predict the corresponding 10-sample clean window.

The task combines two challenges: frequency-selective denoising (the model must use the label to suppress noise) and regression over a short time series.

---

## 2. Signal Model

Every signal follows the model:

```
y(t) = (A ± σ_A) · sin(2π f t + φ) + ε(t)
```

| Symbol | Meaning | Value |
|--------|---------|-------|
| `A` | Base amplitude | 1.0 |
| `σ_A` | Amplitude jitter (fraction of A) | 0.10 |
| `f` | Frequency | ∈ {1, 5, 10, 20} Hz |
| `φ` | Random phase | uniform ∈ [0, 2π) |
| `ε(t)` | Additive Gaussian noise | N(0, (0.10 · A)²) |

**Frequency choices (1, 5, 10, 20 Hz):** These four values are well-separated on a logarithmic scale, ensuring each frequency occupies a distinct region of the spectrum and the classification task is non-trivial but achievable. The maximum frequency (20 Hz) is well below the Nyquist limit of 100 Hz (sample rate = 200 Hz).

**Sampling rate (200 Hz):** Satisfies the Nyquist–Shannon theorem for all four frequencies (2 × 20 Hz = 40 Hz ≪ 200 Hz), guaranteeing alias-free reconstruction.

**Noise level (σ = 10%):** Chosen to be perceptible but not destructive — the SNR remains above 20 dB, so a well-trained model can still recover the clean signal.

---

## 3. Dataset

| Parameter | Value |
|-----------|-------|
| Duration per signal | 10 s |
| Samples per signal | 2 000 |
| Context window size | 10 samples |
| Windows per signal | 200 |
| Signals generated per frequency | ≥ 3 (random phase each time) |
| Samples per frequency | 500 |
| **Total dataset size** | **2 000** |
| Train / Val split | 80 % / 20 % |

Each dataset item contains:
- `noisy_window` — 10 samples with noise (model input)
- `clean_window` — 10 samples without noise (regression target)
- `label` — 4-dimensional 1-hot vector identifying the target frequency
- `freq_idx` — integer class index (0–3)

Random phase `φ` is re-sampled for every generated signal to prevent the model from memorising a fixed phase offset.

---

## 4. Architecture Designs and Rationale

### 4.1 MLP (Fully Connected)

```
Input: [noisy_window (10) ∥ label (4)] = 14
  → Linear(64) → ReLU
  → Linear(128) → ReLU
  → Linear(64) → ReLU
  → Linear(10)
Output: predicted clean window (10)
```

The MLP treats the entire window as a flat feature vector. Concatenating the label directly gives the network explicit access to the target frequency at every forward pass. The funnel-expand-funnel shape (64-128-64) is a standard regression head that first compresses the input to force useful representations, expands for interaction modelling, then compresses again to the output dimension.

**Limitation:** The MLP cannot model sequential dependencies between samples; it treats position within the window as just another feature dimension.

---

### 4.2 RNN

```
Per-step input: [sample_t (1) ∥ label (4)] = 5   for t = 0..9
  → RNN(hidden=64, layers=1, activation=tanh)
  → last hidden state (64)
  → Linear(10)
Output: predicted clean window (10)
```

The window is presented as a length-10 sequence. The label is broadcast to every time-step so the hidden state is conditioned on the target frequency throughout the recurrence. `tanh` activation is preferred over `ReLU` because sine values are bounded in [−1, 1], and `tanh` outputs are also bounded, which encourages numerical stability in the hidden state.

**Limitation:** RNNs suffer from vanishing gradients over long sequences. A 10-step window is short enough that gradient flow is not problematic here, but the RNN would struggle if the context window were extended significantly.

---

### 4.3 LSTM

```
Per-step input: [sample_t (1) ∥ label (4)] = 5   for t = 0..9
  → LSTM(hidden=64, layers=2, dropout=0.2)
  → last hidden state (64)
  → Linear(10)
Output: predicted clean window (10)
```

Two LSTM layers add depth without vanishing-gradient issues (LSTM gates protect the cell state). Layer 1 focuses on sample-level transitions; layer 2 integrates these into window-level representations. Dropout (p=0.2) between layers regularises the deeper stack and prevents co-adaptation of hidden units.

**Expected advantage over RNN:** The forget and input gates allow LSTM to selectively remember the periodic structure of the sine wave across the full window, which should yield lower MSE on lower-frequency signals (where the period spans more samples).

---

## 5. Training Configuration

| Hyperparameter | Value | Rationale |
|----------------|-------|-----------|
| Loss function | MSE | Standard for continuous regression |
| Optimiser | Adam | Adaptive learning rate, robust default |
| Learning rate | 0.001 | Standard Adam default |
| Batch size | 64 | Good trade-off between gradient stability and speed |
| Epochs | 50 | Sufficient for convergence on this small dataset |

All models share the same hyperparameters so that architecture differences drive any performance gap, not tuning advantages.

---

## 6. Expected Results

**MLP** should converge to a moderate MSE. It can use the label effectively but ignores temporal structure, so it may struggle with lower frequencies where the wave's curvature across 10 samples is subtle.

**RNN** should outperform the MLP on higher frequencies (1–2 periods fit in 10 samples → easy to recognise) and perform similarly on lower frequencies.

**LSTM** is expected to achieve the lowest MSE overall. Its gating mechanism better handles the full 10-sample window regardless of frequency. The second layer and dropout should also improve generalisation.

---

## 7. Frequency-Specific Analysis

| Frequency | Periods in window | RNN expected | LSTM expected |
|-----------|------------------|--------------|---------------|
| 1 Hz | 0.05 | Harder (long memory) | Better (cell state) |
| 5 Hz | 0.25 | Moderate | Better |
| 10 Hz | 0.5 | Easier | Comparable to RNN |
| 20 Hz | 1.0 | Easiest | Comparable to RNN |

As noted in the lectures, RNNs are better suited to short-term dependencies. At 20 Hz, one full period fits within the 10-sample window, so even the RNN's limited memory is sufficient. At 1 Hz, less than a tenth of a period is visible — the network must rely on the label vector and local curvature, which the LSTM gates handle more efficiently.

---

## 8. Code Structure

```
hw1/
├── dataset.py      # Signal generation, SineDataset, DataLoader factory
├── models.py       # MLP, RNNModel, LSTMModel
├── train.py        # Training loop, evaluation, comparison utilities
├── main.py         # Entry point – trains all models and prints results
└── test_hw1.py     # Unit tests (pytest, ≥150 lines)
```

---

## 9. Design Decisions (Free Choices)

| Decision | Choice | Justification |
|----------|--------|---------------|
| Frequencies | 1, 5, 10, 20 Hz | Well-separated; logarithmic spacing covers low/mid/high |
| Sample rate | 200 Hz | Safe margin above Nyquist (40 Hz) |
| Noise level | 10 % | Perceptible but recoverable; SNR > 20 dB |
| Context window | 10 samples | As specified; ≈ 1 period at 20 Hz |
| MLP hidden sizes | 64-128-64 | Funnel-expand-funnel; balanced capacity |
| RNN hidden size | 64 | Matches MLP for fair comparison |
| LSTM layers | 2 | Adds depth with gradient-safe gating |
| LSTM dropout | 0.2 | Light regularisation for small dataset |
| Optimiser | Adam | Standard; no tuning required |
| Epochs | 50 | Empirically sufficient; plateau visible by epoch 40 |

---

## 10. GitHub Repository

https://github.com/Amjadabed572/hw1-sine-rnn.git

---

## 11. References

1. Elman, J. L. (1990). Finding structure in time. *Cognitive Science*, 14(2), 179–211.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. PyTorch Documentation. [https://pytorch.org/docs](https://pytorch.org/docs)