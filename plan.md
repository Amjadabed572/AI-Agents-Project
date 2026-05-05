# Plan - Implementation Strategy
## HW1: Signal Frequency Extraction with MLP, RNN, and LSTM

---

## Phase 1: Environment Setup
- Install UV for virtual environment management
- Create and activate `.venv`
- Install dependencies: torch, numpy, pytest, matplotlib
- Set up GitHub repository
- Configure VS Code with settings, launch, and extensions

---

## Phase 2: Dataset Implementation (`dataset.py`)
- Implement `generate_sine()` — generates pure or noisy sine wave
  - Parameters: frequency, duration, sample_rate, amplitude, phase, noise_std
  - Signal model: y(t) = (A ± σ_A) · sin(2π f t + φ) + ε(t)
- Implement `one_hot()` — converts class index to 1-hot vector
- Implement `extract_windows()` — slices signal into non-overlapping windows
- Implement `SineDataset` — PyTorch Dataset class
  - Generates 500 windows per frequency (2000 total)
  - Each item: noisy_window, clean_window, label, freq_idx
- Implement `get_dataloaders()` — train/val split factory
- Verify: dataset length, item shapes, label validity

---

## Phase 3: Model Implementation (`models.py`)
- Implement `MLP`
  - Flat input: [noisy_window ∥ label] = 14
  - Architecture: 64 → 128 → 64 → 10
  - Activation: ReLU
- Implement `RNNModel`
  - Sequential input: (sample_t ∥ label) per step
  - Single layer RNN, hidden=64, tanh
  - Last hidden state → Linear(10)
- Implement `LSTMModel`
  - Sequential input: (sample_t ∥ label) per step
  - 2-layer LSTM, hidden=64, dropout=0.2
  - Last hidden state → Linear(10)
- Verify: output shapes, gradient flow, parameter counts

---

## Phase 4: Training Implementation (`train.py`)
- Implement `train_epoch()` — single training pass
- Implement `evaluate()` — validation pass, no gradients
- Implement `train_model()` — full training loop with history
- Implement `compare_models()` — train all models and collect results
- Implement `count_parameters()` — utility for model size reporting

---

## Phase 5: Entry Point (`main.py`)
- Load dataset and dataloaders
- Instantiate all three models
- Print parameter counts
- Run `compare_models()` for all three
- Print final MSE comparison table

---

## Phase 6: Unit Tests (`test_hw1.py`)
- `TestGenerateSine` — 7 tests on signal generation
- `TestOneHot` — 4 tests on encoding
- `TestExtractWindows` — 3 tests on windowing
- `TestSineDataset` — 7 tests on dataset integrity
- `TestGetDataloaders` — 2 tests on data loading
- `TestMLP` — 3 tests on model shape and gradients
- `TestRNNModel` — 2 tests
- `TestLSTMModel` — 3 tests
- `TestTraining` — 6 parametrised tests on convergence
- Total: 37 tests

---

## Phase 7: Documentation
- Write `README.md` as full lab report
  - Signal model explanation
  - Dataset description
  - Architecture rationale for each model
  - Training configuration and choices
  - Results table with actual MSE values
  - Analysis of why MLP outperformed RNN/LSTM on short windows
  - GitHub link
- Write `prd.md`, `plan.md`, `todo.md`

---

## Phase 8: Submission
- Push all files to GitHub with meaningful commit history
- Fill out submission Word document
- Export as PDF with filename format: `xxxxxxxx-exyy.pdf`
- Email to rmisegal@gmail.com
- Submit individually on the course model

---

## Key Design Decisions
| Decision | Choice | Reason |
|----------|--------|--------|
| Frequencies | 1, 5, 10, 20 Hz | Well-separated, covers low/mid/high range |
| Sample rate | 200 Hz | Safe Nyquist margin (min 40 Hz needed) |
| Noise level | 10% | Perceptible but recoverable |
| Window size | 10 samples | As specified in homework |
| MLP hidden | 64-128-64 | Balanced capacity, funnel shape |
| RNN hidden | 64 | Matches MLP for fair comparison |
| LSTM layers | 2 | Adds depth with gradient-safe gating |
| Optimiser | Adam lr=0.001 | Standard, no tuning needed |
| Epochs | 50 | Sufficient for convergence |