# PRD — Product Requirements Document
## HW1: Signal Frequency Extraction with MLP, RNN, and LSTM

---

## 1. Project Overview
Build and compare three neural network architectures on a frequency extraction task:
given a combined noisy sine wave window and a 1-hot frequency label, predict the
clean version of the target frequency component.

## 2. Problem Statement
A combined signal is a mixture of multiple sine waves at different frequencies with
added noise. The goal is to extract one target frequency component from this mixture,
guided by a 1-hot label identifying the target. This simulates frequency-selective
filtering using learned neural representations.

## 3. Goals
- Implement a combined sine wave dataset generator with controllable noise
- Train and compare MLP, RNN, and LSTM on frequency extraction
- Evaluate using MSE loss on a held-out validation set
- Document findings in a detailed lab report

## 4. KPIs and Acceptance Criteria
| Metric | Target | Achieved |
|--------|--------|----------|
| All unit tests pass | 43/43 | ✅ 44/44 |
| Final Val MSE (MLP) | < 0.05 | ✅ 0.0106 |
| Final Val MSE (RNN) | < 0.20 | ✅ 0.0610 |
| Final Val MSE (LSTM) | < 0.05 | ✅ 0.0071 |
| Code line limit | ≤ 150 per file |
| Test coverage | ≥ 85% |

## 5. Functional Requirements
- Generate sine waves at 4 known frequencies with amplitude and phase noise
- Create combined signals summing all frequency components
- Support 1-hot encoding of target frequency
- Provide sliding window (10 samples) extraction
- Train MLP, RNN, LSTM with identical hyperparameters
- Produce loss curves and signal extraction visualisations

## 6. Non-Functional Requirements
- All Python files ≤ 150 code lines
- Reproducible via fixed random seed
- Virtual environment managed with UV
- No hardcoded values — all constants in constants.py
- Unit tests for all public functions

## 7. Assumptions
- Sample rate 200 Hz satisfies Nyquist for all chosen frequencies
- 10-sample context window is sufficient per homework specification
- 10% noise level is perceptible but recoverable

## 8. Out of Scope
- Real-time signal processing
- Frequencies outside the 4 defined values
- GPU training optimisation

## 9. Timeline
| Phase | Deliverable |
|-------|------------|
| Phase 1 | Environment setup, constants, signals |
| Phase 2 | Dataset, models, training loop |
| Phase 3 | Tests, visualisations, documentation |
| Phase 4 | Submission |

See `docs/PRD_mlp.md`, `docs/PRD_rnn.md`, `docs/PRD_lstm.md` for per-model PRDs.