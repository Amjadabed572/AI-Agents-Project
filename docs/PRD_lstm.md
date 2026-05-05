# PRD — LSTM Algorithm
## Long Short-Term Memory Network for Frequency Extraction

---

## 1. Algorithm Description
The LSTM processes the 10-sample window sequentially with gated memory cells.
Two layers allow layer 1 to capture sample-to-sample transitions and layer 2
to model the global waveform shape. The label is broadcast to every timestep.
Dropout between layers regularises the deeper network.

## 2. Architecture
```
Per-step input: [sample_t (1) || label (4)] = 5
-> LSTM(hidden=64, layers=2, dropout=0.2)
-> last hidden state (64)
-> Linear(10)
```

## 3. Input / Output Specification
| | Shape | Description |
|--|-------|-------------|
| Input (window) | (B, 10) | Mixed signal samples |
| Input (label) | (B, 4) | 1-hot frequency label |
| Sequence input | (B, 10, 5) | Per-step: sample + label |
| Hidden state h0 | (2, B, 64) | Initial hidden (zero) |
| Cell state c0 | (2, B, 64) | Initial cell (zero) |
| Output | (B, 10) | Predicted clean window |

## 4. Design Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden size | 64 | Matches RNN for fair comparison |
| Layers | 2 | Layer 1: local transitions; Layer 2: global shape |
| Dropout | 0.2 | Light regularisation between layers |
| Activation | sigmoid (gates) + tanh (cell) | Standard LSTM gating |
| Trainable params | 52,106 | Largest; 4 gates per cell vs 1 for RNN |

## 5. Gate Mechanisms
- **Forget gate:** decides what to remove from cell state
- **Input gate:** decides what new information to store
- **Output gate:** decides what to output from cell state
- These gates allow LSTM to retain periodic structure across all 10 steps

## 6. Success Criteria
- Output shape (B, 10) for any batch size B ✅
- Gradients flow to all parameters ✅
- More parameters than RNN ✅
- Loss decreases over training ✅
- Validation MSE documented ✅ (achieved: 0.0202)

## 7. Test Cases
- `test_output_shape`: verify (B, 10) output
- `test_grad_flows`: verify all parameter gradients non-null
- `test_lstm_more_params_than_rnn`: verify LSTM > RNN params
- `test_loss_decreases[LSTMModel]`: verify convergence over 5 epochs