# PRD — RNN Algorithm
## Elman Recurrent Neural Network for Frequency Extraction

---

## 1. Algorithm Description
The RNN processes the 10-sample window as a time series, one sample per step.
The 1-hot label is broadcast to every timestep so the hidden state is always
conditioned on the target frequency. The final hidden state is projected to the
10-sample predicted clean window.

## 2. Architecture
```
Per-step input: [sample_t (1) || label (4)] = 5
-> RNN(hidden=64, layers=1, tanh)
-> last hidden state (64)
-> Linear(10)
```

## 3. Input / Output Specification
| | Shape | Description |
|--|-------|-------------|
| Input (window) | (B, 10) | Mixed signal samples |
| Input (label) | (B, 4) | 1-hot frequency label |
| Sequence input | (B, 10, 5) | Per-step: sample + label |
| Output | (B, 10) | Predicted clean window |

## 4. Design Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden size | 64 | Matches MLP for fair comparison |
| Layers | 1 | Keeps parameter count low |
| Activation | tanh | Bounded output suits sine values in [-1, 1] |
| Trainable params | 5,194 | Smallest of three models |

## 5. Limitations
- Vanishing gradients on longer sequences (not an issue for 10 steps)
- Slower convergence than MLP on this task
- Struggles to separate mid-range frequencies from combined signal

## 6. Success Criteria
- Output shape (B, 10) for any batch size B ✅
- Gradients flow to all parameters ✅
- Loss decreases over training ✅
- Validation MSE documented ✅ (achieved: 0.0744)

## 7. Test Cases
- `test_output_shape`: verify (B, 10) output
- `test_grad_flows`: verify all parameter gradients non-null
- `test_loss_decreases[RNNModel]`: verify convergence over 5 epochs