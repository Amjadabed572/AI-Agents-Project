# PRD — MLP Algorithm
## Fully Connected Multi-Layer Perceptron for Frequency Extraction

---

## 1. Algorithm Description
The MLP treats the problem as a flat regression: the 10-sample mixed window and
4-dimensional 1-hot label are concatenated into a 14-dimensional input vector.
Three hidden layers with ReLU activations transform this into a 10-sample
predicted clean window.

## 2. Architecture
```
Input(14) -> Linear(64) -> ReLU -> Linear(128) -> ReLU -> Linear(64) -> ReLU -> Linear(10)
```

## 3. Input / Output Specification
| | Shape | Description |
|--|-------|-------------|
| Input | (B, 14) | Concatenated [mixed_window (10) + label (4)] |
| Output | (B, 10) | Predicted clean window of target frequency |

## 4. Design Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Hidden sizes | [64, 128, 64] | Funnel-expand-funnel: avoids bottleneck |
| Activation | ReLU | Standard for regression, avoids vanishing gradients |
| Trainable params | 18,186 | Sufficient capacity without overfitting |

## 5. Limitations
- No sequential awareness — treats each sample position independently
- Cannot model temporal dependencies within the window
- Performance degrades at low frequencies (1 Hz) where global wave shape matters

## 6. Success Criteria
- Output shape (B, 10) for any batch size B ✅
- Gradients flow to all parameters ✅
- Validation MSE < 0.05 ✅ (achieved: 0.0094)
- Loss decreases over training ✅

## 7. Test Cases
- `test_output_shape`: verify (B, 10) output
- `test_parameter_count_positive`: verify trainable params > 0
- `test_grad_flows`: verify all parameter gradients non-null
- `test_loss_decreases[MLP]`: verify convergence over 5 epochs