# TODO — Task Checklist
## HW1: Signal Frequency Extraction with MLP, RNN, and LSTM

---

## ENVIRONMENT SETUP
- [x] Install UV package manager
- [x] Create and activate virtual environment
- [x] Install PyTorch via uv pip install
- [x] Install NumPy via uv pip install
- [x] Install pytest via uv pip install
- [x] Install matplotlib via uv pip install
- [x] Create GitHub repository (hw1-sine-rnn)
- [x] Clone repository locally
- [x] Configure .gitignore
- [x] Configure VS Code settings.json, launch.json, extensions.json
- [x] Create requirements.txt
- [x] Create pytest.ini

---

## CONSTANTS (constants.py)
- [x] Define FREQUENCIES as List[float] = [1.0, 5.0, 10.0, 20.0]
- [x] Define SAMPLE_RATE = 200
- [x] Define WINDOW_LEN = 10
- [x] Define SIGNAL_DURATION = 10.0
- [x] Define AMPLITUDE = 1.0
- [x] Define NOISE_SIGMA = 0.10
- [x] Define NUM_CLASSES = len(FREQUENCIES)

---

## SIGNAL GENERATION (signals.py)
- [x] Implement generate_sine() with amplitude jitter
- [x] Implement generate_sine() with phase noise (σ_2)
- [x] Implement generate_sine() with additive Gaussian noise
- [x] Implement generate_combined() summing all frequencies
- [x] Implement generate_combined() with additive noise on combined signal
- [x] Implement one_hot() helper
- [x] Implement extract_windows() helper
- [x] Verify float32 output dtype throughout

---

## DATASET (dataset.py)
- [x] Implement SineDataset.__init__() with seed
- [x] Implement SineDataset._generate() for all frequency classes
- [x] Return mixed_window (combined signal) in each item
- [x] Return clean_window (target frequency only) in each item
- [x] Return label (1-hot) in each item
- [x] Return freq_idx in each item
- [x] Implement SineDataset.__len__()
- [x] Implement SineDataset.__getitem__()
- [x] Implement get_dataloaders() with 80/20 split
- [x] Verify dataset total length = 2000
- [x] Verify item shapes are correct
- [x] Verify labels sum to 1.0

---

## MODELS (models.py)
- [x] Implement MLP: Linear(64) → ReLU → Linear(128) → ReLU → Linear(64) → Linear(10)
- [x] MLP forward(): concatenate window + label → net
- [x] Implement RNNModel: single layer, hidden=64, tanh
- [x] RNN forward(): broadcast label to every timestep
- [x] Implement LSTMModel: 2 layers, hidden=64, dropout=0.2
- [x] LSTM forward(): broadcast label to every timestep
- [x] Verify MLP output shape (B, 10)
- [x] Verify RNN output shape (B, 10)
- [x] Verify LSTM output shape (B, 10)
- [x] Verify gradients flow through all models
- [x] Verify LSTM has more parameters than RNN

---

## TRAINING (train.py)
- [x] Implement train_epoch() with Adam optimizer
- [x] Implement evaluate() with torch.no_grad()
- [x] Implement train_model() with epoch logging
- [x] Implement compare_models() for all models
- [x] Implement count_parameters() utility

---

## ENTRY POINT (main.py)
- [x] Load dataloaders
- [x] Instantiate MLP, RNNModel, LSTMModel
- [x] Print parameter counts
- [x] Run compare_models()
- [x] Print final results table

---

## VISUALISATION
- [x] Implement plot_loss_curves() in plot_losses.py
- [x] Implement plot_final_comparison() in plot_losses.py
- [x] Implement _predict_window() in plot_signals.py
- [x] Implement plot_signal_extraction() in plot_signals.py
- [x] Implement plot.py entry point
- [x] Generate and save loss_curves.png
- [x] Generate and save signal_extraction.png
- [x] Generate and save model_comparison.png

---

## UNIT TESTS
- [x] test_signals.py: TestGenerateSine (7 tests)
- [x] test_signals.py: TestGenerateCombined (5 tests)
- [x] test_signals.py: TestOneHot (4 tests)
- [x] test_signals.py: TestExtractWindows (3 tests)
- [x] test_signals.py: TestSineDataset (8 tests)
- [x] test_signals.py: TestGetDataloaders (2 tests)
- [x] test_models.py: TestMLP (3 tests)
- [x] test_models.py: TestRNNModel (2 tests)
- [x] test_models.py: TestLSTMModel (3 tests)
- [x] test_models.py: TestTraining (6 parametrised tests)
- [x] All 43 tests pass

---

## RESULTS
- [x] MLP Final Val MSE: 0.0094
- [x] RNN Final Val MSE: 0.0744
- [x] LSTM Final Val MSE: 0.0202
- [x] Per-frequency MSE table documented
- [x] Analysis written in README.md

---

## DOCUMENTATION
- [x] README.md — full lab report with results, graphs, analysis
- [x] docs/PRD.md — product requirements document
- [x] docs/PLAN.md — architecture and implementation plan
- [x] docs/TODO.md — this file

---

## CODE QUALITY
- [x] All files comply with 150 code-line limit
- [x] All functions have docstrings
- [x] All constants centralised in constants.py
- [x] No hardcoded values in logic files

---

## SUBMISSION
- [ ] Push all files to GitHub with meaningful commit history
- [ ] Verify GitHub repo is public
- [ ] Share repo with rmisegal@gmail.com
- [ ] Fill out uoh-rl07-ex01.docx with student details
- [ ] Add GitHub link to submission form
- [ ] Export as PDF: amj-naji-ex01.pdf
- [ ] Email PDF to rmisegal@gmail.com
- [ ] Student 1 submits individually on Moodle
- [ ] Student 2 submits individually on Moodle