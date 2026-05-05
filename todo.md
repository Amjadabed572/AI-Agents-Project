# TODO List
## HW1: Signal Frequency Extraction with MLP, RNN, and LSTM

---

## ENVIRONMENT SETUP
- [x] Install UV package manager
- [x] Create virtual environment with UV
- [x] Activate virtual environment
- [x] Install PyTorch
- [x] Install NumPy
- [x] Install pytest
- [x] Install matplotlib
- [x] Create GitHub account
- [x] Create GitHub repository (hw1-sine-rnn)
- [x] Clone repository locally
- [x] Set up .gitignore
- [x] Configure VS Code settings.json
- [x] Configure VS Code launch.json
- [x] Configure VS Code extensions.json
- [x] Install recommended VS Code extensions

---

## DATASET (dataset.py)
- [x] Define constants: FREQUENCIES, SAMPLE_RATE, WINDOW_LEN, etc.
- [x] Implement generate_sine() function
- [x] Add amplitude jitter to generate_sine()
- [x] Add additive Gaussian noise to generate_sine()
- [x] Add phase parameter to generate_sine()
- [x] Implement one_hot() helper function
- [x] Implement extract_windows() helper function
- [x] Implement SineDataset class
- [x] Add __len__ to SineDataset
- [x] Add __getitem__ to SineDataset
- [x] Generate 500 windows per frequency in SineDataset
- [x] Randomise phase per generated signal
- [x] Return noisy_window in each dataset item
- [x] Return clean_window in each dataset item
- [x] Return label (1-hot) in each dataset item
- [x] Return freq_idx in each dataset item
- [x] Implement get_dataloaders() factory function
- [x] Add 80/20 train/val split to get_dataloaders()
- [x] Set random seed for reproducibility
- [x] Verify dataset total length = 2000
- [x] Verify item shapes are correct
- [x] Verify labels sum to 1.0

---

## MODELS (models.py)
- [x] Define INPUT_SIZE constant (14)
- [x] Define OUTPUT_SIZE constant (10)
- [x] Implement MLP class
- [x] Add hidden layers 64-128-64 to MLP
- [x] Add ReLU activations to MLP
- [x] Add forward() method to MLP (concatenate window + label)
- [x] Implement RNNModel class
- [x] Add single RNN layer hidden=64 to RNNModel
- [x] Expand label to every timestep in RNNModel
- [x] Add linear projection head to RNNModel
- [x] Add forward() method to RNNModel
- [x] Implement LSTMModel class
- [x] Add 2-layer LSTM hidden=64 to LSTMModel
- [x] Add dropout=0.2 between LSTM layers
- [x] Expand label to every timestep in LSTMModel
- [x] Add linear projection head to LSTMModel
- [x] Add forward() method to LSTMModel
- [x] Verify MLP output shape (B, 10)
- [x] Verify RNN output shape (B, 10)
- [x] Verify LSTM output shape (B, 10)
- [x] Verify gradients flow through all models
- [x] Verify LSTM has more parameters than RNN

---

## TRAINING (train.py)
- [x] Implement train_epoch() function
- [x] Add optimizer.zero_grad() in train_epoch()
- [x] Add loss.backward() in train_epoch()
- [x] Add optimizer.step() in train_epoch()
- [x] Implement evaluate() function
- [x] Add torch.no_grad() context in evaluate()
- [x] Implement train_model() function
- [x] Add epoch logging every 10 epochs in train_model()
- [x] Return loss history from train_model()
- [x] Implement compare_models() function
- [x] Implement count_parameters() utility

---

## ENTRY POINT (main.py)
- [x] Import all modules in main.py
- [x] Set DEVICE (cpu/cuda)
- [x] Set EPOCHS=50, BATCH=64, LR=0.001
- [x] Call get_dataloaders()
- [x] Instantiate MLP, RNNModel, LSTMModel
- [x] Print parameter counts for each model
- [x] Call compare_models()
- [x] Print final results table
- [x] Add flush=True to all print statements

---

## UNIT TESTS (test_hw1.py)
- [x] Write TestGenerateSine::test_output_length
- [x] Write TestGenerateSine::test_output_dtype
- [x] Write TestGenerateSine::test_pure_signal_amplitude
- [x] Write TestGenerateSine::test_noise_increases_variance
- [x] Write TestGenerateSine::test_frequency_content
- [x] Write TestGenerateSine::test_different_frequencies_differ
- [x] Write TestGenerateSine::test_custom_duration
- [x] Write TestOneHot::test_correct_length
- [x] Write TestOneHot::test_single_one
- [x] Write TestOneHot::test_all_indices
- [x] Write TestOneHot::test_dtype
- [x] Write TestExtractWindows::test_window_shape
- [x] Write TestExtractWindows::test_no_overlap
- [x] Write TestExtractWindows::test_partial_window_dropped
- [x] Write TestSineDataset::test_total_length
- [x] Write TestSineDataset::test_item_keys
- [x] Write TestSineDataset::test_window_shapes
- [x] Write TestSineDataset::test_label_shape
- [x] Write TestSineDataset::test_label_is_one_hot
- [x] Write TestSineDataset::test_freq_idx_range
- [x] Write TestSineDataset::test_noisy_differs_from_clean
- [x] Write TestGetDataloaders::test_returns_two_loaders
- [x] Write TestGetDataloaders::test_sizes_sum_to_total
- [x] Write TestMLP::test_output_shape
- [x] Write TestMLP::test_parameter_count_positive
- [x] Write TestMLP::test_grad_flows
- [x] Write TestRNNModel::test_output_shape
- [x] Write TestRNNModel::test_grad_flows
- [x] Write TestLSTMModel::test_output_shape
- [x] Write TestLSTMModel::test_grad_flows
- [x] Write TestLSTMModel::test_lstm_more_params_than_rnn
- [x] Write TestTraining::test_loss_decreases[MLP]
- [x] Write TestTraining::test_loss_decreases[RNNModel]
- [x] Write TestTraining::test_loss_decreases[LSTMModel]
- [x] Write TestTraining::test_evaluate_returns_float[MLP]
- [x] Write TestTraining::test_evaluate_returns_float[RNNModel]
- [x] Write TestTraining::test_evaluate_returns_float[LSTMModel]
- [x] Run all 37 tests and verify they pass

---

## RESULTS & DOCUMENTATION
- [x] Run main.py and collect final MSE values
- [x] MLP Final Val MSE: 0.001842
- [x] RNN Final Val MSE: 0.003279
- [x] LSTM Final Val MSE: 0.001920
- [x] Update README.md with actual results
- [x] Write analysis of why MLP outperformed RNN
- [x] Write analysis of LSTM vs RNN comparison
- [x] Write frequency-specific analysis table
- [x] Add GitHub link to README.md
- [x] Write prd.md
- [x] Write plan.md
- [x] Write todo.md

---

## SUBMISSION
- [ ] Push all files to GitHub
- [ ] Verify GitHub repo is public
- [ ] Verify commit history has multiple commits
- [ ] Fill out uoh-rl07-ex01.docx with student details
- [ ] Add GitHub link to submission form
- [ ] Export submission form as PDF
- [ ] Name PDF correctly: xxxxxxxx-exyy.pdf
- [ ] Email PDF to rmisegal@gmail.com
- [ ] Submit individually on course model
- [ ] Confirm partner has also submitted individually