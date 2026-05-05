# PROMPT_LOG — AI Prompt Engineering Log
## HW1: Signal Frequency Extraction with MLP, RNN, and LSTM

This document logs significant prompts used during development with Claude AI,
including context, outputs, and lessons learned.

---

## Prompt 1 — Initial Project Setup
**Context:** Starting HW1, needed full project structure  
**Prompt:** "Help me implement the homework: build MLP, RNN, LSTM for signal
frequency extraction. Frequencies 1, 5, 10, 20 Hz. Context window 10 samples.
MSE loss. Unit tests required. 150 line limit per file."  
**Output:** Initial dataset.py, models.py, train.py, main.py, test_hw1.py  
**Lesson:** First version used `noisy_window` instead of combined signal —
needed correction after re-reading homework spec.

---

## Prompt 2 — Task Correction
**Context:** Realised the task was frequency extraction from a COMBINED signal,
not just denoising a single frequency  
**Prompt:** "The homework says to create a combined signal built from sines and
cosines and extract one frequency. Fix the dataset to generate combined signals."  
**Output:** Added `generate_combined()`, changed input to `mixed_window`  
**Lesson:** Always re-read the full homework spec before implementing.

---

## Prompt 3 — VS Code Configuration
**Context:** Needed project to work well in VS Code  
**Prompt:** "Make it suitable for VS Code"  
**Output:** .vscode/settings.json, launch.json, extensions.json, pytest.ini  
**Lesson:** VS Code debug configurations (launch.json) save significant time.

---

## Prompt 4 — File Size Compliance
**Context:** Guidelines require max 150 code lines per file  
**Prompt:** "Follow the guidelines — each code file should be less than 150 lines"  
**Output:** Split dataset.py → constants.py + signals.py + dataset.py,
split plot.py → plot_losses.py + plot_signals.py, split test_hw1.py → 2 files  
**Lesson:** Design for modularity from the start to avoid splitting later.

---

## Prompt 5 — Full Restructure
**Context:** Full professional guidelines required src/ package structure  
**Prompt:** "Take the best option and walk me through it" (after reviewing
software_submission_guidelines-V3.pdf)  
**Output:** Full src/hw1/ package, tests/unit/, docs/, config/, pyproject.toml,
.env-example, per-algorithm PRDs, PROMPT_LOG  
**Lesson:** Professional Python packaging (src layout) requires conftest.py
to configure pytest paths correctly.

---

## Best Practices Identified
1. Always verify the task interpretation before implementing
2. Use `constants.py` as single source of truth from the start
3. Design files to be under 150 code lines initially — easier than refactoring
4. Broadcast labels to every RNN/LSTM timestep for better conditioning
5. Use `flush=True` on all print statements in Windows environments
6. Use `uv pip install` not `pip install` to ensure correct venv