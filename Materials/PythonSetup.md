# Python Setup

## General Notes

Here's the quick picture of how things fit together:

- **Python interpreter** — runs your `.py` files. You install it once per machine.
- **pip** — Python's package manager. You use it to install libraries like NumPy, Matplotlib, etc. (`pip install numpy`).
- **PyTorch** — the ML framework we'll use. It needs a version that matches your system (CPU-only vs. CUDA GPU).

**CUDA** is NVIDIA's GPU compute platform. If you have an NVIDIA GPU, PyTorch can use it to run training *much* faster — but only if the PyTorch build matches your installed CUDA version. If you have no NVIDIA GPU (or you're on a Mac), you'll use CPU-only PyTorch, which is fine for learning.

---

## Python Installation

1. Go to **[python.org/downloads](https://www.python.org/downloads/)** — it auto-detects your OS and suggests the latest stable release (3.12.x as of mid-2025). Download and run the installer.

**Windows:**
- ✅ Check **"Add Python to PATH"** during install — easy to miss, causes headaches if skipped.
- Use the default install location.
- Verify in a terminal: `python --version`

**Mac:**
- The system Python (`/usr/bin/python3`) is outdated — install fresh from python.org.
- After install, use `python3` and `pip3` commands (not `python`/`pip`).
- Verify: `python3 --version`

**Install useful ML packages** (after Python is set up):
```bash
pip install numpy matplotlib pandas scikit-learn jupyter
# (Mac: use pip3)
```

---

## Special Notes on CUDA

**Only relevant if you have an NVIDIA GPU.** Skip this section if you're on a Mac or have no discrete NVIDIA GPU.

**Step 1 — Find your CUDA version:**
- Open a terminal and run: `nvidia-smi`, which ships with your NVIDIA display driver — no extra installs needed.
- The top-right of the output shows something like `CUDA Version: 12.4` — note that number.
- If `nvidia-smi` isn't found, your GPU driver needs installing/updating:
  get it from [nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/).

**Step 2 — Pick the matching PyTorch build:**
- When you get to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/), select the CUDA version that is **≤ your installed version** (e.g., if you have CUDA 12.4, pick the CUDA 12.4 or 12.1 PyTorch build).


> **Mac note:** Macs use Apple Silicon (M1/M2/M3) or Intel — neither supports CUDA. PyTorch has Apple MPS (Metal) support for Apple Silicon, which gives some GPU acceleration. The pytorch.org configurator handles this automatically.

---

## PyTorch Installation

Go to **[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/)** and use the configurator:

| Option | Your pick |
|---|---|
| PyTorch Build | Stable |
| Your OS | Windows / Mac |
| Package | Pip |
| Language | Python |
| Compute Platform | CUDA 12.x *or* CPU *or* MPS (Mac) |

Copy and run the generated command. It looks something like:
```bash
# CPU-only example:
pip install torch torchvision torchaudio

# CUDA 12.4 example:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

This may take a few minutes — the CUDA build is ~2 GB.

---

## Verification

Run these scripts to confirm everything works.

**1. Python basics**
```python
import sys
print(f"Python {sys.version}")

import numpy as np
a = np.array([1, 2, 3])
print("NumPy:", a * 2)
```

**2. PyTorch basics**
```python
import torch
print(f"PyTorch version: {torch.__version__}")

x = torch.tensor([1.0, 2.0, 3.0])
print("Tensor:", x)
print("Mean:", x.mean())
```

**3. CUDA check (skip if CPU-only)**
```python
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.tensor([1.0, 2.0]).cuda()
    print("Tensor on GPU:", x)
```

If `cuda.is_available()` returns `False` when you expect `True`, double-check that your PyTorch CUDA build matches `nvidia-smi`'s CUDA version.

---

## Useful Links

**PyTorch**
- [pytorch.org/tutorials/beginner/basics/intro.html](https://docs.pytorch.org/tutorials/beginner/basics/intro.html) — Official "Learn the Basics" series. Covers tensors, datasets, model building, training loops. Start here once you're set up.
- [pytorch.org/docs](https://pytorch.org/docs/stable/index.html) — Full API reference. Useful when you want to look up a specific function.

**Python**
- [docs.python.org/3/tutorial](https://docs.python.org/3/tutorial/index.html) — Official Python tutorial. Good reference for language fundamentals (lists, dicts, classes, etc.) if you need a refresher.

**Learning ML concepts**
- [fast.ai](https://www.fast.ai/) — Practical deep learning course, free, uses PyTorch. Very hands-on and beginner-friendly.
- [d2l.ai](https://d2l.ai/) — "Dive into Deep Learning" — a free interactive textbook with PyTorch code for every concept. More thorough than fast.ai.

**Debugging / community**
- First resort for debugging: paste your error message into Claude, ChatGPT, or similar — describe your OS and GPU too. Usually faster than searching forums. 
- [PyTorch Forums](https://discuss.pytorch.org/) — Official community forum, good for less common questions.