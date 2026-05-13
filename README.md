# Deep Neural Network for Predicting Hydrodynamic Forces on Submerged Objects

> A PyTorch surrogate model that predicts drag, lift, and torque on parameterized submerged objects in milliseconds — trained on 1 M+ synthetic samples. A **pre-trained model checkpoint** (`best_model.pth`) is included in this repository and can be loaded directly to calculate forces without retraining.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-latest-red)](https://pytorch.org/)

---

## What Is It?

In marine engineering and naval architecture, accurately predicting hydrodynamic forces — drag, lift, and torque — on submerged objects such as ship hulls, underwater vehicles, and offshore structures is critical for design optimisation, performance evaluation, and real-time control. Traditional methods rely on Computational Fluid Dynamics (CFD) simulations that are computationally intensive and time-consuming, often requiring hours or days for a single high-fidelity run. Experimental data from water tunnels is expensive and limited in scale.

This project trains a **deep multilayer perceptron (MLP)** as a surrogate model to approximate these forces in microseconds at inference time. The model is trained on a synthetically generated dataset of at least **1 million samples** covering a wide parameter space of ellipsoidal hull geometries and flow conditions, giving it strong generalisation across the design space.

### Key Features

- **Pre-trained checkpoint** — `best_model.pth` is committed to this repo and can be loaded directly via `reload_model.py` to predict forces without any training step
- **1 M+ sample dataset** — synthetic data covers drag, lift, and torque across a broad range of Reynolds numbers, aspect ratios, and flow velocities
- **Deep MLP architecture** — 7-layer network (3 → 128 → 256 → 512 → 512 → 256 → 128 → 3) with ReLU activations
- **GPU-accelerated training** — CUDA support with `torch.compile` (Inductor backend) and multi-worker DataLoaders
- **Early stopping** — patience-based halt to prevent overfitting; best checkpoint is auto-saved
- **80 / 10 / 10 split** — reproducible train / validation / test split seeded at 42
- **Test suite** — pytest tests cover model shape, normalisation, data masking, and DataLoader behaviour

---

## Install

```bash
# Clone the repository
git clone https://github.com/kgupta9191/Deep-neural-network-for-predicting-hydrodynamic-forces-on-submerged-objects.git
cd Deep-neural-network-for-predicting-hydrodynamic-forces-on-submerged-objects

# Install Python dependencies
pip install -r requirements.txt
```

**Requirements:** `torch`, `numpy`, `pandas`, `pytest`

---

## Quickstart — Use the Pre-Trained Model

A retrained model checkpoint (`best_model.pth`) is already included in this repository. You can load it directly to calculate hydrodynamic forces **without running any training**:

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3)
        )
    def forward(self, x):
        return self.net(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLP().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

# Pass a normalised [aspect_ratio, velocity, reynolds_number] tensor
x = torch.tensor([[0.5, 1.2, 300.0]], dtype=torch.float32).to(device)
drag, lift, torque = model(x).squeeze().tolist()
print(f"Drag: {drag:.4f}  Lift: {lift:.4f}  Torque: {torque:.4f}")
```

Or simply run the provided script:

```bash
python src/reload_model.py
```

---

## Training From Scratch

To retrain the model on your own dataset:

1. Place `hydrodynamic_forces.csv` in the working directory (columns: `aspect_ratio`, `velocity`, `reynolds_number`, `drag`, `lift`, `torque`).
2. Run the training script:

```bash
python src/model_weights.py
```

The script will:
- Filter samples where drag ≤ 1000 and lift ≤ 100
- Normalise features and targets (zero-mean, unit-variance)
- Train for up to 5 000 epochs with early stopping (patience = 500)
- Save the best checkpoint to `best_model.pth` whenever validation loss improves

---

## Repository Structure

```
.
├── best_model.pth          # Pre-trained model checkpoint (ready to use)
├── requirements.txt        # Python dependencies
├── src/
│   ├── model_weights.py    # Data loading, training loop, checkpoint saving
│   └── reload_model.py     # Load best_model.pth and run inference
├── test/
│   └── test.py             # pytest test suite
├── Report/                 # Project report and documentation
└── script.sh               # Helper shell script
```

---

## Model Architecture

| Layer | Input → Output | Activation |
|-------|---------------|------------|
| 1 | 3 → 128 | ReLU |
| 2 | 128 → 256 | ReLU |
| 3 | 256 → 512 | ReLU |
| 4 | 512 → 512 | ReLU |
| 5 | 512 → 256 | ReLU |
| 6 | 256 → 128 | ReLU |
| 7 | 128 → 3 | — |

**Inputs (3):** normalised aspect ratio, flow velocity, Reynolds number  
**Outputs (3):** normalised drag force, lift force, torque

Loss: MSE · Optimiser: Adam (lr = 1e-5) · Batch size: 512

---

## Running Tests

```bash
pytest
```

The test suite (`test/test.py`) covers:
- Model instantiation and forward-pass output shapes
- Batch size invariance and finite-output checks
- Wrong-input-dimension error handling
- Normalisation (zero-mean, unit-variance, constant-column safety)
- Data masking logic
- DataLoader batching and train / val / test split sizes

---

## Background & Objective

Traditional CFD methods require hours-to-days per simulation. This project develops a DNN surrogate trained on a synthetic dataset of ≥ 1 million samples to predict hydrodynamic forces on parameterised ellipsoidal hull shapes under varying flow conditions. The goal is to enable rapid force estimation suitable for iterative design loops and real-time control without sacrificing accuracy across a wide parameter space.

---

## Contributing

Pull requests are welcome. Please open an issue first for significant changes.

---

## License

[MIT](LICENSE)
