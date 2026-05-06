# Deep neural network for predicting hydrodynamic forces on submerged objects

## Overview
This repository provides a deep neural network (DNN) surrogate model for predicting hydrodynamic forces on submerged objects. The goal is to replace expensive CFD simulations with a fast, data-driven model that can be used for design iteration, optimization, and real-time analysis.

## Project goals
- Predict hydrodynamic forces (e.g., drag/lift/torque components) from a compact set of input parameters.
- Train on large, synthetic datasets to capture nonlinear flow effects.
- Enable efficient inference using a pre-trained model.

## Repository structure
- `src/model_weights.py` — training script that builds the model, trains it, and saves the best weights.
- `src/reload_model.py` — loading script that restores trained weights for inference.
- `best_model.pth` — pre-trained model weights saved from training.
- `requirements.txt` — Python dependencies.
- `Report/report.pdf` — project report and extended documentation.
- `test/` — unit tests covering the model architecture and utilities.

## Data format
The training script expects a CSV named `hydrodynamic_forces.csv` in the repository root.
The current pipeline assumes:
- **First 3 columns**: input features (geometry/flow parameters).
- **Last 3 columns**: target force components.

The script filters samples where the last two target columns exceed thresholds (`<= 1000` and `<= 100`) before training.

## Model architecture
The network is a multilayer perceptron (MLP) with the following structure:
```
3 → 128 → 256 → 512 → 512 → 256 → 128 → 3
```
- Activation: ReLU between layers
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam (lr = 1e-5)
- Train/val/test split: 80/10/10
- Early stopping based on validation loss (checked every 100 epochs)

## Setup
```bash
pip install -r requirements.txt
```
> The training script uses `torch.compile`, which requires PyTorch 2.x.

## Training
Run the training script from the repo root (expects `hydrodynamic_forces.csv`):
```bash
python src/model_weights.py
```
The script saves the best model weights to `best_model.pth`.

## Pre-trained model (`best_model.pth`)
`best_model.pth` contains trained weights and can be used directly for inference. To use it:
1. Instantiate the same `MLP` architecture (see `src/model_weights.py`).
2. Load the weights from `best_model.pth`.
3. Normalize inputs in the same way as training (mean/std computed from the training data).
4. Run the model to predict hydrodynamic forces.

The provided `src/reload_model.py` is the starting point for loading weights; ensure the `MLP` class definition is available when running it.

## Tests
```bash
pytest
```

## License
See `LICENSE`.
