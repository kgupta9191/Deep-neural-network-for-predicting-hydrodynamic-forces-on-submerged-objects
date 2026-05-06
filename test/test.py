import torch
import torch.nn as nn
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Replicate the core components from src/model_weights.py so they can be
# tested without executing the full training script (which requires a CSV).
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        return self.net(x)


def normalize(train: torch.Tensor) -> torch.Tensor:
    mean = train.mean(dim=0, keepdim=True)
    std = train.std(dim=0, keepdim=True)
    return (train - mean) / (std + 1e-8)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMLP:
    def test_instantiation(self):
        model = MLP()
        assert isinstance(model, nn.Module)

    def test_forward_output_shape(self):
        model = MLP()
        x = torch.randn(16, 3)
        out = model(x)
        assert out.shape == (16, 3), f"Expected (16, 3), got {out.shape}"

    def test_forward_single_sample(self):
        model = MLP()
        x = torch.randn(1, 3)
        out = model(x)
        assert out.shape == (1, 3)

    def test_forward_batch_sizes(self):
        model = MLP()
        for batch in [1, 32, 128, 512]:
            out = model(torch.randn(batch, 3))
            assert out.shape == (batch, 3)

    def test_parameters_exist(self):
        model = MLP()
        params = list(model.parameters())
        assert len(params) > 0

    def test_output_is_finite(self):
        model = MLP()
        x = torch.randn(8, 3)
        out = model(x)
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf"

    def test_wrong_input_raises(self):
        model = MLP()
        with pytest.raises(RuntimeError):
            model(torch.randn(4, 5))  # wrong input dimension


class TestNormalize:
    def test_zero_mean(self):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed = normalize(data)
        assert torch.allclose(normed.mean(dim=0), torch.zeros(2), atol=1e-5)

    def test_unit_variance(self):
        data = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        normed = normalize(data)
        # std ≈ 1 when there are multiple distinct values
        assert torch.allclose(normed.std(dim=0), torch.ones(2), atol=1e-4)

    def test_constant_column_no_nan(self):
        data = torch.tensor([[5.0], [5.0], [5.0]])
        normed = normalize(data)
        assert torch.isfinite(normed).all(), "Constant column produced NaN/Inf"

    def test_output_shape_preserved(self):
        data = torch.randn(50, 3)
        normed = normalize(data)
        assert normed.shape == data.shape


class TestDataMask:
    def _make_data(self):
        # Simulate the last 3 columns: [col-3, col-2, col-1]
        rows = [
            [1.0, 2.0, 3.0, 500.0, 50.0, 0.1],   # passes mask
            [1.0, 2.0, 3.0, 1500.0, 50.0, 0.1],  # fails col-3 <= 1000
            [1.0, 2.0, 3.0, 500.0, 150.0, 0.1],  # fails col-2 <= 100
            [1.0, 2.0, 3.0, 1000.0, 100.0, 0.1], # passes (boundary)
        ]
        return torch.tensor(rows, dtype=torch.float32)

    def test_mask_filters_correctly(self):
        data = self._make_data()
        mask = (data[:, -3] <= 1000) & (data[:, -2] <= 100)
        filtered = data[mask]
        assert filtered.shape[0] == 2  # rows 0 and 3

    def test_features_and_targets_split(self):
        data = self._make_data()
        mask = (data[:, -3] <= 1000) & (data[:, -2] <= 100)
        filtered = data[mask]
        X = filtered[:, :-3]
        y = filtered[:, -3:]
        assert X.shape[1] == 3
        assert y.shape[1] == 3
        assert X.shape[0] == y.shape[0]


class TestDataLoader:
    def test_dataloader_batches(self):
        from torch.utils.data import TensorDataset, DataLoader

        X = torch.randn(100, 3)
        y = torch.randn(100, 3)
        ds = TensorDataset(X, y)
        loader = DataLoader(ds, batch_size=32)
        batches = list(loader)
        assert len(batches) == 4  # ceil(100/32) = 4
        xb, yb = batches[0]
        assert xb.shape == (32, 3)
        assert yb.shape == (32, 3)

    def test_train_val_test_split_sizes(self):
        from torch.utils.data import TensorDataset, DataLoader, random_split

        N = 1000
        X = torch.randn(N, 3)
        y = torch.randn(N, 3)
        dataset = TensorDataset(X, y)
        n_train = int(0.8 * N)
        n_val = int(0.1 * N)
        n_test = N - n_train - n_val
        train_ds, val_ds, test_ds = random_split(
            dataset,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
        assert len(train_ds) == 800
        assert len(val_ds) == 100
        assert len(test_ds) == 100
