# Load modules
import torch
print("CUDA version:", torch.version.cuda)
print("cuDNN version:", torch.backends.cudnn.version())
print("GPU available?", torch.cuda.is_available())
print("Device count:", torch.cuda.device_count())
print("Device name:", torch.cuda.get_device_name(0))
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data import random_split
import numpy as np

# Load data
data = torch.tensor(
    pd.read_csv("hydrodynamic_forces.csv").values,
    dtype=torch.float32
)

# Create boolean mask
mask = (
    (data[:, -3] <= 1000) &
    (data[:, -2] <= 100)
)

# Apply mask
filtered_data = data[mask]
X = filtered_data[:, :-3]
#y = data[:, -3:].unsqueeze(1)
y = filtered_data[:, -3:]

# Normalize data
def normalize(train):
    mean = train.mean(dim=0, keepdim=True)
    std  = train.std(dim=0, keepdim=True)
    train = (train - mean) / (std + 1e-8)
    return train

X_norm = normalize(X)
y_norm = normalize(y)
dataset = TensorDataset(X_norm, y_norm)

# Split data
N = len(dataset)
n_train = int(0.8 * N)
n_val   = int(0.1 * N)
n_test  = N - n_train - n_val
train_ds, val_ds, test_ds = random_split(
    dataset,
    [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)  # reproducible
)

# Neural Network Architecture
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
            nn.Linear(128, 3)
        )

    def forward(self, x):
        return self.net(x)

# Define model, Loss function and Optimizer
model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# Implement GPU for better performance
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = torch.compile(model, backend="inductor")  # or backend="nvfuser" on GPU

# Include workers to parallelize the process and create batches
train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size=512)

# Main Body of the code
epochs = 5000
patience = 500
best_val = float('inf')
wait = 0
best_model_state = None

for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        yb = yb.squeeze(1)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

    if epoch % 100 == 0:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                yb = yb.squeeze(1)
                val_loss = criterion(model(xb), yb).item()

        print(f"Epoch {epoch}: Train Loss = {loss:.4e} Val Loss = {val_loss:.4e}")

        if val_loss < best_val:
            best_val = val_loss
            best_model_state = model.state_dict()
            torch.save(model.state_dict(), "best_model.pth")
            wait = 0
        else:
            wait += 100

        if wait >= patience:
            print("Early stopping triggered")
            break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

