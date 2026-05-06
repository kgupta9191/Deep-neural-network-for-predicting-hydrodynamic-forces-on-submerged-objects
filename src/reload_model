import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim

model = MLP()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model = torch.compile(model, backend="inductor")  # or backend="nvfuser" on GPU

# Reloading the old saved model
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()
