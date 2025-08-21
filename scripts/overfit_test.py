#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from src.model import BurnoutMLP
import yaml

def load_config(path="configs/config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def main():
    # Load config
    config = load_config()

    # Load first client's data with weights_only=False to allow arbitrary tensors
    data_path = os.path.join("data", "processed", "client_1.pt")
    try:
        X, y = torch.load(data_path, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        X, y = torch.load(data_path)

    # Take a small subset (first 50 samples)
    X_small = torch.tensor(X[:50], dtype=torch.float32)
    y_small = torch.tensor(y[:50], dtype=torch.long)

    # DataLoader
    batch_size = config.get("batch_size", 10)
    dataset = TensorDataset(X_small, y_small)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model, loss, optimizer
    model = BurnoutMLP(
        input_dim=X_small.shape[1],
        hidden_dims=tuple(config["hidden_dims"]),
        output_dim=3,
        p=config["dropout"],
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"] * 10)  # higher LR for overfit

    # Train for 100 epochs
    num_epochs = 100
    train_losses = []
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        train_losses.append(epoch_loss / len(dataset))
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} — Loss: {train_losses[-1]:.4f}")

    # Final training accuracy
    model.eval()
    with torch.no_grad():
        preds = model(X_small).argmax(dim=1)
        accuracy = (preds == y_small).float().mean().item()

    print(f"\nFinal training loss: {train_losses[-1]:.4f}")
    print(f"Training accuracy: {accuracy * 100:.2f}%")

    # Plot loss curve
    os.makedirs("results/metrics", exist_ok=True)
    plt.figure()
    plt.plot(range(1, num_epochs + 1), train_losses)
    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")
    plt.title("Overfitting Test Loss Curve")
    plt.tight_layout()
    plt.savefig("results/metrics/overfit_loss_curve.png")
    print("Saved loss curve to results/metrics/overfit_loss_curve.png")

if __name__ == "__main__":
    main()