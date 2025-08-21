# src/model.py
import os
import yaml
import torch.nn as nn


def load_config(path: str = None) -> dict:
    """
    Load the YAML configuration file for model hyperparameters.
    """
    if path is None:
        # Default to configs/config.yaml in project root
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(project_root, "configs", "config.yaml")
    with open(path, "r") as f:
        return yaml.safe_load(f)


class FraudDetectorMLP(nn.Module):
    """
    Feedforward MLP for binary fraud detection.
    Outputs logits for BCEWithLogitsLoss.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] = (64, 32),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            # First hidden block
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Second hidden block
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.LayerNorm(hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Output layer: single logit
            nn.Linear(hidden_dims[1], 1),
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        # returns logits: [batch_size]
        return self.net(x).squeeze(1)


def create_model(input_dim: int, config_path: str = None) -> FraudDetectorMLP:
    """
    Factory function to create a FraudDetectorMLP using hyperparameters
    from the YAML config.

    Args:
        input_dim: Number of input features.
        config_path: Optional path to config YAML file.
    """
    cfg = load_config(config_path)
    # Model architecture params
    hidden_dims = tuple(cfg.get("hidden_dims", [64, 32]))
    dropout = float(cfg.get("dropout", 0.1))
    return FraudDetectorMLP(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout,
    )