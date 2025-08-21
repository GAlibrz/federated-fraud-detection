
# client.py

import os
import logging
import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from src.model import create_model, load_config
from src.data import load_full_dataset_for
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

class FocalLoss(nn.Module):
    """
    Focal Loss for binary classification.
    FL(p_t) = -α * (1 - p_t)^γ * log(p_t)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_factor * focal_factor * bce
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

class FLClient(fl.client.NumPyClient):
    def __init__(self, cid: str, config_path: str = None):
        # Load YAML config
        self.config = load_config(config_path)
        self.client_id = cid

        # Log client and skew
        skew = self.config.get("skew_type", "uniform")
        logging.info(f"Client {cid}: using data skew_type='{skew}'")

        # Load full dataset for this client
        full_ds = load_full_dataset_for(cid, skew_type=skew)
        logging.info(f"Client {cid}: full dataset size = {len(full_ds)} samples")

        # Split into train/val/test
        test_frac = float(self.config.get("test_frac", 0.1))
        val_frac = float(self.config.get("val_frac", 0.1))
        n = len(full_ds)
        n_test = int(test_frac * n)
        n_val = int(val_frac * n)
        n_train = n - n_val - n_test
        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )
        logging.info(
            f"Client {cid}: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
        )

        # Weighted sampler for train loader
        labels = torch.tensor([label for _, label in train_ds], dtype=torch.long)
        class_counts = torch.bincount(labels)
        weights_per_class = 1.0 / class_counts.float()
        sample_weights = weights_per_class[labels]
        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
        )

        batch_size = int(self.config.get("batch_size", 64))
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, sampler=sampler, drop_last=True
        )
        self.test_loader = DataLoader(test_ds, batch_size=batch_size)

        # Training settings
        self.local_epochs = int(self.config.get("local_epochs", 5))

        # Initialize model
        example_x, _ = train_ds[0]
        input_dim = example_x.shape[0]
        self.model = create_model(input_dim, config_path)

        # FocalLoss or other
        alpha = float(self.config.get("focal_alpha", 0.25))
        gamma = float(self.config.get("focal_gamma", 2.0))
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")

        # Optimizer & scheduler
        lr = float(self.config.get("lr", 5e-4))
        weight_decay = float(self.config.get("weight_decay", 1e-5))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        sched_cfg = self.config.get("scheduler", {})
        step_size = int(sched_cfg.get("step_size", 5))
        sched_gamma = float(sched_cfg.get("gamma", 0.5))
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=sched_gamma
        )

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        eps = float(self.config.get("label_smoothing", 0.0))
        for _ in range(self.local_epochs):
            for X, y in self.train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X)
                y_smooth = y.float() * (1 - eps) if eps > 0 else y.float()
                loss = self.criterion(logits, y_smooth)
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
        return self.get_parameters(config), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        all_probs, all_preds, all_true = [], [], []
        loss_sum, total = 0.0, 0
        thresh = float(self.config.get("threshold", 0.5))
        with torch.no_grad():
            for X, y in self.test_loader:
                logits = self.model(X)
                loss = self.criterion(logits, y.float())
                loss_sum += loss.item() * X.size(0)
                probs = torch.sigmoid(logits)
                preds = (probs > thresh).long()
                all_probs.append(probs.cpu())
                all_preds.append(preds.cpu())
                all_true.append(y.cpu())
                total += y.size(0)
        all_probs = torch.cat(all_probs).numpy()
        all_preds = torch.cat(all_preds).numpy()
        all_true = torch.cat(all_true).numpy()
        accuracy = (all_preds == all_true).mean()
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_true, all_preds, average="binary", zero_division=0
        )
        auc = roc_auc_score(all_true, all_probs)
        avg_loss = loss_sum / total
        metrics = {"accuracy": accuracy, "precision": precision,
                   "recall": recall, "f1": f1, "roc_auc": auc}
        return float(avg_loss), total, metrics

if __name__ == "__main__":
    cid = os.environ.get("CLIENT_ID", "0")
    logging.info(f"Starting FLClient for CID={cid}")
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FLClient(cid, config_path="configs/config.yaml"),
    )
