# evaluate_local.py

import os
import json
import torch
import yaml
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)
from torch.utils.data import random_split, DataLoader, WeightedRandomSampler
from src.model import create_model, load_config
from src.data import load_full_dataset_for
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (1 - pt) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal * bce
        return loss.mean() if self.reduction == "mean" else loss.sum()

def train_local_model(train_ds, config):
    # Build model
    input_dim = train_ds[0][0].shape[0]
    model = create_model(input_dim, config_path="configs/config.yaml")
    device = torch.device("cpu")
    model.to(device)

    # Sampler
    labels = torch.tensor([y for _, y in train_ds], dtype=torch.long)
    wpc = 1.0 / torch.bincount(labels).float()
    sw = wpc[labels]
    sampler = WeightedRandomSampler(sw, num_samples=len(sw), replacement=True)

    loader = DataLoader(train_ds, batch_size=int(config["batch_size"]),
                        sampler=sampler, drop_last=True)

    # Loss, optimizer, scheduler
    alpha = float(config.get("focal_alpha", 0.25))
    gamma = float(config.get("focal_gamma", 2.0))
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    opt = torch.optim.Adam(model.parameters(),
                           lr=float(config["lr"]),
                           weight_decay=float(config["weight_decay"]))
    sched = config.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(sched.get("step_size",5)),
        gamma=float(sched.get("gamma",0.5))
    )

    eps = float(config.get("label_smoothing", 0.0))
    for _ in range(int(config["local_epochs"])):
        model.train()
        for X, y in loader:
            opt.zero_grad()
            logits = model(X)
            y_s = y.float() * (1 - eps) if eps > 0 else y.float()
            loss = criterion(logits, y_s)
            loss.backward()
            opt.step()
        scheduler.step()
    return model

def evaluate_model(model, test_ds, threshold):
    model.eval()
    probs, preds, trues = [], [], []
    loader = DataLoader(test_ds, batch_size=64)
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            p = torch.sigmoid(logits).cpu().numpy()
            pr = (p > threshold).astype(int)
            probs.append(p); preds.append(pr); trues.append(y.numpy())
    probs = np.concatenate(probs)
    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    acc = accuracy_score(trues, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        trues, preds, average="binary", zero_division=0)
    auc = roc_auc_score(trues, probs)
    return {"accuracy": acc, "precision": prec,
            "recall": rec, "f1": f1, "roc_auc": auc}

def main():
    cfg = load_config("configs/config.yaml")
    skew = cfg.get("skew_type", "uniform")
    num_clients = int(cfg["num_clients"])
    threshold   = float(cfg.get("threshold", 0.5))
    test_frac   = float(cfg.get("test_frac", 0.1))
    val_frac    = float(cfg.get("val_frac", 0.1))

    out_dir = os.path.join("results", "metrics", skew, "local")
    os.makedirs(out_dir, exist_ok=True)

    all_results = {}
    for cid in map(str, range(num_clients)):
        full_ds = load_full_dataset_for(cid, skew_type=skew)
        n = len(full_ds)
        n_test = int(test_frac * n)
        n_val  = int(val_frac  * n)
        n_train= n - n_val - n_test
        train_ds, val_ds, test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        model = train_local_model(train_ds, cfg)
        metrics = evaluate_model(model, test_ds, threshold)
        all_results[cid] = metrics

        # save per-client JSON
        with open(os.path.join(out_dir, f"client_{cid}.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved Client {cid} local metrics to {out_dir}/client_{cid}.json")

    # Summary
    summary = {m: {
        "mean": np.mean([all_results[c][m] for c in all_results]),
        "std":  np.std ([all_results[c][m] for c in all_results]),
    } for m in ["accuracy","precision","recall","f1","roc_auc"]}

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved local summary to {out_dir}/summary.json")

if __name__ == "__main__":
    main()