# evaluate_centralized.py

import os
import json
import torch
import yaml
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_auc_score,
    accuracy_score,
)
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from src.model import create_model, load_config
from src.data import load_full_dataset_for

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        focal = (1 - pt) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        loss = alpha_t * focal * bce
        return loss.mean() if self.reduction=='mean' else loss.sum()

def main():
    # 1) Load config
    cfg = load_config("configs/config.yaml")
    skew = cfg.get("skew_type", "uniform")
    test_frac = float(cfg.get("test_frac", 0.1))
    val_frac  = float(cfg.get("val_frac", 0.1))
    threshold = float(cfg.get("threshold", 0.5))

    # 2) Gather all data across clients
    num_clients = int(cfg["num_clients"])
    X_list, y_list = [], []
    for cid in map(str, range(num_clients)):
        ds = load_full_dataset_for(cid, skew_type=skew)
        X_c = ds.tensors[0].numpy()
        y_c = ds.tensors[1].numpy()
        X_list.append(X_c)
        y_list.append(y_c)
    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # 3) Stratified split into train / val / test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=42
    )
    rel_val = val_frac / (1 - test_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, stratify=y_trainval, random_state=42
    )

    # Wrap into TensorDatasets
    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                             torch.tensor(y_train, dtype=torch.long))
    val_ds   = TensorDataset(torch.tensor(X_val,   dtype=torch.float32),
                             torch.tensor(y_val,   dtype=torch.long))
    test_ds  = TensorDataset(torch.tensor(X_test,  dtype=torch.float32),
                             torch.tensor(y_test,  dtype=torch.long))

    # 4) DataLoader with weighted sampler
    labels = torch.tensor(y_train, dtype=torch.long)
    weights_per_class = 1.0 / torch.bincount(labels).float()
    sample_weights = weights_per_class[labels]
    sampler = WeightedRandomSampler(weights=sample_weights,
                                    num_samples=len(sample_weights),
                                    replacement=True)
    batch_size = int(cfg["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # 5) Build model & training setup
    input_dim = X_train.shape[1]
    model = create_model(input_dim, config_path="configs/config.yaml")
    device = torch.device("cpu")
    model.to(device)

    # Loss & optimizer
    alpha = float(cfg.get("focal_alpha", 0.25))
    gamma = float(cfg.get("focal_gamma", 2.0))
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    opt = torch.optim.Adam(model.parameters(),
                           lr=float(cfg["lr"]),
                           weight_decay=float(cfg["weight_decay"]))
    sched_cfg = cfg.get("scheduler", {})
    scheduler = torch.optim.lr_scheduler.StepLR(
        opt, step_size=int(sched_cfg.get("step_size",5)),
        gamma=float(sched_cfg.get("gamma",0.5))
    )
    eps = float(cfg.get("label_smoothing", 0.0))
    epochs = int(cfg["local_epochs"])

    # 6) Training loop
    for epoch in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            opt.zero_grad()
            logits = model(Xb)
            yb_smooth = yb.float() * (1-eps) if eps>0 else yb.float()
            loss = criterion(logits, yb_smooth)
            loss.backward()
            opt.step()
        scheduler.step()

    # 7) Evaluate on test set
    all_probs, all_preds, all_true = [], [], []
    with torch.no_grad():
        for Xb, yb in test_loader:
            logits = model(Xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > threshold).astype(int)
            all_probs.append(probs)
            all_preds.append(preds)
            all_true.append(yb.numpy())
    all_probs = np.concatenate(all_probs)
    all_preds = np.concatenate(all_preds)
    all_true = np.concatenate(all_true)

    # Metrics
    cm = confusion_matrix(all_true, all_preds)
    acc = accuracy_score(all_true, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(all_true, all_preds,
                                                        average="binary", zero_division=0)
    auc = roc_auc_score(all_true, all_probs)
    report = {
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
    }

    # 8) Save results
    out_dir = os.path.join("results", "metrics", skew, "centralized")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    print(f"Saved centralized evaluation to {out_dir}")

if __name__ == "__main__":
    import json
    from sklearn.metrics import confusion_matrix
    from src.data import load_full_dataset_for
    main()