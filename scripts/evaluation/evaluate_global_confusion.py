# evaluate_global_confusion.py

import os
import json
import torch
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split, DataLoader
from src.data import load_full_dataset_for, load_config
from src.model import create_model

def evaluate_confusion(model, test_ds, threshold):
    model.eval()
    all_preds, all_true = [], []
    loader = DataLoader(test_ds, batch_size=64)
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > threshold).astype(int)
            all_preds.append(preds)
            all_true.append(y.numpy())
    all_preds = np.concatenate(all_preds)
    all_true  = np.concatenate(all_true)
    return confusion_matrix(all_true, all_preds)

def main():
    cfg = load_config("configs/config.yaml")
    skew = cfg.get("skew_type", "uniform")
    num_clients = int(cfg["num_clients"])
    threshold   = float(cfg.get("threshold", 0.5))
    test_frac   = float(cfg.get("test_frac", 0.1))
    val_frac    = float(cfg.get("val_frac", 0.1))

    out_dir = os.path.join("results", "metrics", skew, "per_client_confusion")
    os.makedirs(out_dir, exist_ok=True)

    # Load global model
    full0 = load_full_dataset_for("0", skew_type=skew)
    input_dim = full0[0][0].shape[0]
    model = create_model(input_dim, config_path="configs/config.yaml")
    ckpt = torch.load("results/model_final.pth", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    # Per-client evaluation
    summary = {}
    for cid in map(str, range(num_clients)):
        full_ds = load_full_dataset_for(cid, skew_type=skew)
        n = len(full_ds)
        n_test = int(test_frac * n)
        n_val  = int(val_frac  * n)
        n_train= n - n_val - n_test
        _, _, test_ds = random_split(
            full_ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        cm = evaluate_confusion(model, test_ds, threshold)
        # Save each as .npy
        np.save(os.path.join(out_dir, f"client_{cid}_cm.npy"), cm)
        summary[cid] = cm.tolist()

    # Save summary JSON
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved per-client confusion matrices to {out_dir}")

if __name__ == "__main__":
    main()