# evaluate_cm.py

import os
import json
import torch
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import random_split
from src.client import FLClient           # root‐level client
from src.model import create_model, load_config
from src.data import load_full_dataset_for

def main():
    # 1) Load config
    cfg = load_config("configs/config.yaml")
    skew = cfg.get("skew_type", "uniform")
    num_clients = int(cfg["num_clients"])
    threshold   = float(cfg.get("threshold", 0.5))
    test_frac   = float(cfg.get("test_frac", 0.1))
    val_frac    = float(cfg.get("val_frac", 0.1))

    # Prepare output dir
    out_dir = os.path.join("results", "metrics", skew, "confusion")
    os.makedirs(out_dir, exist_ok=True)

    # 2) Build model and load final global weights
    temp_client = FLClient("0", config_path="configs/config.yaml")
    full0 = load_full_dataset_for("0", skew_type=skew)
    x0, _ = full0[0]
    model = create_model(x0.shape[0], config_path="configs/config.yaml")
    ckpt = torch.load("results/model_final.pth", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 3) Collect all preds & truths
    all_true, all_pred = [], []
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
        for x, y in test_ds:
            with torch.no_grad():
                logits = model(x.unsqueeze(0))
                prob   = torch.sigmoid(logits).item()
                pred   = int(prob > threshold)
            all_true.append(int(y.item()))
            all_pred.append(pred)

    # 4) Compute and save
    cm = confusion_matrix(all_true, all_pred)
    report = classification_report(all_true, all_pred, digits=4, output_dict=True)

    # Save confusion matrix as .npy
    np.save(os.path.join(out_dir, "global_cm.npy"), cm)
    # Save classification report as JSON
    with open(os.path.join(out_dir, "global_classification_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"Saved global confusion matrix and report to {out_dir}")

if __name__ == "__main__":
    main()