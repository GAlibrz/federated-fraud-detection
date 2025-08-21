# evaluate_all_skews.py

import os
import json
import torch
import numpy as np
import yaml
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader, random_split

from src.model import create_model, load_config
from src.data import load_full_dataset_for

def evaluate_skew(skew: str, cfg: dict):
    """
    Load the global model for `skew`, evaluate it on each client's test split,
    and write metrics to results/metrics/{skew}/fed_per_client/.
    """
    num_clients = int(cfg["num_clients"])
    test_frac   = float(cfg.get("test_frac", 0.1))
    val_frac    = float(cfg.get("val_frac", 0.1))
    threshold   = float(cfg.get("threshold", 0.5))

    # Paths
    model_path   = os.path.join("results", "models", skew, "model_final.pth")
    metrics_dir  = os.path.join("results", "metrics", skew, "fed_per_client")
    os.makedirs(metrics_dir, exist_ok=True)

    if not os.path.isfile(model_path):
        print(f"[WARN] No model found for skew={skew} at {model_path}, skipping.")
        return

    # 1) Load model
    #    Infer input_dim from client 0
    ds0 = load_full_dataset_for("0", skew_type=skew)
    x0, _ = ds0[0]
    model = create_model(x0.shape[0], config_path="configs/config.yaml")
    ckpt = torch.load(model_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    summary = {}
    # 2) Evaluate per client
    for cid in map(str, range(num_clients)):
        ds = load_full_dataset_for(cid, skew_type=skew)
        n = len(ds)
        n_test = int(test_frac * n)
        n_val  = int(val_frac  * n)
        n_train= n - n_val - n_test

        _, _, test_ds = random_split(
            ds,
            [n_train, n_val, n_test],
            generator=torch.Generator().manual_seed(42),
        )

        y_true, y_pred, y_prob = [], [], []
        loader = DataLoader(test_ds, batch_size=64)
        with torch.no_grad():
            for X, y in loader:
                logits = model(X)
                probs  = torch.sigmoid(logits).cpu().numpy().ravel()
                preds  = (probs > threshold).astype(int)
                y_true.extend(y.numpy().tolist())
                y_pred.extend(preds.tolist())
                y_prob.extend(probs.tolist())

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        auc = roc_auc_score(y_true, y_prob)
        cm  = confusion_matrix(y_true, y_pred).tolist()

        metrics = {
            "accuracy":         acc,
            "precision":        prec,
            "recall":           rec,
            "f1":               f1,
            "roc_auc":          auc,
            "confusion_matrix": cm,
        }
        summary[cid] = metrics

        # write per-client file
        out_path = os.path.join(metrics_dir, f"client_{cid}.json")
        with open(out_path, "w") as fp:
            json.dump(metrics, fp, indent=2)
        print(f"  • [{skew}] client {cid} → metrics saved to {out_path}")

    # 3) summary.json
    summary_path = os.path.join(metrics_dir, "summary.json")
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2)
    print(f"[DONE] Skew={skew}: per-client summary → {summary_path}\n")


def main():
    # Read the shared config for common params
    cfg = load_config("configs/config.yaml")

    # Discover all skews for which a model exists
    models_root = os.path.join("results", "models")
    if not os.path.isdir(models_root):
        raise RuntimeError(f"No 'results/models/' directory found.")
    skews = [
        name for name in os.listdir(models_root)
        if os.path.isdir(os.path.join(models_root, name))
    ]
    print(f"Found skews with saved models: {skews}\n")

    # Evaluate each
    for skew in skews:
        evaluate_skew(skew, cfg)


if __name__ == "__main__":
    main()