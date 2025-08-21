# evaluate_global_confusion.py

import torch
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix
from torch.utils.data import random_split, DataLoader
from src.data import load_full_dataset_for
from src.model import create_model, load_config

def evaluate_confusion(model, test_ds, threshold):
    model.eval()
    all_preds = []
    all_true  = []
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
    # Load config
    cfg = load_config("configs/config.yaml")
    num_clients = int(cfg["num_clients"])
    threshold   = float(cfg.get("threshold", 0.5))
    test_frac   = float(cfg.get("test_frac", 0.1))
    val_frac    = float(cfg.get("val_frac", 0.1))

    # Load global model
    # Infer input_dim from client 0
    full0 = load_full_dataset_for("0")
    x0, _ = full0[0]
    input_dim = x0.shape[0]
    model = create_model(input_dim, config_path="configs/config.yaml")
    ckpt = torch.load("results/model_final.pth", map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])

    # Evaluate per-client
    for cid in map(str, range(num_clients)):
        # Load full data and split off test
        # New: respect skew_type from config
        skew = cfg.get("skew_type", "uniform")
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

        # Compute confusion matrix
        cm = evaluate_confusion(model, test_ds, threshold)
        print(f"\nClient {cid} confusion matrix (threshold={threshold}):")
        print(cm)

if __name__ == "__main__":
    main()