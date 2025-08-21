
# src/data.py

import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import yaml
import numpy as np


def load_config(path: str) -> dict:
    """
    Load a YAML configuration file and return the config dict.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def preprocess_and_split(config: dict) -> None:
    """
    1) Load raw CSV data
    2) Impute missing values
    3) Encode categorical features and target
    4) Scale numeric features
    5) Partition into clients with optional skew:
       - uniform
       - volume (Dirichlet over total volume)
       - label   (Dirichlet over class labels)
       - feature (Dirichlet over a specified feature column)
    6) Optionally combine skews
    7) Save to data/processed/<skew_type>/client_<i>.pt
    """
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(project_root, "data", "raw", "payment_fraud.csv")

    # 1) Load
    df = pd.read_csv(raw_path)

    # 2) Impute
    cat_cols = ["paymentMethod", "Category"]
    df = df.fillna({col: "UNKNOWN" for col in cat_cols})
    num_cols = ["accountAgeDays", "numItems", "localTime", "paymentMethodAgeDays", "isWeekend"]
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # 3) Encode categorical
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    # 4) Encode target
    if "label" not in df.columns:
        raise KeyError("`label` column not found in raw data.")
    df["label"] = df["label"].astype(int)

    # 5) Scale numeric
    X_df = df.drop(columns=["label"])
    scaler = StandardScaler().fit(X_df[num_cols])
    X_df[num_cols] = scaler.transform(X_df[num_cols])
    X = X_df.values
    y = df["label"].values

    # 6) Partition parameters
    num_clients = int(config.get("num_clients", 5))
    skew_type   = config.get("skew_type", "uniform")
    combine     = bool(config.get("combine_skew", False))
    processed_dir = os.path.join(project_root, "data", "processed", skew_type)
    os.makedirs(processed_dir, exist_ok=True)

    N = len(X)
    classes = np.unique(y)
    rng = np.random.default_rng(config.get("seed", 42))

    # Precompute volume Dirichlet
    alpha_vol  = float(config.get("volume_alpha", 1.0))
    p_vol      = rng.dirichlet([alpha_vol] * num_clients)
    counts_vol = (p_vol * N).astype(int)
    counts_vol[np.argmax(counts_vol)] += N - counts_vol.sum()

    # Precompute label Dirichlet
    alpha_lbl = float(config.get("label_alpha", 0.5))
    P_label = np.stack([
        rng.dirichlet([alpha_lbl] * num_clients)
        for _ in classes
    ], axis=0)

    # Decide indices per client
    client_indices = {i: [] for i in range(num_clients)}

    if skew_type == "uniform":
        # simple equal split
        size = N // num_clients
        for i in range(num_clients):
            start = i * size
            end   = (i + 1) * size if i < num_clients - 1 else N
            client_indices[i] = list(range(start, end))

    elif skew_type == "volume":
        # only volume skew
        idxs = np.arange(N)
        rng.shuffle(idxs)
        start = 0
        for i, c in enumerate(counts_vol):
            client_indices[i] = idxs[start:start+c].tolist()
            start += c

    elif skew_type == "label":
        # only label skew
        for k_idx, k in enumerate(classes):
            idxs_k = np.where(y == k)[0]
            rng.shuffle(idxs_k)
            counts_k = (P_label[k_idx] * len(idxs_k)).astype(int)
            counts_k[np.argmax(P_label[k_idx])] += len(idxs_k) - counts_k.sum()
            start = 0
            for i, c in enumerate(counts_k):
                client_indices[i].extend(idxs_k[start:start+c].tolist())
                start += c
        # enforce uniform volume
        target = N // num_clients
        for i in range(num_clients):
            idxs = client_indices[i]
            if len(idxs) > target:
                client_indices[i] = rng.choice(idxs, size=target, replace=False).tolist()
            elif len(idxs) < target:
                extra = rng.choice(idxs, size=target - len(idxs), replace=True).tolist()
                client_indices[i].extend(extra)

    elif skew_type == "feature":
        # feature-based skew
        feat_col = config.get("feature_column")
        if feat_col not in df.columns:
            raise KeyError(f"Feature column '{feat_col}' not in raw data.")
        feat_vals = df[feat_col].values
        unique_feats = np.unique(feat_vals)
        alpha_ft = float(config.get("feature_alpha", 0.5))
        P_feat = np.stack([
            rng.dirichlet([alpha_ft] * num_clients)
            for _ in unique_feats
        ], axis=0)
        for f_idx, f_val in enumerate(unique_feats):
            idxs_f = np.where(feat_vals == f_val)[0]
            rng.shuffle(idxs_f)
            counts_f = (P_feat[f_idx] * len(idxs_f)).astype(int)
            counts_f[np.argmax(P_feat[f_idx])] += len(idxs_f) - counts_f.sum()
            start = 0
            for i, c in enumerate(counts_f):
                client_indices[i].extend(idxs_f[start:start+c].tolist())
                start += c
        # enforce uniform volume
        target = N // num_clients
        for i in range(num_clients):
            idxs = client_indices[i]
            if len(idxs) > target:
                client_indices[i] = rng.choice(idxs, size=target, replace=False).tolist()
            elif len(idxs) < target:
                extra = rng.choice(idxs, size=target - len(idxs), replace=True).tolist()
                client_indices[i].extend(extra)

    else:
        raise ValueError(f"Unknown skew_type: {skew_type}")

    # combine volume if requested
    if combine and skew_type != "volume":
        for i in range(num_clients):
            idxs = client_indices[i]
            target = counts_vol[i]
            if len(idxs) > target:
                client_indices[i] = rng.choice(idxs, size=target, replace=False).tolist()
            elif len(idxs) < target:
                extra = rng.choice(idxs, size=target - len(idxs), replace=True).tolist()
                client_indices[i].extend(extra)

    # save and report
    actual = []
    for i in range(num_clients):
        idxs = client_indices[i]
        xi, yi = X[idxs], y[idxs]
        torch.save((xi, yi), os.path.join(processed_dir, f"client_{i}.pt"))
        actual.append(len(idxs))
    print(f"[Split={skew_type}, combine={combine}] client sizes: {actual}")


def load_full_dataset_for(client_id: str, skew_type: str = None) -> Dataset:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if skew_type is None:
        cfg = load_config(os.path.join(project_root, "configs", "config.yaml"))
        skew_type = cfg.get("skew_type", "uniform")
    path = os.path.join(project_root, "data", "processed", skew_type, f"client_{client_id}.pt")
    data = torch.load(path, weights_only=False)
    if isinstance(data, Dataset):
        return data
    X_arr, y_arr = data
    return TensorDataset(torch.tensor(X_arr, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.long))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python data.py <config_path>")
        sys.exit(1)
    cfg = load_config(sys.argv[1])
    preprocess_and_split(cfg)
