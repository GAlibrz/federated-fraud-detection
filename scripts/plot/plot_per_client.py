# plot_per_client.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_fed_metrics(skew: str):
    """
    Load all client_{cid}.json files from results/metrics/{skew}/fed_per_client/
    Returns a dict cid -> metrics.
    """
    base = os.path.join("results", "metrics", skew, "fed_per_client")
    if not os.path.isdir(base):
        raise FileNotFoundError(f"No federated metrics folder: {base}")
    data = {}
    for fname in os.listdir(base):
        if fname.startswith("client_") and fname.endswith(".json"):
            cid = fname.split("_")[1].split(".")[0]
            path = os.path.join(base, fname)
            data[cid] = json.load(open(path))
    return data

def plot_per_client_all(skew: str):
    """
    For a given skew, plot per-client metrics (recall, f1, precision, roc_auc)
    and save under results/metrics/per_client/{skew}/ with expressive filenames.
    """
    metrics_dict = load_fed_metrics(skew)
    client_ids = sorted(metrics_dict.keys(), key=int)
    n = len(client_ids)

    # Select metrics to plot
    metrics = ["recall", "f1", "precision", "roc_auc"]

    # Prepare output folder
    out_dir = os.path.join("results", "metrics", "per_client", skew)
    os.makedirs(out_dir, exist_ok=True)

    x = np.arange(n)
    labels = [f"Client {cid}" for cid in client_ids]

    for metric in metrics:
        # Gather values, skip if missing
        vals = [metrics_dict[cid].get(metric) for cid in client_ids]
        if any(v is None for v in vals):
            continue

        plt.figure(figsize=(8, 4))
        bars = plt.bar(x, vals, color="cornflowerblue")
        plt.xticks(x, labels, rotation=45)
        plt.ylabel(metric.capitalize())
        plt.title(f"{skew.capitalize()} Skew: {metric.capitalize()} per Client")
        plt.ylim(0, 1)

        # Annotate bar values
        for bar, v in zip(bars, vals):
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                v + 0.01,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        filename = f"{skew}_{metric}_per_client.png"
        plt.savefig(os.path.join(out_dir, filename))
        plt.close()
        print(f"Saved plot: {os.path.join(out_dir, filename)}")

def main():
    root = os.path.join("results", "metrics")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"No metrics root directory: {root}")

    # Determine skews (exclude the helper 'per_client' folder)
    skews = [
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d != "per_client"
    ]

    for skew in skews:
        fed_folder = os.path.join(root, skew, "fed_per_client")
        if os.path.isdir(fed_folder):
            print(f"\nPlotting per-client metrics for skew = {skew}")
            plot_per_client_all(skew)
        else:
            print(f"Skipping skew {skew}: no federated metrics found.")

if __name__ == "__main__":
    main()