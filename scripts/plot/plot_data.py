# plot_skew_analysis.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_volume_and_fraud(
    skew_types,
    data_root="data/processed",
    num_clients=5,
    out_root="results/data_plots"
):
    """
    For each skew in `skew_types`, plot:
      - Data volume (# samples) per client
      - Fraud rate (frac positives) per client
    Saves to results/data_plots/{skew}/volume_and_fraud.png
    """
    os.makedirs(out_root, exist_ok=True)
    client_ids = list(range(num_clients))

    for skew in skew_types:
        base = os.path.join(data_root, skew)
        volumes = []
        fraud_rates = []
        labels = [f"Client {i}" for i in client_ids]

        # Load each client shard
        for cid in client_ids:
            path = os.path.join(base, f"client_{cid}.pt")
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Shard not found: {path}")
            X, y = torch.load(path, weights_only=False)
            y = np.array(y)
            n = len(y)
            volumes.append(n)
            fraud_rates.append(y.sum() / n)

        # Plot
        fig, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(num_clients)
        width = 0.35

        # Volume bars
        bar_vol = ax1.bar(
            x - width/2, volumes, width,
            color="skyblue", label="Volume (# samples)"
        )
        ax1.set_ylabel("Volume", color="skyblue")
        ax1.tick_params(axis="y", labelcolor="skyblue")

        # Fraud rate bars on twin axis
        ax2 = ax1.twinx()
        bar_fraud = ax2.bar(
            x + width/2, fraud_rates, width,
            color="crimson", alpha=0.8, label="Fraud Rate"
        )
        ax2.set_ylabel("Fraud Rate", color="crimson")
        ax2.tick_params(axis="y", labelcolor="crimson")
        ax2.set_ylim(0, max(fraud_rates)*1.2 if max(fraud_rates)>0 else 1.0)

        # X‐axis labels
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45)

        plt.title(f"Client Volume & Fraud Rate — {skew.capitalize()} Skew")
        fig.tight_layout()

        # Annotate volumes
        for b in bar_vol:
            h = b.get_height()
            ax1.text(
                b.get_x() + b.get_width()/2,
                h + max(volumes)*0.01,
                f"{int(h)}",
                ha="center",
                va="bottom",
                color="skyblue",
                fontsize=8,
            )
        # Annotate fraud rates
        for b, rate in zip(bar_fraud, fraud_rates):
            h = b.get_height()
            ax2.text(
                b.get_x() + b.get_width()/2,
                h + max(fraud_rates)*0.01,
                f"{rate:.1%}",
                ha="center",
                va="bottom",
                color="crimson",
                fontsize=8,
            )

        # Save figure
        out_dir = os.path.join(out_root, skew)
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, "volume_and_fraud.png")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved volume & fraud plot for '{skew}' at {fig_path}")

if __name__ == "__main__":
    plot_volume_and_fraud(
        skew_types=["uniform", "volume", "label", "feature"],
        data_root="data/processed",
        num_clients=5,
        out_root="results/data_plots"
    )