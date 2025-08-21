# plot_fl_benefit.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_all_metrics(skew: str):
    base = os.path.join("results", "metrics", skew)

    # Local summary
    loc = json.load(open(os.path.join(base, "local", "summary.json")))
    local_recall = loc["recall"]["mean"]
    local_f1     = loc["f1"]["mean"]

    # Federated/global
    fed_report = json.load(open(os.path.join(base, "confusion", "global_classification_report.json")))
    fed_recall = fed_report["1"]["recall"]
    fed_f1     = fed_report["1"]["f1-score"]

    # Centralized
    cen = json.load(open(os.path.join(base, "centralized", "metrics.json")))
    cen_recall = cen["recall"]
    cen_f1     = cen["f1"]

    return (local_recall, local_f1), (fed_recall, fed_f1), (cen_recall, cen_f1)

def plot_benefit(skew: str):
    (loc_rec, loc_f1), (fed_rec, fed_f1), (cen_rec, cen_f1) = load_all_metrics(skew)

    metrics = ["Recall", "F1 Score"]
    local_vals = [loc_rec, loc_f1]
    fed_vals   = [fed_rec, fed_f1]
    cen_vals   = [cen_rec, cen_f1]

    x = np.arange(len(metrics))
    width = 0.2

    plt.figure(figsize=(6,4))
    plt.bar(x - width, local_vals, width, label="Local Only", color="lightgray")
    plt.bar(x,        fed_vals,   width, label="Federated", color="cornflowerblue")
    plt.bar(x + width,cen_vals,   width, label="Centralized", color="seagreen")
    plt.xticks(x, metrics)
    plt.ylim(0,1)
    plt.ylabel("Score")
    plt.title(f"Benefit of FL vs Local (and Centralized Upper Bound)\n[{skew.capitalize()} Skew]")
    plt.legend()
    plt.tight_layout()

    out_dir = os.path.join("results", "plots", "benefit")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{skew}_fl_benefit.png")
    plt.savefig(out_path)
    plt.show()
    print(f"Saved FL benefit plot at {out_path}")

if __name__ == "__main__":
    for skew in ["uniform", "volume", "label", "feature", "combined"]:
        print(f"Plotting FL benefit for skew: {skew}")
        plot_benefit(skew)