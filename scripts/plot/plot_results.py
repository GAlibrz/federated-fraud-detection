# plot_results.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt

def load_metrics(skew: str):
    base = os.path.join("results", "metrics", skew)

    # Centralized
    cen_dir = os.path.join(base, "centralized")
    cen_metrics = json.load(open(os.path.join(cen_dir, "metrics.json")))
    cen_cm = np.load(os.path.join(cen_dir, "confusion_matrix.npy"))

    # Federated/global combined report
    fed_dir = os.path.join(base, "confusion")
    fed_report = json.load(open(os.path.join(fed_dir, "global_classification_report.json")))
    fed_metrics = {
        "accuracy": fed_report["accuracy"],
        "precision": fed_report["1"]["precision"],
        "recall":    fed_report["1"]["recall"],
        "f1":        fed_report["1"]["f1-score"],
        # `roc_auc` may be absent or None
        "roc_auc":   fed_report.get("roc_auc", None),
    }

    # Local summary
    loc_dir = os.path.join(base, "local")
    loc_summary = json.load(open(os.path.join(loc_dir, "summary.json")))

    # Per-client confusion matrices
    per_client_dir = os.path.join(base, "per_client_confusion")
    client_cms = {}
    if os.path.isdir(per_client_dir):
        for fname in os.listdir(per_client_dir):
            if fname.endswith("_cm.npy"):
                cid = fname.split("_")[1]
                cm = np.load(os.path.join(per_client_dir, fname))
                client_cms[cid] = cm

    # Federated training history
    hist_acc_path = os.path.join("results", "history_accuracy.txt")
    hist_f1_path  = os.path.join("results", "history_f1.txt")
    history = {"rounds": None, "accuracy": None, "f1": None}
    if os.path.isfile(hist_acc_path) and os.path.isfile(hist_f1_path):
        hist_acc = np.loadtxt(hist_acc_path)
        hist_f1  = np.loadtxt(hist_f1_path)
        history = {
            "rounds": np.arange(1, len(hist_acc) + 1),
            "accuracy": hist_acc,
            "f1": hist_f1,
        }

    return cen_metrics, cen_cm, fed_metrics, loc_summary, client_cms, history

def plot_comparison(skew: str):
    cen_metrics, cen_cm, fed_metrics, loc_summary, client_cms, history = load_metrics(skew)

    # Only include metrics that are not None in all three sources
    all_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    keys = [k for k in all_keys
            if cen_metrics.get(k) is not None
            and fed_metrics.get(k)    is not None
            and loc_summary.get(k, {}).get("mean") is not None]

    # Assemble values
    cen_vals = [cen_metrics[k] for k in keys]
    loc_vals = [loc_summary[k]["mean"] for k in keys]
    fed_vals = [fed_metrics[k] for k in keys]

    x = np.arange(len(keys))
    width = 0.25

    # Prepare plot folder
    plot_dir = os.path.join("results", "plots", skew)
    os.makedirs(plot_dir, exist_ok=True)

    # 1) Centralized vs Local vs Federated bar chart
    plt.figure(figsize=(8, 4))
    plt.bar(x - width, cen_vals, width, label="Centralized")
    plt.bar(x,        loc_vals, width, label="Local Avg")
    plt.bar(x + width, fed_vals, width, label="Federated")
    plt.xticks(x, [k.capitalize() for k in keys], rotation=45)
    plt.ylabel("Score")
    plt.title(f"Centralized vs Local vs Federated ({skew} skew)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "comparison_bar.png"))
    plt.show()

    # 2) Centralized confusion matrix heatmap
    plt.figure(figsize=(4, 4))
    plt.imshow(cen_cm, interpolation="nearest", cmap="Blues")
    plt.title("Centralized Model Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "centralized_cm.png"))
    plt.show()

    # 3) Per-client confusion matrices
    for cid, cm in client_cms.items():
        plt.figure(figsize=(3, 3))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Client {cid} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f"client_{cid}_cm.png"))
        plt.close()

    # 4) Federated training history
    if history["rounds"] is not None:
        rounds = history["rounds"]
        plt.figure(figsize=(6, 3))
        plt.plot(rounds, history["accuracy"], marker="o", label="Accuracy")
        plt.plot(rounds, history["f1"],        marker="s", label="F1 Score")
        plt.xlabel("Round")
        plt.ylabel("Metric")
        plt.title(f"Federated Training History ({skew} skew)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "fed_history.png"))
        plt.show()

if __name__ == "__main__":
    for skew in ["uniform", "volume", "label", "feature"]:
        print(f"Plotting results for skew: {skew}")
        plot_comparison(skew)