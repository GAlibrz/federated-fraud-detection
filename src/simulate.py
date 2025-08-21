
import os
import warnings
import logging
import ray
import yaml
import torch

# 1) Silence DeprecationWarning
warnings.simplefilter("ignore", category=DeprecationWarning)

# 2) Initialize Ray without driver logging
ray.init(log_to_driver=False, ignore_reinit_error=True, configure_logging=False)

# 3) Turn down Ray & Flower logs
logging.getLogger("ray").setLevel(logging.ERROR)
logging.getLogger("flwr").setLevel(logging.WARNING)

import flwr as fl
from flwr.common.parameter import parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from src.client import FLClient                # your root‐level client
from src.model import create_model, load_config


class SaveModelFedAvg(FedAvg):
    """FedAvg that saves the last aggregated Parameters."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_parameters = None

    def aggregate_fit(self, rnd, results, failures):
        # Call parent and capture its return
        aggregated = super().aggregate_fit(rnd, results, failures)
        if aggregated is not None:
            # aggregated[0] is a `Parameters` object
            self.latest_parameters = aggregated[0]
        return aggregated


def weighted_metric_aggregation(
    results: list[tuple[int, dict[str, float]]]
) -> dict[str, float]:
    """Aggregate per-client metrics into a single dict."""
    total_examples = sum(n for n, _ in results)
    agg: dict[str, float] = {}
    # Initialize keys
    for _, metrics in results:
        for name in metrics:
            agg.setdefault(name, 0.0)
    # Sum weighted by example count
    for n, metrics in results:
        for name, value in metrics.items():
            agg[name] += n * value
    # Normalize
    return {name: val / total_examples for name, val in agg.items()}


def client_fn(cid: str) -> fl.client.Client:
    """Create a Flower client."""
    return FLClient(cid, config_path="configs/config.yaml").to_client()


def main() -> None:
    # Load config
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    num_clients = int(cfg.get("num_clients", 5))
    num_rounds  = int(cfg.get("rounds",      10))

    # Strategy that saves final parameters
    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        evaluate_metrics_aggregation_fn=weighted_metric_aggregation,
    )

    # Run simulation
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # Print loss
    print("Simulation history (loss):")
    for rnd, loss_val in history.losses_distributed:
        print(f"  round {rnd}: {loss_val:.6f}")

    # Print metrics
    metric_names = list(history.metrics_distributed.keys())
    print("\nAvailable metrics:", metric_names)
    for name, values in history.metrics_distributed.items():
        print(f"\nSimulation history ({name}):")
        for rnd, val in values:
            print(f"  round {rnd}: {val:.6f}")

    # --- Save final model ---
    if strategy.latest_parameters is None:
        raise RuntimeError("No aggregated parameters to save.")

    # 1) Reconstruct model
    temp_client = FLClient("0", config_path="configs/config.yaml")
    example_x, _ = next(iter(temp_client.train_loader))
    input_dim = example_x.shape[1]
    model = create_model(input_dim, config_path="configs/config.yaml")

    # 2) Convert Parameters → list of NumPy ndarrays
    ndarrays = parameters_to_ndarrays(strategy.latest_parameters)

    # 3) Map to state_dict keys & load
    keys = list(model.state_dict().keys())
    state_dict = {k: torch.tensor(v) for k, v in zip(keys, ndarrays)}
    model.load_state_dict(state_dict)

    # 4) Save checkpoint
    skew = cfg.get("skew_type", "uniform")

    # Build the output directory: results/models/{skew}/
    model_dir = os.path.join("results", "models", skew)
    os.makedirs(model_dir, exist_ok=True)

    # Save the final global model there
    save_path = os.path.join(model_dir, "model_final.pth")
    torch.save({"state_dict": state_dict}, save_path)
    print(f"Saved final global model to {save_path}")


if __name__ == "__main__":
    main()