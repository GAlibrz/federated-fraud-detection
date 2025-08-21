import flwr as fl

# FedAvg strategy on the server
strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.5,
    fraction_evaluate=0.5,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
)

if __name__ == "__main__":
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )