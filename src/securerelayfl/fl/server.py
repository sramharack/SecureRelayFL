"""
SecureRelayFL — Flower FL Server + Simulation Runner

Runs federated training across 5 facility clients using Flower's simulation
engine. Supports FedAvg with hooks for impairment injection.

Usage:
    python -m securerelayfl.fl.server --model cnn --rounds 30 --seed 42
    python -m securerelayfl.fl.server --model tcn --rounds 30 --local-epochs 5
"""

import argparse
import json
import os
import time
from collections import OrderedDict

import flwr as fl
from flwr.common import Metrics
from flwr.server.strategy import FedAvg
import numpy as np
import torch

from securerelayfl.models.fault_classifier import FaultClassifier, MultiTaskLoss
from securerelayfl.models.tcn_classifier import FaultClassifierTCN
from securerelayfl.fl.client import client_fn

NUM_FACILITIES = 5


def weighted_average(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """Aggregate evaluation metrics across clients (weighted by n_samples)."""
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    result = {}
    for key in ["fault_type_acc", "fault_zone_acc", "protection_action_acc"]:
        result[key] = sum(n * m.get(key, 0.0) for n, m in metrics) / total
    return result


def get_initial_parameters(model_name: str) -> fl.common.Parameters:
    """Get initial model parameters for the server."""
    if model_name == "cnn":
        model = FaultClassifier()
    elif model_name == "tcn":
        model = FaultClassifierTCN()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)


def get_evaluate_fn(model_name: str, data_dir: str, device: str, seed: int):
    """
    Return a server-side evaluation function.
    Evaluates the global model on pooled test data after each round.
    """
    from data.dataset import get_dataloaders

    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir,
        facility_ids=None,  # All facilities pooled
        batch_size=64,
        val_split=0.15,
        test_split=0.15,
        seed=seed,
    )

    if model_name == "cnn":
        model = FaultClassifier().to(device)
    else:
        model = FaultClassifierTCN().to(device)
    criterion = MultiTaskLoss().to(device)

    def evaluate(server_round: int, parameters, config):
        # Load global parameters into model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v).to(device) for k, v in params_dict}
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()

        total_loss = 0.0
        correct_ft, correct_fz, correct_pa = 0, 0, 0
        n_samples = 0

        with torch.no_grad():
            for wf, ft, fz, pa in test_loader:
                wf = wf.to(device)
                ft, fz, pa = ft.to(device), fz.to(device), pa.to(device)

                preds = model(wf)
                loss, _ = criterion(preds, ft, fz, pa)

                bs = wf.size(0)
                total_loss += loss.item() * bs
                correct_ft += (preds["fault_type"].argmax(1) == ft).sum().item()
                correct_fz += (preds["fault_zone"].argmax(1) == fz).sum().item()
                correct_pa += (preds["protection_action"].argmax(1) == pa).sum().item()
                n_samples += bs

        avg_loss = total_loss / max(n_samples, 1)
        metrics = {
            "fault_type_acc": correct_ft / max(n_samples, 1),
            "fault_zone_acc": correct_fz / max(n_samples, 1),
            "protection_action_acc": correct_pa / max(n_samples, 1),
        }
        return float(avg_loss), metrics

    return evaluate


def run_simulation(args):
    """Run FL simulation using Flower."""

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"

    # ---- Strategy ----
    strategy = FedAvg(
        fraction_fit=1.0,           # All clients participate every round
        fraction_evaluate=1.0,
        min_fit_clients=NUM_FACILITIES,
        min_evaluate_clients=NUM_FACILITIES,
        min_available_clients=NUM_FACILITIES,
        initial_parameters=get_initial_parameters(args.model),
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(args.model, args.data_dir, device, args.seed),
        on_fit_config_fn=lambda r: {
            "local_epochs": args.local_epochs,
        },
    )

    # ---- Client resources ----
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}

    # ---- Run config passed to each client ----
    run_config = {
        "model": args.model,
        "data_dir": args.data_dir,
        "local_epochs": str(args.local_epochs),
        "batch_size": str(args.batch_size),
        "lr": str(args.lr),
        "device": device,
        "seed": str(args.seed),
    }

    print(f"Starting FL simulation: {args.model.upper()}, "
          f"{args.rounds} rounds, {args.local_epochs} local epochs/round")
    print(f"Device: {device} | Clients: {NUM_FACILITIES}")

    t0 = time.time()

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_FACILITIES,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources=client_resources,
        actor_kwargs={"run_config": run_config},
        ray_init_args={"num_gpus": 0},
    )

    elapsed = time.time() - t0
    print(f"\nSimulation complete in {elapsed:.1f}s")

    # ---- Extract and save results ----
    os.makedirs(args.output_dir, exist_ok=True)

    results = {
        "args": vars(args),
        "training_time_s": elapsed,
        "rounds_centralized_loss": [],
        "rounds_centralized_metrics": [],
        "rounds_distributed_loss": [],
        "rounds_distributed_metrics": [],
    }

    # Server-side (centralized) evaluation per round
    if history.losses_centralized:
        for r, loss in history.losses_centralized:
            results["rounds_centralized_loss"].append({"round": r, "loss": loss})
    if history.metrics_centralized:
        for key, values in history.metrics_centralized.items():
            for r, val in values:
                # Find or create entry for this round
                entry = next(
                    (e for e in results["rounds_centralized_metrics"] if e["round"] == r),
                    None,
                )
                if entry is None:
                    entry = {"round": r}
                    results["rounds_centralized_metrics"].append(entry)
                entry[key] = val

    # Client-side (distributed) evaluation per round
    if history.losses_distributed:
        for r, loss in history.losses_distributed:
            results["rounds_distributed_loss"].append({"round": r, "loss": loss})
    if history.metrics_distributed:
        for key, values in history.metrics_distributed.items():
            for r, val in values:
                entry = next(
                    (e for e in results["rounds_distributed_metrics"] if e["round"] == r),
                    None,
                )
                if entry is None:
                    entry = {"round": r}
                    results["rounds_distributed_metrics"].append(entry)
                entry[key] = val

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_dir}/results.json")

    # Print final round metrics
    if results["rounds_centralized_metrics"]:
        final = results["rounds_centralized_metrics"][-1]
        print(f"\nFinal global test metrics (round {final.get('round', '?')}):")
        for k, v in final.items():
            if k != "round":
                print(f"  {k}: {v:.4f}")


def main():
    parser = argparse.ArgumentParser(description="SecureRelayFL FL Simulation")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "tcn"])
    parser.add_argument("--rounds", type=int, default=30,
                        help="Number of FL communication rounds")
    parser.add_argument("--local-epochs", type=int, default=3,
                        help="Local training epochs per round per client")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"results/{args.model}_fedavg"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_simulation(args)


if __name__ == "__main__":
    main()