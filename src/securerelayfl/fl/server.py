"""
SecureRelayFL — Flower FL Server (v2)

v2 changes:
    - Supports --model cnn_v2 (FaultClassifierV2 with config features)
    - evaluate_fn unpacks config_features and passes to model
    - --fedprox-mu for FedProx support
    - Saves per-round metrics to JSON

Usage:
    # FedAvg ideal
    python -m securerelayfl.fl.server --model cnn_v2 --rounds 50 --lr 3e-4 --seed 42

    # FedProx
    python -m securerelayfl.fl.server --model cnn_v2 --rounds 50 --lr 3e-4 \\
        --fedprox-mu 0.01 --seed 42 --output-dir results/cnn_v2_fedprox
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.dataset import FaultWaveformDataset
from securerelayfl.fl.client import FaultClient, get_parameters, set_parameters


def get_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "cnn_v1":
        from securerelayfl.models.fault_classifier import FaultClassifier
        return FaultClassifier().to(device)
    elif model_name == "cnn_v2":
        from securerelayfl.models.fault_classifier_v2 import FaultClassifierV2
        return FaultClassifierV2().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_evaluate_fn(
    model_name: str,
    data_dir: str,
    device: torch.device,
    output_dir: Path,
):
    """Return a server-side evaluation function (centralized eval on all data)."""

    dataset = FaultWaveformDataset(data_dir=data_dir, facility_ids=list(range(5)))
    # Use 20% as server-side test set
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    _, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(999),  # fixed split
    )
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)

    model = get_model(model_name, device)
    round_metrics = []

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: dict
    ) -> tuple[float, dict] | None:
        set_parameters(model, parameters)
        model.eval()
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_ft, correct_fz, correct_pa = 0, 0, 0
        total = 0

        with torch.no_grad():
            for batch in val_loader:
                wf, cf, ft, fz, pa = batch
                wf, cf = wf.to(device), cf.to(device)
                ft, fz, pa = ft.to(device), fz.to(device), pa.to(device)

                if model_name == "cnn_v2":
                    preds = model(wf, cf)
                else:
                    preds = model(wf)

                loss = (
                    loss_fn(preds["fault_type"], ft)
                    + loss_fn(preds["fault_zone"], fz)
                    + loss_fn(preds["protection_action"], pa)
                )
                total_loss += loss.item() * ft.size(0)
                correct_ft += (preds["fault_type"].argmax(1) == ft).sum().item()
                correct_fz += (preds["fault_zone"].argmax(1) == fz).sum().item()
                correct_pa += (preds["protection_action"].argmax(1) == pa).sum().item()
                total += ft.size(0)

        avg_loss = total_loss / max(total, 1)
        metrics = {
            "acc_ft": correct_ft / max(total, 1),
            "acc_fz": correct_fz / max(total, 1),
            "acc_pa": correct_pa / max(total, 1),
        }

        round_metrics.append({"round": server_round, "loss": avg_loss, **metrics})
        print(f"  Round {server_round:3d} | loss={avg_loss:.4f} | "
              f"FT={metrics['acc_ft']:.3f}  FZ={metrics['acc_fz']:.3f}  PA={metrics['acc_pa']:.3f}")

        # Save metrics incrementally
        with open(output_dir / "round_metrics.json", "w") as f:
            json.dump(round_metrics, f, indent=2)

        # Save best model
        if server_round == 1 or metrics["acc_pa"] >= max(
            m["acc_pa"] for m in round_metrics[:-1]
        ):
            torch.save(model.state_dict(), output_dir / "best_model.pt")

        return avg_loss, metrics

    return evaluate


def client_fn(
    cid: str,
    model_name: str,
    data_dir: str,
    local_epochs: int,
    batch_size: int,
    lr: float,
    fedprox_mu: float,
) -> FaultClient:
    """Flower client factory for simulation."""
    return FaultClient(
        facility_id=int(cid),
        model_name=model_name,
        data_dir=data_dir,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        fedprox_mu=fedprox_mu,
        device=torch.device("cpu"),  # clients always on CPU for Ray sim
    )


def main():
    parser = argparse.ArgumentParser(description="SecureRelayFL FL server")
    parser.add_argument("--model", type=str, default="cnn_v2", choices=["cnn_v1", "cnn_v2"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--fedprox-mu", type=float, default=0.0,
                        help="FedProx proximal term weight. 0 = FedAvg.")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_clients = 5

    if args.output_dir is None:
        tag = "fedprox" if args.fedprox_mu > 0 else "fedavg"
        args.output_dir = f"results/{args.model}_{tag}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # Initial model parameters
    init_model = get_model(args.model, device)
    init_params = ndarrays_to_parameters(get_parameters(init_model))
    print(f"Model: {args.model} | Params: {init_model.count_parameters():,}")
    print(f"Strategy: {'FedProx' if args.fedprox_mu > 0 else 'FedAvg'} | "
          f"Rounds: {args.rounds} | Clients: {n_clients}")

    # Strategy
    strategy = FedAvg(
        fraction_fit=1.0,           # all clients every round
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=init_params,
        evaluate_fn=get_evaluate_fn(args.model, args.data_dir, device, out_dir),
    )

    # Client factory
    client_fn_partial = partial(
        client_fn,
        model_name=args.model,
        data_dir=args.data_dir,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        fedprox_mu=args.fedprox_mu,
    )

    t0 = time.time()

    # Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn_partial,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        ray_init_args={"num_gpus": 0},
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )

    elapsed = time.time() - t0
    print(f"\nSimulation complete in {elapsed:.1f}s")
    print(f"Results saved to {out_dir}")


if __name__ == "__main__":
    main()