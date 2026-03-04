"""
SecureRelayFL — Axis 1: Network Impairment Sweep

Runs FedAvg under varying network conditions:
    - Packet loss: 0%, 5%, 10%, 15%, 25%
    - Quantization: 32-bit, 16-bit, 8-bit
    - Channel noise: 0, 0.001, 0.005, 0.01

Usage:
    python -m securerelayfl.experiments.axis1_impairment --seed 42
    python -m securerelayfl.experiments.axis1_impairment --rounds 50 --quick
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from itertools import product

import numpy as np
import torch
import flwr as fl
from flwr.server import ServerConfig

from securerelayfl.models.fault_classifier import FaultClassifier, MultiTaskLoss
from securerelayfl.fl.client import make_client_fn
from securerelayfl.fl.impaired_strategy import ImpairedFedAvg


NUM_FACILITIES = 5


def get_initial_parameters(model_name: str):
    model = FaultClassifier()
    ndarrays = [val.cpu().numpy() for _, val in model.state_dict().items()]
    return fl.common.ndarrays_to_parameters(ndarrays)


def weighted_average(metrics):
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    result = {}
    for key in ["fault_type_acc", "fault_zone_acc", "protection_action_acc"]:
        result[key] = sum(n * m.get(key, 0.0) for n, m in metrics) / total
    return result


def get_evaluate_fn(data_dir, device, seed):
    from data.dataset import get_dataloaders

    _, _, test_loader, _ = get_dataloaders(
        data_dir=data_dir, facility_ids=None,
        batch_size=64, val_split=0.15, test_split=0.15, seed=seed,
    )
    model = FaultClassifier().to(device)
    criterion = MultiTaskLoss().to(device)

    def evaluate(server_round, parameters, config):
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
                wf, ft, fz, pa = (x.to(device) for x in (wf, ft, fz, pa))
                preds = model(wf)
                loss, _ = criterion(preds, ft, fz, pa)
                bs = wf.size(0)
                total_loss += loss.item() * bs
                correct_ft += (preds["fault_type"].argmax(1) == ft).sum().item()
                correct_fz += (preds["fault_zone"].argmax(1) == fz).sum().item()
                correct_pa += (preds["protection_action"].argmax(1) == pa).sum().item()
                n_samples += bs

        avg_loss = total_loss / max(n_samples, 1)
        return float(avg_loss), {
            "fault_type_acc": correct_ft / max(n_samples, 1),
            "fault_zone_acc": correct_fz / max(n_samples, 1),
            "protection_action_acc": correct_pa / max(n_samples, 1),
        }

    return evaluate


def run_one_experiment(config, args, device):
    """Run a single impairment configuration."""
    label = config["label"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  packet_loss={config['packet_loss']}, quant={config['quantize_bits']}bit, "
          f"noise={config['noise_std']}")
    print(f"{'='*60}")

    strategy = ImpairedFedAvg(
        packet_loss_rate=config["packet_loss"],
        quantize_bits=config["quantize_bits"],
        noise_std=config["noise_std"],
        seed=args.seed,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_FACILITIES,
        min_evaluate_clients=NUM_FACILITIES,
        min_available_clients=NUM_FACILITIES,
        initial_parameters=get_initial_parameters("cnn"),
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(args.data_dir, device, args.seed),
        on_fit_config_fn=lambda r: {"local_epochs": args.local_epochs},
    )

    cfn = make_client_fn(
        model_name="cnn",
        data_dir=args.data_dir,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device="cpu",
        seed=args.seed,
    )

    t0 = time.time()
    history = fl.simulation.start_simulation(
        client_fn=cfn,
        num_clients=NUM_FACILITIES,
        config=ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"num_gpus": 0, "include_dashboard": False},
    )
    elapsed = time.time() - t0

    # Extract final and best metrics
    cm = []
    if history.metrics_centralized:
        rounds_seen = set()
        for key, values in history.metrics_centralized.items():
            for r, val in values:
                if r not in rounds_seen:
                    rounds_seen.add(r)
                    cm.append({"round": r})
                entry = next(e for e in cm if e["round"] == r)
                entry[key] = val

    best_pa = max(cm, key=lambda x: x.get("protection_action_acc", 0)) if cm else {}
    final = cm[-1] if cm else {}

    result = {
        "config": config,
        "training_time_s": elapsed,
        "best_round": best_pa,
        "final_round": final,
        "impairment_log": strategy.impairment_log,
        "all_rounds": cm,
    }

    print(f"  Time: {elapsed:.0f}s")
    if final:
        print(f"  Final: FT={final.get('fault_type_acc',0):.3f} "
              f"FZ={final.get('fault_zone_acc',0):.3f} "
              f"PA={final.get('protection_action_acc',0):.3f}")
    if best_pa:
        print(f"  Best PA: round {best_pa.get('round','?')} "
              f"FT={best_pa.get('fault_type_acc',0):.3f} "
              f"PA={best_pa.get('protection_action_acc',0):.3f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Axis 1: Network Impairment Sweep")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/axis1_impairment")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced sweep for testing")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- Define experiment grid ----
    if args.quick:
        configs = [
            {"label": "ideal", "packet_loss": 0.0, "quantize_bits": None, "noise_std": 0.0},
            {"label": "loss_10pct", "packet_loss": 0.10, "quantize_bits": None, "noise_std": 0.0},
            {"label": "quant_8bit", "packet_loss": 0.0, "quantize_bits": 8, "noise_std": 0.0},
            {"label": "noise_0.01", "packet_loss": 0.0, "quantize_bits": None, "noise_std": 0.01},
        ]
    else:
        configs = [
            # Baseline
            {"label": "ideal", "packet_loss": 0.0, "quantize_bits": None, "noise_std": 0.0},
            # Packet loss sweep
            {"label": "loss_5pct", "packet_loss": 0.05, "quantize_bits": None, "noise_std": 0.0},
            {"label": "loss_10pct", "packet_loss": 0.10, "quantize_bits": None, "noise_std": 0.0},
            {"label": "loss_15pct", "packet_loss": 0.15, "quantize_bits": None, "noise_std": 0.0},
            {"label": "loss_25pct", "packet_loss": 0.25, "quantize_bits": None, "noise_std": 0.0},
            # Quantization sweep (bandwidth constraint)
            {"label": "quant_16bit", "packet_loss": 0.0, "quantize_bits": 16, "noise_std": 0.0},
            {"label": "quant_8bit", "packet_loss": 0.0, "quantize_bits": 8, "noise_std": 0.0},
            # Channel noise sweep
            {"label": "noise_0.001", "packet_loss": 0.0, "quantize_bits": None, "noise_std": 0.001},
            {"label": "noise_0.005", "packet_loss": 0.0, "quantize_bits": None, "noise_std": 0.005},
            {"label": "noise_0.01", "packet_loss": 0.0, "quantize_bits": None, "noise_std": 0.01},
            # Combined worst-case
            {"label": "combined_moderate", "packet_loss": 0.05, "quantize_bits": 16, "noise_std": 0.001},
            {"label": "combined_severe", "packet_loss": 0.15, "quantize_bits": 8, "noise_std": 0.005},
        ]

    print(f"Running {len(configs)} impairment configurations")
    print(f"Rounds: {args.rounds}, Local epochs: {args.local_epochs}, LR: {args.lr}")
    print(f"Device: {device}")

    all_results = []
    for config in configs:
        result = run_one_experiment(config, args, device)
        all_results.append(result)

        # Save incrementally
        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # ---- Print summary table ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Config':<25} {'Best FT':>8} {'Best FZ':>8} {'Best PA':>8} {'Final PA':>8}")
    print("-" * 60)
    for r in all_results:
        bp = r["best_round"]
        fp = r["final_round"]
        print(f"{r['config']['label']:<25} "
              f"{bp.get('fault_type_acc',0):>8.3f} "
              f"{bp.get('fault_zone_acc',0):>8.3f} "
              f"{bp.get('protection_action_acc',0):>8.3f} "
              f"{fp.get('protection_action_acc',0):>8.3f}")

    print(f"\nAll results saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()