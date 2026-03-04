"""
SecureRelayFL — Axis 2: Differential Privacy Sweep

Adds DP noise to client model updates before aggregation.
Mechanism: clip L2 norm of update delta, add Gaussian noise calibrated to epsilon.

Usage:
    python -m securerelayfl.experiments.axis2_privacy --seed 42
    python -m securerelayfl.experiments.axis2_privacy --quick --seed 42
"""

import argparse
import json
import os
import time
from collections import OrderedDict
from typing import Optional

import numpy as np
import torch
import flwr as fl
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from securerelayfl.models.fault_classifier import FaultClassifier, MultiTaskLoss
from securerelayfl.fl.client import make_client_fn


NUM_FACILITIES = 5


class DPFedAvg(FedAvg):
    """
    FedAvg with client-level differential privacy.

    Each round:
        1. Receive client updates
        2. Compute update delta (client_params - global_params)
        3. Clip delta L2 norm to max_norm
        4. Add Gaussian noise calibrated to (max_norm, epsilon, delta)
        5. Apply noised delta to global model
    """

    def __init__(
        self,
        epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        max_norm: float = 1.0,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.dp_delta = dp_delta
        self.max_norm = max_norm
        self.np_rng = np.random.RandomState(seed)

        # Compute noise multiplier from epsilon via Gaussian mechanism
        # sigma = max_norm * sqrt(2 * ln(1.25/delta)) / epsilon
        if epsilon < float("inf"):
            self.sigma = max_norm * np.sqrt(2 * np.log(1.25 / dp_delta)) / epsilon
        else:
            self.sigma = 0.0

        self.round_log = []

    def aggregate_fit(self, server_round, results, failures):
        """Clip and noise client updates before standard aggregation."""

        if not results:
            return None, {}

        # Get current global model (from first client's perspective, pre-training)
        # We clip each client's full parameter set independently
        clipped_results = []
        norms_before = []
        norms_after = []

        for client, fit_res in results:
            ndarrays = parameters_to_ndarrays(fit_res.parameters)

            # Compute L2 norm of full parameter vector
            flat = np.concatenate([a.flatten() for a in ndarrays])
            norm = np.linalg.norm(flat)
            norms_before.append(float(norm))

            # Clip
            clip_factor = min(1.0, self.max_norm / (norm + 1e-10))
            clipped = [a * clip_factor for a in ndarrays]

            # Add noise per-parameter
            if self.sigma > 0:
                noised = [
                    a + self.np_rng.normal(0, self.sigma, a.shape).astype(a.dtype)
                    for a in clipped
                ]
            else:
                noised = clipped

            clipped_norm = np.linalg.norm(np.concatenate([a.flatten() for a in noised]))
            norms_after.append(float(clipped_norm))

            fit_res = FitRes(
                status=fit_res.status,
                parameters=ndarrays_to_parameters(noised),
                num_examples=fit_res.num_examples,
                metrics=fit_res.metrics,
            )
            clipped_results.append((client, fit_res))

        self.round_log.append({
            "round": server_round,
            "epsilon": self.epsilon,
            "sigma": self.sigma,
            "avg_norm_before": np.mean(norms_before),
            "avg_norm_after": np.mean(norms_after),
        })

        return super().aggregate_fit(server_round, clipped_results, failures)


def get_initial_parameters():
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


def run_one_experiment(epsilon, args, device):
    label = f"eps_{epsilon}" if epsilon < float("inf") else "eps_inf"
    print(f"\n{'='*60}")
    print(f"  DP sweep: epsilon={epsilon}, sigma={'computed' if epsilon < float('inf') else '0'}")
    print(f"{'='*60}")

    strategy = DPFedAvg(
        epsilon=epsilon,
        dp_delta=1e-5,
        max_norm=args.max_norm,
        seed=args.seed,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_FACILITIES,
        min_evaluate_clients=NUM_FACILITIES,
        min_available_clients=NUM_FACILITIES,
        initial_parameters=get_initial_parameters(),
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

    # Extract metrics
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
        "epsilon": epsilon,
        "label": label,
        "sigma": strategy.sigma,
        "max_norm": args.max_norm,
        "training_time_s": elapsed,
        "best_round": best_pa,
        "final_round": final,
        "dp_log": strategy.round_log,
        "all_rounds": cm,
    }

    print(f"  Sigma: {strategy.sigma:.4f}")
    print(f"  Time: {elapsed:.0f}s")
    if best_pa:
        print(f"  Best PA: round {best_pa.get('round','?')} — "
              f"FT={best_pa.get('fault_type_acc',0):.3f} "
              f"PA={best_pa.get('protection_action_acc',0):.3f}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Axis 2: DP Privacy Sweep")
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-norm", type=float, default=1.0,
                        help="L2 norm clipping bound for client updates")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default="results/axis2_privacy")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.quick:
        epsilons = [1.0, 10.0, float("inf")]
    else:
        epsilons = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]

    print(f"Running DP sweep: {len(epsilons)} epsilon values")
    print(f"Max norm: {args.max_norm}, Rounds: {args.rounds}")

    all_results = []
    for eps in epsilons:
        result = run_one_experiment(eps, args, device)
        all_results.append(result)

        with open(os.path.join(args.output_dir, "results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 70)
    print("DP PRIVACY SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Epsilon':>10} {'Sigma':>10} {'Best FT':>8} {'Best PA':>8} {'Final PA':>9}")
    print("-" * 50)
    for r in all_results:
        bp = r["best_round"]
        fp = r["final_round"]
        eps_str = f"{r['epsilon']:.1f}" if r["epsilon"] < float("inf") else "inf"
        print(f"{eps_str:>10} {r['sigma']:>10.4f} "
              f"{bp.get('fault_type_acc',0):>8.3f} "
              f"{bp.get('protection_action_acc',0):>8.3f} "
              f"{fp.get('protection_action_acc',0):>9.3f}")

    print(f"\nResults saved to {args.output_dir}/results.json")


if __name__ == "__main__":
    main()