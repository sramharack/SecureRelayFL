"""
SecureRelayFL — Axis 2: Differential Privacy Sweep (v2)

Implements DPFedAvg: Gaussian mechanism applied to client updates before aggregation.
Sweeps privacy budget ε ∈ [0.5, 1, 2, 5, 10, ∞] with fixed δ = 1/N².

v2 changes:
    - Uses cnn_v2 model by default (passes config_features)

Usage:
    python -m securerelayfl.experiments.axis2_privacy --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

import flwr as fl
from flwr.common import ndarrays_to_parameters
from flwr.server.strategy import FedAvg

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from securerelayfl.fl.client import FaultClient, get_parameters
from securerelayfl.fl.server import get_model, get_evaluate_fn


# Privacy budgets to sweep (inf = no DP)
EPSILON_VALUES = [0.5, 1.0, 2.0, 5.0, 10.0, float("inf")]
DELTA_FACTOR = 1.0  # δ = DELTA_FACTOR / N² where N = total training samples


def compute_noise_multiplier(
    epsilon: float,
    delta: float,
    sensitivity: float = 1.0,
) -> float:
    """Compute Gaussian noise multiplier σ for (ε, δ)-DP.

    Uses the analytic Gaussian mechanism:
        σ = sensitivity * sqrt(2 * ln(1.25 / δ)) / ε
    """
    if epsilon == float("inf"):
        return 0.0
    import math
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def clip_and_noise_parameters(
    parameters: list[np.ndarray],
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Apply gradient clipping and Gaussian noise for DP.

    1. Clip: scale parameter update so L2 norm ≤ clip_norm
    2. Noise: add N(0, (clip_norm * noise_multiplier)²) per parameter
    """
    if noise_multiplier == 0.0:
        return parameters

    # Compute L2 norm across all parameter tensors
    total_norm = np.sqrt(sum(np.sum(p ** 2) for p in parameters))

    # Clip
    clip_factor = min(1.0, clip_norm / (total_norm + 1e-10))
    clipped = [p * clip_factor for p in parameters]

    # Add noise
    noise_std = clip_norm * noise_multiplier
    noised = [
        p + rng.normal(0, noise_std, p.shape).astype(p.dtype)
        for p in clipped
    ]
    return noised


class DPFedAvg(FedAvg):
    """FedAvg with client-level differential privacy (Gaussian mechanism)."""

    def __init__(
        self,
        noise_multiplier: float = 0.0,
        clip_norm: float = 1.0,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.noise_multiplier = noise_multiplier
        self.clip_norm = clip_norm
        self.rng = np.random.default_rng(seed)

    def aggregate_fit(self, server_round, results, failures):
        if not results:
            return None, {}

        # Apply DP to each client's parameters
        dp_results = []
        for client_proxy, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            dp_params = clip_and_noise_parameters(
                params, self.clip_norm, self.noise_multiplier, self.rng
            )
            fit_res.parameters = fl.common.ndarrays_to_parameters(dp_params)
            dp_results.append((client_proxy, fit_res))

        return super().aggregate_fit(server_round, dp_results, failures)


def run_dp_experiment(
    epsilon: float,
    model_name: str,
    data_dir: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    clip_norm: float,
    n_total_samples: int,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run a single DP configuration."""
    device = torch.device("cpu")
    n_clients = 5

    delta = DELTA_FACTOR / (n_total_samples ** 2)
    noise_mult = compute_noise_multiplier(epsilon, delta)

    tag = f"eps_{epsilon:.1f}" if epsilon != float("inf") else "eps_inf"
    exp_dir = output_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    init_model = get_model(model_name, device)
    init_params = ndarrays_to_parameters(get_parameters(init_model))

    strategy = DPFedAvg(
        noise_multiplier=noise_mult,
        clip_norm=clip_norm,
        seed=seed,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=n_clients,
        min_evaluate_clients=n_clients,
        min_available_clients=n_clients,
        initial_parameters=init_params,
        evaluate_fn=get_evaluate_fn(model_name, data_dir, device, exp_dir),
    )

    def _client_fn(context: fl.common.Context) -> fl.client.Client:
        partition_id = int(context.node_config.get("partition-id", 0))
        return FaultClient(
            facility_id=partition_id,
            model_name=model_name,
            data_dir=data_dir,
            local_epochs=local_epochs,
            batch_size=batch_size,
            lr=lr,
            fedprox_mu=0.0,
            device=device,
        ).to_client()

    t0 = time.time()
    fl.simulation.start_simulation(
        client_fn=_client_fn,
        num_clients=n_clients,
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        ray_init_args={"num_gpus": 0},
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
    )
    elapsed = time.time() - t0

    # Load final round metrics
    metrics_path = exp_dir / "round_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            round_metrics = json.load(f)
        final = round_metrics[-1] if round_metrics else {}
    else:
        final = {}

    result = {
        "epsilon": epsilon,
        "delta": delta,
        "noise_multiplier": noise_mult,
        "clip_norm": clip_norm,
        "elapsed_s": elapsed,
        **{k: v for k, v in final.items() if k != "round"},
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Axis 2: DP privacy sweep")
    parser.add_argument("--model", type=str, default="cnn_v2", choices=["cnn_v1", "cnn_v2"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--output-dir", type=str, default="results/axis2_privacy")
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Total training samples (for δ computation)")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Axis 2 — DP Sweep: {len(EPSILON_VALUES)} configurations")
    print(f"Model: {args.model} | Rounds: {args.rounds} | Clip: {args.clip_norm}\n")

    all_results = []
    for i, eps in enumerate(EPSILON_VALUES):
        eps_str = f"ε={eps}" if eps != float("inf") else "ε=∞ (no DP)"
        print(f"[{i+1}/{len(EPSILON_VALUES)}] {eps_str}")

        result = run_dp_experiment(
            epsilon=eps,
            model_name=args.model,
            data_dir=args.data_dir,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            clip_norm=args.clip_norm,
            n_total_samples=args.n_samples,
            seed=args.seed,
            output_dir=output_dir,
        )
        all_results.append(result)
        print(f"  → PA={result.get('acc_pa', 'N/A'):.3f}  σ={result['noise_multiplier']:.4f}\n")

    # Save consolidated results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()