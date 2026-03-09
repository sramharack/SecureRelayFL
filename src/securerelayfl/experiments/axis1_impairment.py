"""
SecureRelayFL — Axis 1: Network Impairment Sweep (v2)

Runs FedAvg with ImpairedFedAvg strategy across a grid of:
    - Packet loss:   [0%, 5%, 10%, 15%, 25%]
    - Quantization:  [32-bit, 16-bit, 8-bit]
    - Gaussian noise: [0, 0.001, 0.01]

v2 changes:
    - Uses cnn_v2 model by default (passes config_features)
    - Impairment→scenario mapping for paper:
        5%  loss → normal industrial EMI
        10% loss → fault event network congestion
        15% loss → cascading fault with switch buffer overflow
        25% loss → partial network failure
        8-bit quant → bandwidth-constrained WAN (1 Mbps satellite)
        16-bit quant → shared corporate WAN (10 Mbps allocated)

Usage:
    python -m securerelayfl.experiments.axis1_impairment --seed 42
"""

from __future__ import annotations

import argparse
import itertools
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


# Impairment grid
PACKET_LOSS_RATES = [0.0, 0.05, 0.10, 0.15, 0.25]
QUANT_BITS = [32, 16, 8]
NOISE_SCALES = [0.0, 0.001, 0.01]

# Scenario labels for paper
SCENARIO_LABELS = {
    0.0:  "Ideal (no loss)",
    0.05: "Normal industrial EMI",
    0.10: "Fault event congestion",
    0.15: "Cascading fault / buffer overflow",
    0.25: "Partial network failure",
}


def impair_parameters(
    parameters: list[np.ndarray],
    packet_loss: float,
    quant_bits: int,
    noise_scale: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Apply network impairments to model parameter arrays.

    Simulates:
        - Packet loss: randomly zero out fraction of parameter tensors
        - Quantization: reduce precision to quant_bits
        - Additive Gaussian noise: communication channel noise
    """
    impaired = []
    for arr in parameters:
        a = arr.copy()

        # Packet loss — simulate dropped gradient chunks
        if packet_loss > 0:
            mask = rng.random(a.shape) > packet_loss
            a = a * mask

        # Quantization
        if quant_bits < 32:
            a_min, a_max = a.min(), a.max()
            if a_max - a_min > 1e-10:
                levels = 2 ** quant_bits - 1
                a_norm = (a - a_min) / (a_max - a_min)
                a_quant = np.round(a_norm * levels) / levels
                a = a_quant * (a_max - a_min) + a_min

        # Gaussian noise
        if noise_scale > 0:
            a = a + rng.normal(0, noise_scale, a.shape).astype(a.dtype)

        impaired.append(a)
    return impaired


class ImpairedFedAvg(FedAvg):
    """FedAvg with configurable network impairments applied to client updates."""

    def __init__(
        self,
        packet_loss: float = 0.0,
        quant_bits: int = 32,
        noise_scale: float = 0.0,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.packet_loss = packet_loss
        self.quant_bits = quant_bits
        self.noise_scale = noise_scale
        self.rng = np.random.default_rng(seed)

    def aggregate_fit(self, server_round, results, failures):
        """Override to impair client parameters before aggregation."""
        if not results:
            return None, {}

        # Apply impairments to each client's parameters
        impaired_results = []
        for client_proxy, fit_res in results:
            params = fl.common.parameters_to_ndarrays(fit_res.parameters)
            impaired_params = impair_parameters(
                params, self.packet_loss, self.quant_bits, self.noise_scale, self.rng
            )
            fit_res.parameters = fl.common.ndarrays_to_parameters(impaired_params)
            impaired_results.append((client_proxy, fit_res))

        return super().aggregate_fit(server_round, impaired_results, failures)


def run_impairment_experiment(
    packet_loss: float,
    quant_bits: int,
    noise_scale: float,
    model_name: str,
    data_dir: str,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run a single impairment configuration."""
    device = torch.device("cpu")
    n_clients = 5

    tag = f"pl{packet_loss:.2f}_q{quant_bits}_n{noise_scale:.4f}"
    exp_dir = output_dir / tag
    exp_dir.mkdir(parents=True, exist_ok=True)

    init_model = get_model(model_name, device)
    init_params = ndarrays_to_parameters(get_parameters(init_model))

    strategy = ImpairedFedAvg(
        packet_loss=packet_loss,
        quant_bits=quant_bits,
        noise_scale=noise_scale,
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
        "packet_loss": packet_loss,
        "quant_bits": quant_bits,
        "noise_scale": noise_scale,
        "scenario": SCENARIO_LABELS.get(packet_loss, ""),
        "elapsed_s": elapsed,
        **{k: v for k, v in final.items() if k != "round"},
    }

    with open(exp_dir / "summary.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    parser = argparse.ArgumentParser(description="Axis 1: Impairment sweep")
    parser.add_argument("--model", type=str, default="cnn_v2", choices=["cnn_v1", "cnn_v2"])
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--output-dir", type=str, default="results/axis1_impairment")
    parser.add_argument("--quick", action="store_true",
                        help="Run reduced grid for quick testing")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Grid
    if args.quick:
        grid = [(0.0, 32, 0.0), (0.10, 32, 0.0), (0.0, 8, 0.0)]
    else:
        # Full grid: packet loss × quant_bits (noise fixed at 0 for main sweep)
        # Plus noise sub-sweep at default settings
        grid = []
        for pl in PACKET_LOSS_RATES:
            for qb in QUANT_BITS:
                grid.append((pl, qb, 0.0))
        # Noise sub-sweep (ideal loss, 32-bit)
        for ns in NOISE_SCALES:
            if ns > 0:
                grid.append((0.0, 32, ns))

    print(f"Axis 1 — Impairment Sweep: {len(grid)} configurations")
    print(f"Model: {args.model} | Rounds: {args.rounds}\n")

    all_results = []
    for i, (pl, qb, ns) in enumerate(grid):
        scenario = SCENARIO_LABELS.get(pl, "")
        print(f"[{i+1}/{len(grid)}] PL={pl:.0%} Q={qb}bit N={ns:.4f} — {scenario}")

        result = run_impairment_experiment(
            packet_loss=pl,
            quant_bits=qb,
            noise_scale=ns,
            model_name=args.model,
            data_dir=args.data_dir,
            rounds=args.rounds,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            output_dir=output_dir,
        )
        all_results.append(result)
        print(f"  → PA={result.get('acc_pa', 'N/A'):.3f}  FT={result.get('acc_ft', 'N/A'):.3f}\n")

    # Save consolidated results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()