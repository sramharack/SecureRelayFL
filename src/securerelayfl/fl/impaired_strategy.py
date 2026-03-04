"""
SecureRelayFL — Network-Impaired FedAvg Strategy

Wraps FedAvg to simulate industrial communication network impairments:
    - Latency: delays per-client (simulated, tracked for analysis)
    - Packet loss: randomly drops client updates before aggregation
    - Bandwidth constraint: quantizes model updates to simulate limited bandwidth

These impairments act on the FL aggregation layer, not on the model itself.
The dual-layer analysis (GOOSE + FL traffic) builds on top of this.
"""

import random
from typing import Optional

import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


class ImpairedFedAvg(FedAvg):
    """
    FedAvg with configurable network impairments.

    Parameters
    ----------
    packet_loss_rate : float
        Probability [0, 1] that a client's update is dropped entirely.
        Simulates UDP-like unreliable transport or network congestion.
    quantize_bits : int or None
        If set, quantize model updates to this many bits (8, 16, or 32).
        Simulates bandwidth-constrained links. None = no quantization (float32).
    latency_ms : float
        Simulated one-way latency per client in milliseconds.
        Not enforced as real delay — tracked in metrics for analysis.
    latency_jitter_ms : float
        Random jitter added to latency (uniform ± jitter).
    noise_std : float
        Gaussian noise std added to parameters after aggregation.
        Simulates channel noise or lossy compression artifacts.
    seed : int
        Random seed for reproducibility of impairment patterns.
    **kwargs
        Passed through to FedAvg (fraction_fit, initial_parameters, etc.)
    """

    def __init__(
        self,
        packet_loss_rate: float = 0.0,
        quantize_bits: Optional[int] = None,
        latency_ms: float = 0.0,
        latency_jitter_ms: float = 0.0,
        noise_std: float = 0.0,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.packet_loss_rate = packet_loss_rate
        self.quantize_bits = quantize_bits
        self.latency_ms = latency_ms
        self.latency_jitter_ms = latency_jitter_ms
        self.noise_std = noise_std
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        # Track impairment stats per round
        self.impairment_log = []

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[tuple[ClientProxy, FitRes] | BaseException],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Apply impairments to client results before aggregation."""

        n_received = len(results)
        round_stats = {
            "round": server_round,
            "clients_received": n_received,
            "clients_dropped": 0,
            "quantize_bits": self.quantize_bits or 32,
            "latency_ms": [],
        }

        # ---- 1. Packet loss: drop client updates ----
        if self.packet_loss_rate > 0 and results:
            surviving = []
            for client, fit_res in results:
                if self.rng.random() > self.packet_loss_rate:
                    surviving.append((client, fit_res))
                else:
                    round_stats["clients_dropped"] += 1
            results = surviving

        # If all clients dropped, skip aggregation
        if not results:
            round_stats["clients_surviving"] = 0
            self.impairment_log.append(round_stats)
            return None, {"impairment_all_dropped": True}

        round_stats["clients_surviving"] = len(results)

        # ---- 2. Quantization: reduce precision of client updates ----
        if self.quantize_bits is not None and self.quantize_bits < 32:
            quantized_results = []
            for client, fit_res in results:
                ndarrays = parameters_to_ndarrays(fit_res.parameters)
                q_arrays = [self._quantize(arr) for arr in ndarrays]
                fit_res = FitRes(
                    status=fit_res.status,
                    parameters=ndarrays_to_parameters(q_arrays),
                    num_examples=fit_res.num_examples,
                    metrics=fit_res.metrics,
                )
                quantized_results.append((client, fit_res))
            results = quantized_results

        # ---- 3. Simulate latency (record, don't actually sleep) ----
        for _ in results:
            lat = self.latency_ms
            if self.latency_jitter_ms > 0:
                lat += self.rng.uniform(-self.latency_jitter_ms, self.latency_jitter_ms)
            round_stats["latency_ms"].append(max(0.0, lat))

        self.impairment_log.append(round_stats)

        # ---- 4. Standard FedAvg aggregation ----
        aggregated_params, metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # ---- 5. Post-aggregation channel noise ----
        if self.noise_std > 0 and aggregated_params is not None:
            ndarrays = parameters_to_ndarrays(aggregated_params)
            noisy = [
                arr + self.np_rng.normal(0, self.noise_std, arr.shape).astype(arr.dtype)
                for arr in ndarrays
            ]
            aggregated_params = ndarrays_to_parameters(noisy)

        return aggregated_params, metrics

    def _quantize(self, arr: np.ndarray) -> np.ndarray:
        """Uniform quantization to simulate bandwidth constraint."""
        if self.quantize_bits is None or self.quantize_bits >= 32:
            return arr

        # Dynamic range quantization
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min < 1e-10:
            return arr

        n_levels = 2 ** self.quantize_bits
        scale = (a_max - a_min) / (n_levels - 1)
        quantized = np.round((arr - a_min) / scale) * scale + a_min
        return quantized.astype(arr.dtype)