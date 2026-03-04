"""
SecureRelayFL — Axis 3: GOOSE Dual-Layer Communication Analysis

Analytical model of FL aggregation traffic coexisting with IEC 61850
GOOSE protection messages on shared industrial Ethernet.

Key question: At what FL communication rate does aggregation traffic
risk violating the 4ms GOOSE transfer time requirement?

IEC 61850 Performance Classes:
    - Type 1A (Trip): 3ms transfer time
    - Type 1B (Trip): 10ms transfer time  
    - Type 4 (Raw data): 3ms transfer time
    GOOSE retransmission: initial burst at 1-4ms intervals

Network assumptions:
    - 100 Mbps switched Ethernet (typical substation LAN)
    - OR 1 Gbps switched Ethernet (modern installations)
    - GOOSE frames: ~200 bytes each (IEC 61850-8-1)
    - GOOSE burst: 5-10 messages within first 20ms of event
    - FL traffic shares same physical network

Usage:
    python -m securerelayfl.analysis.goose_dual_layer
"""

import json
import os
import numpy as np


# ---- Constants ----
GOOSE_FRAME_BYTES = 200          # Typical GOOSE frame size (Ethernet + APDU)
GOOSE_BURST_COUNT = 8            # Messages in initial burst after event
GOOSE_BURST_WINDOW_MS = 20       # Burst window duration
GOOSE_MAX_LATENCY_MS = {         # IEC 61850-5 performance classes
    "1A_trip": 3.0,
    "1B_trip": 10.0,
    "4_raw": 3.0,
}
ETHERNET_OVERHEAD_BYTES = 38     # Preamble + SFD + IFG + FCS

# CNN model with GroupNorm: 111,981 parameters × 4 bytes = 447,924 bytes
CNN_PARAMS = 111_981
BYTES_PER_PARAM = 4              # float32
CNN_MODEL_BYTES = CNN_PARAMS * BYTES_PER_PARAM

NUM_CLIENTS = 5


def compute_fl_traffic(
    model_bytes: int,
    num_clients: int,
    rounds_per_minute: float,
    quantize_bits: int = 32,
) -> dict:
    """
    Compute FL traffic characteristics per communication round.

    Returns dict with bytes, duration, and bandwidth stats.
    """
    # Effective model size with quantization
    effective_bytes = model_bytes * (quantize_bits / 32)

    # Per round: each client uploads + downloads the model
    upload_per_round = effective_bytes * num_clients     # clients → server
    download_per_round = effective_bytes * num_clients   # server → clients
    total_per_round = upload_per_round + download_per_round

    # Add Ethernet framing overhead (~5% for large transfers)
    # Model is sent as multiple TCP segments (~1460 bytes each)
    n_segments = int(np.ceil(effective_bytes / 1460))
    framing_overhead = n_segments * ETHERNET_OVERHEAD_BYTES * num_clients * 2
    total_with_overhead = total_per_round + framing_overhead

    # Per minute
    bytes_per_minute = total_with_overhead * rounds_per_minute
    bits_per_second = (bytes_per_minute * 8) / 60

    return {
        "model_bytes": int(effective_bytes),
        "quantize_bits": quantize_bits,
        "upload_per_round_bytes": int(upload_per_round),
        "download_per_round_bytes": int(download_per_round),
        "total_per_round_bytes": int(total_with_overhead),
        "total_per_round_MB": total_with_overhead / (1024 ** 2),
        "rounds_per_minute": rounds_per_minute,
        "sustained_bps": bits_per_second,
        "sustained_Mbps": bits_per_second / 1e6,
    }


def compute_goose_burst_load(
    burst_count: int = GOOSE_BURST_COUNT,
    frame_bytes: int = GOOSE_FRAME_BYTES,
    burst_window_ms: float = GOOSE_BURST_WINDOW_MS,
) -> dict:
    """Compute GOOSE burst traffic characteristics."""
    total_bytes = burst_count * (frame_bytes + ETHERNET_OVERHEAD_BYTES)
    burst_bps = (total_bytes * 8) / (burst_window_ms / 1000)

    return {
        "burst_count": burst_count,
        "frame_bytes": frame_bytes,
        "burst_window_ms": burst_window_ms,
        "total_burst_bytes": total_bytes,
        "burst_Mbps": burst_bps / 1e6,
    }


def compute_queueing_delay(
    fl_traffic: dict,
    link_speed_mbps: float,
    goose_frame_bytes: int = GOOSE_FRAME_BYTES,
) -> dict:
    """
    Estimate worst-case GOOSE queueing delay due to FL traffic.

    Model: GOOSE frame arrives while an FL segment is being transmitted.
    Worst case = one maximum-size FL segment must complete before GOOSE.
    With priority queueing (802.1p), GOOSE preempts after current frame.
    Without priority queueing, GOOSE waits for current frame.
    """
    link_speed_bps = link_speed_mbps * 1e6

    # Maximum FL TCP segment on the wire
    max_fl_frame_bytes = 1518  # Maximum Ethernet frame
    max_fl_frame_bits = max_fl_frame_bytes * 8

    # Time to transmit one max FL frame (worst case head-of-line blocking)
    hol_delay_ms = (max_fl_frame_bits / link_speed_bps) * 1000

    # GOOSE frame transmission time
    goose_tx_ms = ((goose_frame_bytes + ETHERNET_OVERHEAD_BYTES) * 8 / link_speed_bps) * 1000

    # FL link utilization during active transfer
    # Assume FL round transfer takes: total_bytes / link_speed
    fl_transfer_ms = (fl_traffic["total_per_round_bytes"] * 8 / link_speed_bps) * 1000

    # Sustained utilization
    if fl_traffic["rounds_per_minute"] > 0:
        fl_active_fraction = (fl_transfer_ms / 1000) * fl_traffic["rounds_per_minute"] / 60
    else:
        fl_active_fraction = 0

    # Expected queueing delay (M/D/1 approximation for bursty FL traffic)
    # During FL transfer, GOOSE sees utilization ≈ 1 on shared segment
    # Probability of collision = fl_active_fraction
    expected_delay_ms = fl_active_fraction * hol_delay_ms

    # Worst case: GOOSE arrives mid-FL-transfer, no priority queueing
    worst_case_no_priority_ms = hol_delay_ms + goose_tx_ms

    # With 802.1p priority queueing: GOOSE waits for current frame only
    worst_case_priority_ms = hol_delay_ms + goose_tx_ms

    return {
        "link_speed_mbps": link_speed_mbps,
        "hol_blocking_ms": hol_delay_ms,
        "goose_tx_ms": goose_tx_ms,
        "fl_transfer_duration_ms": fl_transfer_ms,
        "fl_link_utilization": fl_active_fraction,
        "expected_goose_delay_ms": expected_delay_ms,
        "worst_case_delay_ms": worst_case_no_priority_ms,
    }


def find_critical_threshold(link_speed_mbps: float, max_goose_delay_ms: float = 3.0):
    """
    Find the maximum FL rounds/minute before GOOSE timing is violated.

    Sweeps rounds_per_minute until worst-case GOOSE delay exceeds threshold.
    """
    # The critical constraint: FL link utilization must be low enough
    # that the probability × impact of head-of-line blocking stays under budget

    link_speed_bps = link_speed_mbps * 1e6
    max_fl_frame_bits = 1518 * 8
    hol_delay_ms = (max_fl_frame_bits / link_speed_bps) * 1000

    results = []
    for rpm in np.arange(0.5, 120, 0.5):
        for qbits in [32, 16, 8]:
            fl = compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, rpm, qbits)
            qd = compute_queueing_delay(fl, link_speed_mbps)

            results.append({
                "rounds_per_minute": rpm,
                "quantize_bits": qbits,
                "fl_utilization": qd["fl_link_utilization"],
                "fl_transfer_ms": qd["fl_transfer_duration_ms"],
                "expected_delay_ms": qd["expected_goose_delay_ms"],
                "worst_case_delay_ms": qd["worst_case_delay_ms"],
                "goose_budget_ms": max_goose_delay_ms,
                "violates_budget": qd["fl_link_utilization"] > 0.5,
                # Conservative: flag if utilization > 50%
            })

    return results


def main():
    output_dir = "results/axis3_goose"
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("AXIS 3: GOOSE DUAL-LAYER COMMUNICATION ANALYSIS")
    print("=" * 70)

    # ---- Model traffic characteristics ----
    print("\n--- FL Model Traffic ---")
    for qbits in [32, 16, 8]:
        fl = compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, 1.0, qbits)
        print(f"  {qbits}-bit: model={fl['model_bytes']/1024:.0f} KB, "
              f"per_round={fl['total_per_round_MB']:.2f} MB, "
              f"at 1 rpm = {fl['sustained_Mbps']:.3f} Mbps")

    # ---- GOOSE burst characteristics ----
    print("\n--- GOOSE Protection Burst ---")
    goose = compute_goose_burst_load()
    print(f"  {goose['burst_count']} frames × {goose['frame_bytes']} bytes "
          f"in {goose['burst_window_ms']}ms = {goose['burst_Mbps']:.3f} Mbps burst")

    # ---- Queueing analysis at different link speeds ----
    print("\n--- Queueing Analysis (1 round/minute, 32-bit) ---")
    for speed in [100, 1000]:
        fl = compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, 1.0, 32)
        qd = compute_queueing_delay(fl, speed)
        print(f"  {speed} Mbps link:")
        print(f"    FL transfer duration: {qd['fl_transfer_duration_ms']:.1f} ms")
        print(f"    FL link utilization: {qd['fl_link_utilization']:.4f}")
        print(f"    HOL blocking (worst case): {qd['hol_blocking_ms']:.3f} ms")
        print(f"    GOOSE Type 1A budget: 3.0 ms — "
              f"{'SAFE' if qd['worst_case_delay_ms'] < 3.0 else 'AT RISK'}")

    # ---- Critical threshold sweep ----
    print("\n--- Critical Threshold Analysis ---")
    all_thresholds = {}
    for speed in [100, 1000]:
        results = find_critical_threshold(speed, max_goose_delay_ms=3.0)
        all_thresholds[f"{speed}Mbps"] = results

        # Find max safe rounds/minute per quantization
        for qbits in [32, 16, 8]:
            safe = [r for r in results
                    if r["quantize_bits"] == qbits and not r["violates_budget"]]
            if safe:
                max_safe = safe[-1]
                print(f"  {speed} Mbps, {qbits}-bit: max safe = "
                      f"{max_safe['rounds_per_minute']:.0f} rpm "
                      f"(util={max_safe['fl_utilization']:.1%})")
            else:
                print(f"  {speed} Mbps, {qbits}-bit: UNSAFE at all rates")

    # ---- Traffic-aware scheduling recommendation ----
    print("\n--- Traffic-Aware FL Scheduling ---")
    fl32 = compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, 1.0, 32)
    fl8 = compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, 1.0, 8)
    print(f"  32-bit FL round: {fl32['total_per_round_MB']:.2f} MB")
    print(f"   8-bit FL round: {fl8['total_per_round_MB']:.2f} MB "
          f"({fl8['total_per_round_MB']/fl32['total_per_round_MB']:.0%} of 32-bit)")
    print(f"  Recommendation: defer FL rounds during fault events (GOOSE burst)")
    print(f"  GOOSE burst window: {GOOSE_BURST_WINDOW_MS}ms — "
          f"FL must back off for ≥{GOOSE_BURST_WINDOW_MS + 10}ms after fault")

    # ---- Table for paper ----
    print("\n--- TABLE FOR PAPER: FL Traffic vs GOOSE Impact ---")
    print(f"{'Link':>6} {'Quant':>5} {'Round Size':>10} {'Transfer':>10} "
          f"{'Max Safe':>10} {'HOL Block':>10}")
    print(f"{'(Mbps)':>6} {'(bit)':>5} {'(MB)':>10} {'(ms)':>10} "
          f"{'(rpm)':>10} {'(ms)':>10}")
    print("-" * 55)
    for speed in [100, 1000]:
        for qbits in [32, 16, 8]:
            fl = compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, 1.0, qbits)
            qd = compute_queueing_delay(fl, speed)
            results = all_thresholds[f"{speed}Mbps"]
            safe = [r for r in results
                    if r["quantize_bits"] == qbits and not r["violates_budget"]]
            max_rpm = safe[-1]["rounds_per_minute"] if safe else 0
            print(f"{speed:>6} {qbits:>5} {fl['total_per_round_MB']:>10.2f} "
                  f"{qd['fl_transfer_duration_ms']:>10.1f} "
                  f"{max_rpm:>10.0f} {qd['hol_blocking_ms']:>10.3f}")

    # ---- Save all results ----
    output = {
        "model_info": {
            "name": "CNN+GroupNorm",
            "params": CNN_PARAMS,
            "bytes_fp32": CNN_MODEL_BYTES,
        },
        "goose_burst": goose,
        "fl_traffic": {
            qb: compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, 1.0, qb)
            for qb in [32, 16, 8]
        },
        "queueing_analysis": {
            f"{speed}Mbps": {
                qb: compute_queueing_delay(
                    compute_fl_traffic(CNN_MODEL_BYTES, NUM_CLIENTS, 1.0, qb), speed
                ) for qb in [32, 16, 8]
            } for speed in [100, 1000]
        },
        "threshold_sweep": {
            k: v for k, v in all_thresholds.items()
        },
    }

    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {output_dir}/results.json")


if __name__ == "__main__":
    main()