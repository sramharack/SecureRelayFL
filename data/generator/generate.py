#!/usr/bin/env python3
"""
SecureRelayFL — Synthetic Fault Waveform Generator (v2)

v2 changes:
    - Each facility now has 2-3 operating configurations (tie open/closed, DG on/off, etc.)
    - Configuration sampled per waveform based on probability weights
    - Protection action assignment is now configuration-aware
    - Outputs config_id.npy (N,) int64 and config_features.npy (N,3) float32

Physics references:
    - Asymmetrical fault current:         IEEE 551-2006, IEC 60909
    - Grounding-dependent sag/swell:      IEEE C62.92
    - High-impedance arcing faults:       IEEE PSRC WG D15

Usage:
    python data/generator/generate.py --n-samples 1000 --seed 42
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

# ──────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────
FREQ_HZ = 60
SAMPLE_RATE = 15_360          # 256 samples / cycle
CYCLES_PRE = 3
CYCLES_POST = 7
TOTAL_CYCLES = CYCLES_PRE + CYCLES_POST
N_SAMPLES_PER_WAVEFORM = TOTAL_CYCLES * (SAMPLE_RATE // FREQ_HZ)  # 2560
N_CHANNELS = 6                # Va Vb Vc Ia Ib Ic
FAULT_INCEPTION_SAMPLE = CYCLES_PRE * (SAMPLE_RATE // FREQ_HZ)    # 768

FAULT_TYPES = {0: "no_fault", 1: "SLG", 2: "LL", 3: "HIF"}
FAULT_ZONES = {0: "bus", 1: "near", 2: "far", 3: "none"}
PROTECTION_ACTIONS = {
    0: "no_action",
    1: "trip_instantaneous",
    2: "trip_delayed",
    3: "ZSI_block",
    4: "alarm_only",
}

# ──────────────────────────────────────────────────────────────────
# Facility Profiles (v2 — with operating configurations)
# ──────────────────────────────────────────────────────────────────
FACILITY_PROFILES = {
    0: {
        "name": "Data Center",
        "voltage_kv": 13.8,
        "grounding": "solidly_grounded",
        "snr_db": 40,
        "fault_type_weights": [0.10, 0.35, 0.35, 0.20],
        "fault_zone_weights": [0.25, 0.40, 0.35],
        "configs": [
            {   # Config 0: Normal — both mains closed, tie closed
                "id": 0, "name": "both_mains_tie_closed",
                "fault_mva": 250, "xr_ratio": 8,
                "source_count": 2,
                "probability": 0.5,
            },
            {   # Config 1: One main open, tie open (maintenance)
                "id": 1, "name": "single_main_tie_open",
                "fault_mva": 125, "xr_ratio": 6,
                "source_count": 1,
                "probability": 0.3,
            },
            {   # Config 2: Both mains, tie open, DG online
                "id": 2, "name": "both_mains_dg_online",
                "fault_mva": 300, "xr_ratio": 10,
                "source_count": 3,
                "probability": 0.2,
            },
        ],
    },
    1: {
        "name": "Steel Plant",
        "voltage_kv": 34.5,
        "grounding": "low_r_grounded",
        "snr_db": 25,
        "fault_type_weights": [0.08, 0.30, 0.42, 0.20],
        "fault_zone_weights": [0.20, 0.45, 0.35],
        "configs": [
            {"id": 0, "name": "arc_furnace_on", "fault_mva": 500, "xr_ratio": 15,
             "source_count": 2, "probability": 0.4},
            {"id": 1, "name": "arc_furnace_off", "fault_mva": 350, "xr_ratio": 12,
             "source_count": 1, "probability": 0.3},
            {"id": 2, "name": "bus_coupler_open", "fault_mva": 250, "xr_ratio": 10,
             "source_count": 1, "probability": 0.3},
        ],
    },
    2: {
        "name": "Petrochemical",
        "voltage_kv": 13.8,
        "grounding": "high_r_grounded",
        "snr_db": 35,
        "fault_type_weights": [0.10, 0.25, 0.30, 0.35],
        "fault_zone_weights": [0.20, 0.35, 0.45],
        "configs": [
            {"id": 0, "name": "normal_dual_incomer", "fault_mva": 150, "xr_ratio": 5,
             "source_count": 2, "probability": 0.5},
            {"id": 1, "name": "single_incomer_transfer", "fault_mva": 80, "xr_ratio": 4,
             "source_count": 1, "probability": 0.3},
            {"id": 2, "name": "emergency_gen_island", "fault_mva": 60, "xr_ratio": 3,
             "source_count": 1, "probability": 0.2},
        ],
    },
    3: {
        "name": "Pharmaceutical",
        "voltage_kv": 4.16,
        "grounding": "resistance_grounded",
        "snr_db": 38,
        "fault_type_weights": [0.12, 0.35, 0.33, 0.20],
        "fault_zone_weights": [0.25, 0.40, 0.35],
        "configs": [
            {"id": 0, "name": "normal_ups_online", "fault_mva": 200, "xr_ratio": 10,
             "source_count": 2, "probability": 0.5},
            {"id": 1, "name": "ups_bypass", "fault_mva": 220, "xr_ratio": 8,
             "source_count": 2, "probability": 0.3},
            {"id": 2, "name": "single_source_maintenance", "fault_mva": 100, "xr_ratio": 7,
             "source_count": 1, "probability": 0.2},
        ],
    },
    4: {
        "name": "Cement Plant",
        "voltage_kv": 34.5,
        "grounding": "low_r_grounded",
        "snr_db": 28,
        "fault_type_weights": [0.10, 0.30, 0.40, 0.20],
        "fault_zone_weights": [0.25, 0.40, 0.35],
        "configs": [
            {"id": 0, "name": "full_production", "fault_mva": 400, "xr_ratio": 12,
             "source_count": 2, "probability": 0.4},
            {"id": 1, "name": "kiln_only", "fault_mva": 300, "xr_ratio": 10,
             "source_count": 2, "probability": 0.3},
            {"id": 2, "name": "single_transformer", "fault_mva": 200, "xr_ratio": 8,
             "source_count": 1, "probability": 0.3},
        ],
    },
}

# ──────────────────────────────────────────────────────────────────
# Waveform Physics
# ──────────────────────────────────────────────────────────────────

def _time_array() -> np.ndarray:
    """Return time vector for one waveform window."""
    return np.arange(N_SAMPLES_PER_WAVEFORM) / SAMPLE_RATE


def _three_phase_prefault(
    t: np.ndarray,
    v_peak: float,
    i_peak: float,
    phi_v: float = 0.0,
    phi_i_lag: float = 0.52,   # ~30° load angle
) -> np.ndarray:
    """Generate balanced 3-phase pre-fault V & I.  Shape: (6, len(t))."""
    omega = 2 * math.pi * FREQ_HZ
    phases = [0.0, -2 * math.pi / 3, 2 * math.pi / 3]
    wf = np.zeros((6, len(t)), dtype=np.float32)
    for k, ph in enumerate(phases):
        wf[k] = v_peak * np.sin(omega * t + phi_v + ph)
        wf[k + 3] = i_peak * np.sin(omega * t + phi_v + ph - phi_i_lag)
    return wf


def _apply_fault_slg(
    wf: np.ndarray,
    t: np.ndarray,
    inception: int,
    fault_mva: float,
    xr_ratio: float,
    v_peak: float,
    grounding: str,
) -> np.ndarray:
    """Single-line-to-ground fault on phase A (IEEE 551 / IEC 60909)."""
    omega = 2 * math.pi * FREQ_HZ
    z_mag = (v_peak / 1e3) / (fault_mva + 1e-6) * 1e3  # simplified impedance
    tau_dc = xr_ratio / (2 * math.pi * FREQ_HZ)         # DC offset time constant

    i_fault_peak = v_peak / (z_mag + 1e-6)
    t_fault = t[inception:] - t[inception]

    # Asymmetric fault current on phase A  (IEEE 551-2006 eq. 1)
    i_ac = i_fault_peak * np.sin(omega * t_fault)
    i_dc = i_fault_peak * np.exp(-t_fault / (tau_dc + 1e-9))
    wf[3, inception:] = i_ac + i_dc   # Ia

    # Voltage sag on faulted phase (IEEE C62.92)
    sag = 0.15 if grounding == "solidly_grounded" else 0.35
    if grounding == "high_r_grounded":
        sag = 0.70  # limited fault current → less sag
    wf[0, inception:] *= sag

    # Swell on unfaulted phases
    swell = 1.0 + (1.0 - sag) * 0.3
    wf[1, inception:] *= swell
    wf[2, inception:] *= swell

    return wf


def _apply_fault_ll(
    wf: np.ndarray,
    t: np.ndarray,
    inception: int,
    fault_mva: float,
    xr_ratio: float,
    v_peak: float,
) -> np.ndarray:
    """Line-to-line fault between phases A and B."""
    omega = 2 * math.pi * FREQ_HZ
    z_mag = (v_peak / 1e3) / (fault_mva + 1e-6) * 1e3
    tau_dc = xr_ratio / (2 * math.pi * FREQ_HZ)

    i_fault_peak = (v_peak * math.sqrt(3)) / (2 * z_mag + 1e-6)
    t_fault = t[inception:] - t[inception]

    i_ac = i_fault_peak * np.sin(omega * t_fault + math.pi / 6)
    i_dc = i_fault_peak * 0.8 * np.exp(-t_fault / (tau_dc + 1e-9))

    wf[3, inception:] = i_ac + i_dc       # Ia
    wf[4, inception:] = -(i_ac + i_dc)    # Ib = -Ia for LL fault

    # Voltage depression on both faulted phases
    wf[0, inception:] *= 0.5
    wf[1, inception:] *= 0.5

    return wf


def _apply_fault_hif(
    wf: np.ndarray,
    t: np.ndarray,
    inception: int,
    v_peak: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """High-impedance arcing fault (IEEE PSRC WG D15 model)."""
    omega = 2 * math.pi * FREQ_HZ
    t_fault = t[inception:] - t[inception]

    # Low-magnitude arcing current with randomized re-ignition
    i_arc_peak = v_peak * 0.005 * (1 + 0.3 * rng.standard_normal())
    i_arc = i_arc_peak * np.sin(omega * t_fault)

    # Arc re-ignition noise bursts (PSRC D15 characteristic)
    for _ in range(rng.integers(3, 8)):
        burst_start = rng.integers(0, max(len(t_fault) - 50, 1))
        burst_len = rng.integers(20, 60)
        burst_end = min(burst_start + burst_len, len(t_fault))
        i_arc[burst_start:burst_end] += i_arc_peak * 0.5 * rng.standard_normal(
            burst_end - burst_start
        )

    # Mild voltage distortion
    wf[0, inception:] *= 0.97 + 0.03 * rng.standard_normal(len(t_fault))
    wf[3, inception:] += i_arc

    return wf


def _apply_zone_attenuation(wf: np.ndarray, inception: int, zone: int) -> np.ndarray:
    """Apply impedance attenuation based on fault zone (distance from relay)."""
    # zone 0=bus (closest), 1=near, 2=far
    atten = {0: 1.0, 1: 0.65, 2: 0.35}
    factor = atten.get(zone, 1.0)
    # Attenuate post-fault current channels only
    for ch in [3, 4, 5]:
        delta = wf[ch, inception:] - wf[ch, inception - 1]
        wf[ch, inception:] = wf[ch, inception - 1] + delta * factor
    return wf


def _add_noise(wf: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise at the specified SNR."""
    sig_power = np.mean(wf ** 2, axis=1, keepdims=True) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = rng.standard_normal(wf.shape).astype(np.float32) * np.sqrt(noise_power)
    return wf + noise


# ──────────────────────────────────────────────────────────────────
# Configuration-Aware Protection Action (v2)
# ──────────────────────────────────────────────────────────────────

def _assign_protection_action(
    fault_type: int, fault_zone: int, config: dict, grounding: str
) -> int:
    """Configuration-aware protection action assignment.

    The key v2 insight: with multiple source contributions, coordination
    (ZSI blocking, delayed trip) is needed. With single-source configs,
    simpler instantaneous tripping suffices.
    """
    source_count = config["source_count"]

    if fault_type == 0:   # no fault
        return 0          # no_action

    if fault_type == 3:   # HIF — arcing, low magnitude
        # HRG systems can't reliably trip on ground fault current
        return 4          # alarm_only

    # SLG (1) and LL (2) faults
    if fault_zone == 0:   # bus fault
        if source_count >= 2:
            return 3      # ZSI_block (coordinate multiple sources)
        else:
            return 1      # trip_instantaneous (single source)

    elif fault_zone == 1: # near fault
        if source_count >= 2:
            return 2      # trip_delayed (coordinated downstream)
        else:
            return 1      # trip_instantaneous

    elif fault_zone == 2: # far fault
        return 2          # trip_delayed (wait for downstream to clear)

    return 0  # fallback


# ──────────────────────────────────────────────────────────────────
# Per-Sample Generation
# ──────────────────────────────────────────────────────────────────

def _generate_sample(
    profile: dict, rng: np.random.Generator
) -> tuple[np.ndarray, int, int, int, int, np.ndarray]:
    """Generate one fault waveform sample.

    Returns:
        waveform   (6, 2560) float32
        fault_type  int
        fault_zone  int
        prot_action int
        config_id   int
        config_feat (3,) float32  — [fault_mva_norm, xr_ratio_norm, source_count_norm]
    """
    t = _time_array()
    v_kv = profile["voltage_kv"]
    v_peak = v_kv * 1e3 * math.sqrt(2) / math.sqrt(3)  # phase peak voltage
    grounding = profile["grounding"]

    # --- Sample operating configuration ---
    configs = profile["configs"]
    probs = np.array([c["probability"] for c in configs], dtype=np.float64)
    probs /= probs.sum()  # safety normalize
    cfg_idx = rng.choice(len(configs), p=probs)
    cfg = configs[cfg_idx]

    fault_mva = cfg["fault_mva"]
    xr_ratio = cfg["xr_ratio"]

    # Nominal load current (proportional to fault MVA for realism)
    i_peak = (fault_mva / (v_kv * math.sqrt(3))) * 1e3 * 0.05  # ~5% of fault current

    # --- Sample fault type ---
    ft_weights = np.array(profile["fault_type_weights"], dtype=np.float64)
    ft_weights /= ft_weights.sum()
    fault_type = int(rng.choice(4, p=ft_weights))

    # --- Sample fault zone (only for actual faults) ---
    if fault_type == 0:
        fault_zone = 3   # none
    else:
        fz_weights = np.array(profile["fault_zone_weights"], dtype=np.float64)
        fz_weights /= fz_weights.sum()
        fault_zone = int(rng.choice(3, p=fz_weights))

    # --- Generate pre-fault waveform ---
    phi_v = rng.uniform(0, 2 * math.pi)
    phi_i_lag = rng.uniform(0.35, 0.65)   # 20°–37° power factor angle
    wf = _three_phase_prefault(t, v_peak, i_peak, phi_v, phi_i_lag)

    # --- Apply fault ---
    inception = FAULT_INCEPTION_SAMPLE + rng.integers(-10, 10)
    inception = max(10, min(inception, N_SAMPLES_PER_WAVEFORM - 100))

    if fault_type == 1:    # SLG
        wf = _apply_fault_slg(wf, t, inception, fault_mva, xr_ratio, v_peak, grounding)
    elif fault_type == 2:  # LL
        wf = _apply_fault_ll(wf, t, inception, fault_mva, xr_ratio, v_peak)
    elif fault_type == 3:  # HIF
        wf = _apply_fault_hif(wf, t, inception, v_peak, rng)

    # --- Zone attenuation ---
    if fault_type != 0 and fault_zone < 3:
        wf = _apply_zone_attenuation(wf, inception, fault_zone)

    # --- Add noise ---
    wf = _add_noise(wf, profile["snr_db"], rng)

    # --- Protection action (v2: config-aware) ---
    prot_action = _assign_protection_action(fault_type, fault_zone, cfg, grounding)

    # --- Config features for model input (normalized) ---
    config_features = np.array([
        cfg["fault_mva"] / 500.0,
        cfg["xr_ratio"] / 15.0,
        cfg["source_count"] / 3.0,
    ], dtype=np.float32)

    return wf, fault_type, fault_zone, prot_action, cfg["id"], config_features


# ──────────────────────────────────────────────────────────────────
# Main Generation Loop
# ──────────────────────────────────────────────────────────────────

def generate_facility(
    facility_id: int,
    n_samples: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Generate and save waveforms for one facility."""
    profile = FACILITY_PROFILES[facility_id]
    rng = np.random.default_rng(seed + facility_id)

    fdir = output_dir / f"facility_{facility_id}"
    fdir.mkdir(parents=True, exist_ok=True)

    waveforms = np.zeros((n_samples, N_CHANNELS, N_SAMPLES_PER_WAVEFORM), dtype=np.float32)
    fault_types = np.zeros(n_samples, dtype=np.int64)
    fault_zones = np.zeros(n_samples, dtype=np.int64)
    prot_actions = np.zeros(n_samples, dtype=np.int64)
    config_ids = np.zeros(n_samples, dtype=np.int64)
    config_feats = np.zeros((n_samples, 3), dtype=np.float32)

    for i in range(n_samples):
        wf, ft, fz, pa, cid, cf = _generate_sample(profile, rng)
        waveforms[i] = wf
        fault_types[i] = ft
        fault_zones[i] = fz
        prot_actions[i] = pa
        config_ids[i] = cid
        config_feats[i] = cf

    # Save (backward-compatible with v1 + new config files)
    np.save(fdir / "waveforms.npy", waveforms)
    np.save(fdir / "fault_type.npy", fault_types)
    np.save(fdir / "fault_zone.npy", fault_zones)
    np.save(fdir / "protection_action.npy", prot_actions)
    np.save(fdir / "config_id.npy", config_ids)             # NEW v2
    np.save(fdir / "config_features.npy", config_feats)     # NEW v2

    # Stats
    print(f"  Facility {facility_id} ({profile['name']}):")
    print(f"    Samples:  {n_samples}")
    print(f"    FT dist:  {np.bincount(fault_types, minlength=4)}")
    print(f"    FZ dist:  {np.bincount(fault_zones, minlength=4)}")
    print(f"    PA dist:  {np.bincount(prot_actions, minlength=5)}")
    print(f"    Cfg dist: {np.bincount(config_ids, minlength=3)}")
    print(f"    Saved to: {fdir}")


def main():
    parser = argparse.ArgumentParser(description="SecureRelayFL waveform generator (v2)")
    parser.add_argument("--n-samples", type=int, default=1000, help="Samples per facility")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir", type=str, default="data/generated",
        help="Output directory for generated data",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    print(f"SecureRelayFL v2 — Generating {args.n_samples} samples/facility, seed={args.seed}")
    print(f"Output: {output_dir.resolve()}\n")

    for fid in FACILITY_PROFILES:
        generate_facility(fid, args.n_samples, args.seed, output_dir)

    total = args.n_samples * len(FACILITY_PROFILES)
    print(f"\nDone. {total} total samples across {len(FACILITY_PROFILES)} facilities.")


if __name__ == "__main__":
    main()