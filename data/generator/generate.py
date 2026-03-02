"""
SecureRelayFL - Synthetic Fault Waveform Generator
===================================================

Generates physics-based fault current and voltage waveforms for 5 industrial
facility profiles. Each facility has distinct electrical characteristics that
create naturally non-IID data distributions for federated learning experiments.

Waveform model:
    Pre-fault:  v(t) = V_peak * sin(ωt + φ_v)
                i(t) = I_peak * sin(ωt + φ_i)

    Post-fault: i(t) = I_f_peak * [sin(ωt + φ_f) - sin(φ_f) * exp(-t/τ)] + noise
                v(t) = V_sag * sin(ωt + φ_v) + harmonics + noise

    where τ = L/R = (X/R) / ω  (DC offset decay time constant)

Fault types:
    0 - No fault (normal operation / motor starting / load switching)
    1 - Single line-to-ground (SLG)
    2 - Line-to-line (LL)
    3 - High-impedance arcing fault (HIF)

Protection actions:
    0 - No action
    1 - Trip instantaneous
    2 - Trip with time delay
    3 - ZSI block (send restraint signal)
    4 - Alarm only (HIF on HRG system)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
import os
import json


# ============================================================================
# Facility Profiles
# ============================================================================

@dataclass
class FacilityProfile:
    """Electrical characteristics defining a facility."""
    name: str
    voltage_kv: float
    fault_mva: float
    xr_ratio: float
    grounding: str              # "solid", "low_r", "high_r", "ungrounded", "resistance"
    noise_snr_db: float         # Signal-to-noise ratio
    ct_saturation_prob: float   # Probability of CT saturation per sample
    harmonic_distortion: float  # THD as fraction (0.0 - 0.3)
    fault_type_dist: dict       # {fault_type: probability}
    description: str = ""


FACILITY_PROFILES = {
    0: FacilityProfile(
        name="Data Center",
        voltage_kv=13.8,
        fault_mva=250,
        xr_ratio=8,
        grounding="solid",
        noise_snr_db=40,
        ct_saturation_prob=0.05,
        harmonic_distortion=0.03,
        fault_type_dist={0: 0.25, 1: 0.45, 2: 0.15, 3: 0.15},
        description="Main-tie-main bus, UPS loads, clean waveforms"
    ),
    1: FacilityProfile(
        name="Steel Plant",
        voltage_kv=34.5,
        fault_mva=500,
        xr_ratio=15,
        grounding="low_r",
        noise_snr_db=25,
        ct_saturation_prob=0.25,
        harmonic_distortion=0.15,
        fault_type_dist={0: 0.20, 1: 0.30, 2: 0.35, 3: 0.15},
        description="Arc furnace, large motors, high fault current, CT saturation"
    ),
    2: FacilityProfile(
        name="Petrochemical",
        voltage_kv=13.8,
        fault_mva=150,
        xr_ratio=6,
        grounding="high_r",
        noise_snr_db=35,
        ct_saturation_prob=0.10,
        harmonic_distortion=0.05,
        fault_type_dist={0: 0.20, 1: 0.35, 2: 0.15, 3: 0.30},
        description="Induction motors, VFDs, HRG system, high-impedance faults"
    ),
    3: FacilityProfile(
        name="Pharmaceutical",
        voltage_kv=4.16,
        fault_mva=100,
        xr_ratio=5,
        grounding="resistance",
        noise_snr_db=38,
        ct_saturation_prob=0.08,
        harmonic_distortion=0.08,
        fault_type_dist={0: 0.25, 1: 0.40, 2: 0.20, 3: 0.15},
        description="Clean room loads, sensitive equipment, moderate fault level"
    ),
    4: FacilityProfile(
        name="Cement Plant",
        voltage_kv=34.5,
        fault_mva=400,
        xr_ratio=12,
        grounding="low_r",
        noise_snr_db=28,
        ct_saturation_prob=0.20,
        harmonic_distortion=0.12,
        fault_type_dist={0: 0.20, 1: 0.30, 2: 0.30, 3: 0.20},
        description="Large mill motors, high inertia loads, dusty environment"
    ),
}


# ============================================================================
# Waveform Physics
# ============================================================================

FREQ_HZ = 60
OMEGA = 2 * np.pi * FREQ_HZ
SAMPLES_PER_CYCLE = 256
CYCLES_PREFAULT = 3
CYCLES_POSTFAULT = 7
TOTAL_CYCLES = CYCLES_PREFAULT + CYCLES_POSTFAULT
TOTAL_SAMPLES = TOTAL_CYCLES * SAMPLES_PER_CYCLE
SAMPLE_RATE = FREQ_HZ * SAMPLES_PER_CYCLE  # 15360 Hz
DT = 1.0 / SAMPLE_RATE

# Fault onset sample index
FAULT_ONSET = CYCLES_PREFAULT * SAMPLES_PER_CYCLE


def _time_array():
    """Generate time array for the full waveform window."""
    return np.arange(TOTAL_SAMPLES) * DT


def _add_noise(signal: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian white noise at specified SNR."""
    signal_power = np.mean(signal ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def _add_harmonics(signal: np.ndarray, t: np.ndarray, thd: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Add harmonic distortion (3rd, 5th, 7th, 11th, 13th)."""
    if thd < 0.001:
        return signal
    harmonics = [3, 5, 7, 11, 13]
    # Distribute THD across harmonics with decreasing amplitude
    weights = np.array([0.40, 0.25, 0.15, 0.12, 0.08])
    fund_amplitude = np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else 1.0
    for h, w in zip(harmonics, weights):
        h_amp = fund_amplitude * thd * w
        h_phase = rng.uniform(0, 2 * np.pi)
        signal = signal + h_amp * np.sin(h * OMEGA * t + h_phase)
    return signal


def _ct_saturation(signal: np.ndarray, severity: float) -> np.ndarray:
    """Simulate CT saturation by clipping and distorting peaks."""
    clip_level = np.max(np.abs(signal)) * (1.0 - 0.5 * severity)
    saturated = np.clip(signal, -clip_level, clip_level)
    # Add mild exponential rounding near saturation
    mask = np.abs(saturated) > 0.8 * clip_level
    saturated[mask] = np.sign(saturated[mask]) * clip_level * (
        1 - 0.1 * np.exp(-3 * (np.abs(saturated[mask]) / clip_level))
    )
    return saturated


def _arcing_fault_current(t_fault: np.ndarray, i_peak: float, omega: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Generate high-impedance arcing fault current with realistic characteristics.

    Based on IEEE PSRC WG D15 report and Emanuel two-diode model:
    - Intermittent arc extinction and restrike (random envelope)
    - Asymmetry between positive and negative half-cycles (different Rp, Rn)
    - Strong low-order harmonics (2nd, 3rd dominant) per PSRC findings
    - Fault currents typically 0-75A with very erratic waveforms
    """
    phi_arc = rng.uniform(0, 2 * np.pi)
    base = i_peak * np.sin(omega * t_fault + phi_arc)

    # Asymmetry between positive and negative half-cycles (Emanuel model)
    # Different effective resistance in each half-cycle
    asym_ratio = rng.uniform(0.6, 0.9)  # Rn/Rp ratio
    negative_mask = base < 0
    base[negative_mask] *= asym_ratio

    # Random arc extinction and restrike envelope
    arc_envelope = np.ones_like(t_fault)
    n_samples = len(t_fault)
    n_extinctions = rng.integers(3, 12)
    for _ in range(n_extinctions):
        start = rng.integers(0, max(1, n_samples - 50))
        duration = rng.integers(10, 80)
        end = min(start + duration, n_samples)
        arc_envelope[start:end] *= rng.uniform(0.0, 0.3)

    # Explicit low-order harmonics (PSRC D15: 2nd-7th strongly present in HIF)
    # 3rd harmonic has unique phase relationship to faulted phase voltage
    h3 = 0.20 * i_peak * np.sin(3 * omega * t_fault + phi_arc + rng.uniform(-0.3, 0.3))
    # 2nd harmonic (indicator of asymmetry / nonlinearity)
    h2 = 0.15 * i_peak * np.sin(2 * omega * t_fault + rng.uniform(0, np.pi))
    # 5th harmonic
    h5 = 0.08 * i_peak * np.sin(5 * omega * t_fault + rng.uniform(0, 2 * np.pi))

    return base * arc_envelope + h3 * arc_envelope + h2 * arc_envelope + h5 * arc_envelope


# ============================================================================
# Main Generator
# ============================================================================

def generate_single_sample(
    facility: FacilityProfile,
    fault_type: int,
    rng: np.random.Generator,
    fault_location: Optional[float] = None,  # 0.0 = bus, 1.0 = end of feeder
) -> dict:
    """
    Generate a single 3-phase fault waveform sample.

    Returns dict with:
        - 'va', 'vb', 'vc': voltage waveforms (TOTAL_SAMPLES,)
        - 'ia', 'ib', 'ic': current waveforms (TOTAL_SAMPLES,)
        - 'fault_type': int (0-3)
        - 'fault_zone': int (0=bus, 1=near, 2=far)
        - 'protection_action': int (0-4)
        - 'facility_id': int
        - 'metadata': dict with generation parameters
    """
    t = _time_array()
    t_fault = t[FAULT_ONSET:] - t[FAULT_ONSET]  # Time relative to fault onset

    # Base system parameters
    v_base = facility.voltage_kv * 1000 * np.sqrt(2) / np.sqrt(3)  # Phase peak voltage
    z_base = (facility.voltage_kv ** 2) / facility.fault_mva  # Ohms
    i_base = (facility.fault_mva * 1e6) / (np.sqrt(3) * facility.voltage_kv * 1e3)
    i_peak_base = i_base * np.sqrt(2)

    # Random pre-fault load (0.3 - 0.9 pu)
    load_pu = rng.uniform(0.3, 0.9)
    pf_angle = rng.uniform(0.15, 0.45)  # Power factor angle (radians)

    # Fault location affects impedance seen
    if fault_location is None:
        fault_location = rng.uniform(0.0, 1.0)

    # Impedance to fault (affects fault current magnitude)
    z_fault_pu = 0.05 + fault_location * 0.4  # 0.05 to 0.45 pu
    tau = facility.xr_ratio / OMEGA  # DC offset time constant

    # Phase angles (120° apart)
    phase_offsets = np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])
    phi_v = rng.uniform(0, 2 * np.pi)  # Random point-on-wave

    # ---- Pre-fault waveforms ----
    va_pre = v_base * np.sin(OMEGA * t[:FAULT_ONSET] + phi_v + phase_offsets[0])
    vb_pre = v_base * np.sin(OMEGA * t[:FAULT_ONSET] + phi_v + phase_offsets[1])
    vc_pre = v_base * np.sin(OMEGA * t[:FAULT_ONSET] + phi_v + phase_offsets[2])

    ia_pre = i_peak_base * load_pu * np.sin(OMEGA * t[:FAULT_ONSET] + phi_v + phase_offsets[0] - pf_angle)
    ib_pre = i_peak_base * load_pu * np.sin(OMEGA * t[:FAULT_ONSET] + phi_v + phase_offsets[1] - pf_angle)
    ic_pre = i_peak_base * load_pu * np.sin(OMEGA * t[:FAULT_ONSET] + phi_v + phase_offsets[2] - pf_angle)

    # ---- Post-fault waveforms ----
    if fault_type == 0:
        # No fault - continue normal operation with possible load change
        load_change = rng.uniform(0.8, 1.3)  # Sudden load change
        va_post = v_base * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[0])
        vb_post = v_base * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[1])
        vc_post = v_base * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2])
        ia_post = i_peak_base * load_pu * load_change * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[0] - pf_angle)
        ib_post = i_peak_base * load_pu * load_change * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[1] - pf_angle)
        ic_post = i_peak_base * load_pu * load_change * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2] - pf_angle)

        fault_zone = -1
        protection_action = 0

    elif fault_type == 1:
        # SLG fault on phase A
        # Fault current magnitude depends on grounding and location
        # IEEE 551 / IEC 60909: i(t) = sqrt(2)*If*[sin(wt+θ-φ) - sin(θ-φ)*exp(-t/τ)]
        # where θ = voltage angle at inception, φ = impedance angle ≈ atan(X/R)
        if facility.grounding == "solid":
            i_fault_mult = rng.uniform(3.0, 8.0) / (z_fault_pu + 0.01)
        elif facility.grounding in ("low_r", "resistance"):
            i_fault_mult = rng.uniform(1.5, 5.0) / (z_fault_pu + 0.05)
        elif facility.grounding == "high_r":
            i_fault_mult = rng.uniform(0.5, 2.0)  # Limited by grounding resistor
        else:
            i_fault_mult = rng.uniform(2.0, 6.0) / (z_fault_pu + 0.02)

        i_fault_peak = min(i_fault_mult * i_peak_base, 20 * i_peak_base)
        # Impedance angle φ = atan(X/R); fault inception angle on voltage wave
        phi_z = np.arctan(facility.xr_ratio)  # Impedance angle
        theta = phi_v + phase_offsets[0]       # Voltage angle at fault inception
        alpha = theta - phi_z                  # Current reference angle

        # Phase A: asymmetrical fault current with DC offset (IEEE 551 form)
        # i(t) = I_peak * [sin(ωt + α) - sin(α) * exp(-t/τ)]
        # At t=0: i(0) = I_peak*[sin(α) - sin(α)] = 0 (satisfies initial condition)
        ia_post = i_fault_peak * (
            np.sin(OMEGA * t_fault + alpha)
            - np.sin(alpha) * np.exp(-t_fault / tau)
        )
        # Phases B, C: slightly affected (negative/zero sequence contribution)
        neg_seq_factor = rng.uniform(0.05, 0.20)
        ib_post = i_peak_base * load_pu * (1 + neg_seq_factor) * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[1] - pf_angle)
        ic_post = i_peak_base * load_pu * (1 + neg_seq_factor) * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2] - pf_angle)

        # Voltage sag/swell depends on grounding method
        # Governed by ratio m = Z0/Z1 (zero-seq to pos-seq impedance)
        # Solidly grounded (m ≈ 1-3): unfaulted swell 1.0-1.25 pu, faulted sag 0.1-0.5 pu
        # Low-R grounded (m ≈ 3-10): unfaulted swell 1.1-1.4 pu, faulted sag 0.15-0.55 pu
        # High-R grounded (m → ∞): unfaulted swell up to √3=1.73 pu, faulted sag 0.3-0.8 pu
        if facility.grounding == "solid":
            v_sag = rng.uniform(0.10, 0.50)
            v_swell = rng.uniform(1.00, 1.25)
        elif facility.grounding in ("low_r", "resistance"):
            v_sag = rng.uniform(0.15, 0.55)
            v_swell = rng.uniform(1.10, 1.40)
        elif facility.grounding == "high_r":
            v_sag = rng.uniform(0.30, 0.80)   # Less collapse, limited fault current
            v_swell = rng.uniform(1.40, 1.73)  # Up to line-to-line voltage
        else:  # ungrounded
            v_sag = rng.uniform(0.40, 0.90)
            v_swell = rng.uniform(1.50, 1.73)

        va_post = v_base * v_sag * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[0])
        vb_post = v_base * v_swell * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[1])
        vc_post = v_base * v_swell * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2])

        fault_zone = 0 if fault_location < 0.2 else (1 if fault_location < 0.6 else 2)
        # Protection action depends on grounding and zone
        if facility.grounding == "high_r":
            protection_action = 4  # Alarm only (HRG)
        elif fault_zone == 0:
            protection_action = 1  # Trip instantaneous (bus fault)
        else:
            protection_action = 2  # Trip with delay

    elif fault_type == 2:
        # Line-to-line fault (A-B)
        i_fault_mult = rng.uniform(2.5, 7.0) / (z_fault_pu + 0.01)
        i_fault_peak = min(i_fault_mult * i_peak_base, 18 * i_peak_base)

        # Impedance angle and current reference (IEEE 551 form)
        phi_z = np.arctan(facility.xr_ratio)
        theta_a = phi_v + phase_offsets[0]
        theta_b = phi_v + phase_offsets[1]
        alpha_a = theta_a - phi_z
        alpha_b = theta_b - phi_z

        # Both faulted phases see high current with DC offset
        ia_post = i_fault_peak * (
            np.sin(OMEGA * t_fault + alpha_a)
            - np.sin(alpha_a) * np.exp(-t_fault / tau)
        )
        ib_post = -i_fault_peak * (
            np.sin(OMEGA * t_fault + alpha_b)
            - np.sin(alpha_b) * np.exp(-t_fault / tau)
        )
        # Phase C relatively unaffected
        ic_post = i_peak_base * load_pu * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2] - pf_angle)

        # LL fault voltage collapse on both faulted phases
        # Less grounding-dependent than SLG, but still varies with fault location
        v_sag = rng.uniform(0.15, 0.55)
        va_post = v_base * v_sag * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[0])
        vb_post = v_base * v_sag * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[1])
        vc_post = v_base * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2])

        fault_zone = 0 if fault_location < 0.2 else (1 if fault_location < 0.6 else 2)
        if fault_zone == 0:
            protection_action = 1  # Trip instantaneous
        else:
            protection_action = 2  # Trip with delay

    elif fault_type == 3:
        # High-impedance arcing fault on phase A
        # Much lower fault current, intermittent
        if facility.grounding == "high_r":
            i_hif_peak = rng.uniform(0.5, 5.0)  # Very low, amps
        else:
            i_hif_peak = rng.uniform(5.0, 50.0)  # Still relatively low

        ia_post = _arcing_fault_current(t_fault, i_hif_peak, OMEGA, rng)
        # Add load current back
        ia_post += i_peak_base * load_pu * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[0] - pf_angle)

        ib_post = i_peak_base * load_pu * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[1] - pf_angle)
        ic_post = i_peak_base * load_pu * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2] - pf_angle)

        # HIF causes minimal voltage disturbance on solidly grounded systems
        # but significant on HRG/ungrounded (same physics as SLG, just less current)
        if facility.grounding == "high_r":
            v_sag = rng.uniform(0.85, 0.98)     # Slight sag on faulted phase
            v_swell = rng.uniform(1.10, 1.50)   # Moderate swell (less than bolted SLG)
        elif facility.grounding in ("low_r", "resistance"):
            v_sag = rng.uniform(0.90, 0.99)
            v_swell = rng.uniform(1.02, 1.15)
        else:  # solid
            v_sag = rng.uniform(0.92, 0.99)
            v_swell = rng.uniform(1.00, 1.05)   # Barely noticeable on solidly grounded

        va_post = v_base * v_sag * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[0])
        vb_post = v_base * v_swell * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[1])
        vc_post = v_base * v_swell * np.sin(OMEGA * t[FAULT_ONSET:] + phi_v + phase_offsets[2])

        fault_zone = 0 if fault_location < 0.3 else (1 if fault_location < 0.7 else 2)
        # HIF protection action
        if facility.grounding == "high_r":
            protection_action = 4  # Alarm only
        else:
            protection_action = 3  # ZSI block / delayed response

    else:
        raise ValueError(f"Unknown fault type: {fault_type}")

    # ---- Concatenate pre + post ----
    va = np.concatenate([va_pre, va_post])
    vb = np.concatenate([vb_pre, vb_post])
    vc = np.concatenate([vc_pre, vc_post])
    ia = np.concatenate([ia_pre, ia_post])
    ib = np.concatenate([ib_pre, ib_post])
    ic = np.concatenate([ic_pre, ic_post])

    # ---- Apply facility-specific distortions ----

    # Harmonics (applied to full waveform for consistency)
    if facility.harmonic_distortion > 0.001:
        va = _add_harmonics(va, t, facility.harmonic_distortion * rng.uniform(0.5, 1.5), rng)
        vb = _add_harmonics(vb, t, facility.harmonic_distortion * rng.uniform(0.5, 1.5), rng)
        vc = _add_harmonics(vc, t, facility.harmonic_distortion * rng.uniform(0.5, 1.5), rng)
        ia = _add_harmonics(ia, t, facility.harmonic_distortion * rng.uniform(0.5, 1.5), rng)
        ib = _add_harmonics(ib, t, facility.harmonic_distortion * rng.uniform(0.5, 1.5), rng)
        ic = _add_harmonics(ic, t, facility.harmonic_distortion * rng.uniform(0.5, 1.5), rng)

    # CT saturation (current channels only, probabilistic)
    if fault_type > 0 and rng.random() < facility.ct_saturation_prob:
        severity = rng.uniform(0.2, 0.8)
        ia = _ct_saturation(ia, severity)
        if fault_type == 2:  # LL fault affects phase B too
            ib = _ct_saturation(ib, severity * rng.uniform(0.5, 1.0))

    # Noise
    va = _add_noise(va, facility.noise_snr_db, rng)
    vb = _add_noise(vb, facility.noise_snr_db, rng)
    vc = _add_noise(vc, facility.noise_snr_db, rng)
    ia = _add_noise(ia, facility.noise_snr_db, rng)
    ib = _add_noise(ib, facility.noise_snr_db, rng)
    ic = _add_noise(ic, facility.noise_snr_db, rng)

    return {
        'va': va, 'vb': vb, 'vc': vc,
        'ia': ia, 'ib': ib, 'ic': ic,
        'fault_type': fault_type,
        'fault_zone': fault_zone if fault_type > 0 else -1,
        'protection_action': protection_action,
        'facility_id': list(FACILITY_PROFILES.keys())[
            list(FACILITY_PROFILES.values()).index(facility)
        ],
        'metadata': {
            'fault_location': float(fault_location),
            'load_pu': float(load_pu),
            'phi_v': float(phi_v),
            'v_base': float(v_base),
            'i_peak_base': float(i_peak_base),
        }
    }


def generate_facility_dataset(
    facility_id: int,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Generate a complete dataset for one facility.

    Returns dict with:
        - 'waveforms': np.ndarray (n_samples, 6, TOTAL_SAMPLES) [va,vb,vc,ia,ib,ic]
        - 'fault_type': np.ndarray (n_samples,)
        - 'fault_zone': np.ndarray (n_samples,)
        - 'protection_action': np.ndarray (n_samples,)
        - 'facility_id': int
        - 'metadata': list of dicts
    """
    facility = FACILITY_PROFILES[facility_id]
    rng = np.random.default_rng(seed + facility_id * 1000)

    # Sample fault types according to facility distribution
    fault_types_list = list(facility.fault_type_dist.keys())
    fault_probs = list(facility.fault_type_dist.values())
    sampled_fault_types = rng.choice(fault_types_list, size=n_samples, p=fault_probs)

    waveforms = np.zeros((n_samples, 6, TOTAL_SAMPLES), dtype=np.float32)
    fault_types = np.zeros(n_samples, dtype=np.int64)
    fault_zones = np.zeros(n_samples, dtype=np.int64)
    protection_actions = np.zeros(n_samples, dtype=np.int64)
    metadata_list = []

    for i in range(n_samples):
        sample = generate_single_sample(facility, int(sampled_fault_types[i]), rng)
        waveforms[i, 0] = sample['va']
        waveforms[i, 1] = sample['vb']
        waveforms[i, 2] = sample['vc']
        waveforms[i, 3] = sample['ia']
        waveforms[i, 4] = sample['ib']
        waveforms[i, 5] = sample['ic']
        fault_types[i] = sample['fault_type']
        fault_zones[i] = sample['fault_zone']
        protection_actions[i] = sample['protection_action']
        metadata_list.append(sample['metadata'])

    return {
        'waveforms': waveforms,
        'fault_type': fault_types,
        'fault_zone': fault_zones,
        'protection_action': protection_actions,
        'facility_id': facility_id,
        'metadata': metadata_list,
    }


def generate_all_facilities(
    n_samples_per_facility: int = 1000,
    seed: int = 42,
    output_dir: str = "data/generated",
) -> dict:
    """
    Generate datasets for all 5 facilities and save to disk.

    Returns dict mapping facility_id -> dataset dict.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_data = {}

    for fid in FACILITY_PROFILES:
        print(f"Generating {n_samples_per_facility} samples for "
              f"Facility {fid}: {FACILITY_PROFILES[fid].name}...")
        dataset = generate_facility_dataset(fid, n_samples_per_facility, seed)

        # Save as numpy archives
        facility_dir = os.path.join(output_dir, f"facility_{fid}")
        os.makedirs(facility_dir, exist_ok=True)

        np.save(os.path.join(facility_dir, "waveforms.npy"), dataset['waveforms'])
        np.save(os.path.join(facility_dir, "fault_type.npy"), dataset['fault_type'])
        np.save(os.path.join(facility_dir, "fault_zone.npy"), dataset['fault_zone'])
        np.save(os.path.join(facility_dir, "protection_action.npy"), dataset['protection_action'])

        # Save metadata as JSON
        with open(os.path.join(facility_dir, "metadata.json"), 'w') as f:
            json.dump({
                'facility_profile': {
                    'name': FACILITY_PROFILES[fid].name,
                    'voltage_kv': FACILITY_PROFILES[fid].voltage_kv,
                    'fault_mva': FACILITY_PROFILES[fid].fault_mva,
                    'xr_ratio': FACILITY_PROFILES[fid].xr_ratio,
                    'grounding': FACILITY_PROFILES[fid].grounding,
                    'description': FACILITY_PROFILES[fid].description,
                },
                'n_samples': n_samples_per_facility,
                'seed': seed,
                'waveform_params': {
                    'freq_hz': FREQ_HZ,
                    'samples_per_cycle': SAMPLES_PER_CYCLE,
                    'total_cycles': TOTAL_CYCLES,
                    'cycles_prefault': CYCLES_PREFAULT,
                    'cycles_postfault': CYCLES_POSTFAULT,
                    'sample_rate_hz': SAMPLE_RATE,
                },
                'label_counts': {
                    'fault_type': {str(k): int(v) for k, v in
                                   zip(*np.unique(dataset['fault_type'], return_counts=True))},
                    'protection_action': {str(k): int(v) for k, v in
                                          zip(*np.unique(dataset['protection_action'], return_counts=True))},
                },
            }, f, indent=2)

        all_data[fid] = dataset
        print(f"  -> Saved to {facility_dir}/")
        print(f"     Fault types: {dict(zip(*np.unique(dataset['fault_type'], return_counts=True)))}")
        print(f"     Protection actions: {dict(zip(*np.unique(dataset['protection_action'], return_counts=True)))}")

    # Save summary
    summary = {
        'n_facilities': len(FACILITY_PROFILES),
        'n_samples_per_facility': n_samples_per_facility,
        'total_samples': n_samples_per_facility * len(FACILITY_PROFILES),
        'facilities': {fid: FACILITY_PROFILES[fid].name for fid in FACILITY_PROFILES},
        'fault_type_map': {0: 'no_fault', 1: 'SLG', 2: 'LL', 3: 'HIF'},
        'protection_action_map': {
            0: 'no_action', 1: 'trip_instantaneous', 2: 'trip_delayed',
            3: 'zsi_block', 4: 'alarm_only'
        },
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone. Total: {n_samples_per_facility * len(FACILITY_PROFILES)} samples across {len(FACILITY_PROFILES)} facilities.")
    return all_data


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SecureRelayFL Synthetic Fault Waveform Generator")
    parser.add_argument("--n-samples", type=int, default=1000, help="Samples per facility")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", type=str, default="data/generated", help="Output directory")
    args = parser.parse_args()

    generate_all_facilities(args.n_samples, args.seed, args.output_dir)
