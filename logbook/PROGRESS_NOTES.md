# SecureRelayFL — Progress Notes for Manuscript

**Paper Title:** Privacy-Preserving Adaptive Relay Settings via Federated Learning Over Industrial Communication Networks

**Target:** MAIN 2026 (Mediterranean Artificial Intelligence and Networking Conference), Palermo, Italy, July 1–3, 2026

**Deadline:** March 14, 2026 | **Format:** 9 pages, IEEE 2-column | **Venue:** IEEE ComSoc co-sponsored, IEEE Xplore indexed

**Repo:** SecureRelayFL | **Description:** Privacy-preserving federated learning for adaptive relay protection over industrial communication networks

---

## 1. Paper Framing

This is a **communications/networking paper** that uses industrial power system protection as the application domain. The primary contributions are in federated learning under network impairments, privacy–accuracy trade-offs, and the dual-communication-layer problem — not in power system modeling itself.

The key novelty is the **dual communication layer insight**: industrial protection already relies on fast communication (GOOSE-based ZSI, pilot-assisted distance relaying), and we overlay a federated learning communication layer on top. The paper analyzes how these two communication requirements interact — whether FL aggregation traffic can coexist on the same industrial Ethernet without degrading protection signaling performance.

### Conference Topic Alignment

- Machine Learning and AI in and for networks
- Network security and privacy
- AI-as-a-Service
- AI-driven IoT networks
- Digital twins
- Edge, fog, and cloud computing

---

## 2. Problem Statement

Multiple industrial facilities (data centers, steel plants, petrochemical plants, pharmaceutical plants, cement plants) operate heterogeneous electrical protection systems. Optimizing protection relay settings requires facility-specific fault data that operators refuse to share due to proprietary and security concerns (data silos). Federated learning enables collaborative model training without centralizing raw data, but introduces communication challenges unique to safety-critical industrial environments.

---

## 3. Experimental Design

### 3.1 Facility Profiles (5 FL Clients)

| Parameter | Client 0: Data Center | Client 1: Steel Plant | Client 2: Petrochemical | Client 3: Pharmaceutical | Client 4: Cement Plant |
|---|---|---|---|---|---|
| Voltage (kV) | 13.8 | 34.5 | 13.8 | 4.16 | 34.5 |
| Fault MVA | 250 | 500+ | 150 | 100 | 400 |
| X/R Ratio | 8 | 15 | 6 | 5 | 12 |
| Grounding | Solidly grounded | Low-R grounded | High-R grounded | Resistance grounded | Low-R grounded |
| Noise (SNR dB) | 40 | 25 | 35 | 38 | 28 |
| CT Saturation Prob | 5% | 25% | 10% | 8% | 20% |
| THD | 3% | 15% | 5% | 8% | 12% |
| Key Loads | UPS, HVAC, static | Arc furnace, large motors | Induction motors, VFDs | Clean room, sensitive | Large mill motors |
| Primary Protection | ZSI (50/51) + 50G | Distance (21) + 87B + 51N | ZSI (50/51) + 51G | ZSI (50/51) + 50G | ZSI (50/51) + 51N |
| Comms-Dependent | ZSI blocking | Pilot distance + 87B | ZSI blocking | ZSI blocking | ZSI blocking |

### 3.2 Fault Types

- **0 — No fault:** Normal operation with sudden load change (motor starting, load switching). k_change ∈ [0.8, 1.3].
- **1 — Single line-to-ground (SLG):** Most common industrial fault. Grounding method determines fault current magnitude and voltage behavior.
- **2 — Line-to-line (LL):** Phase-to-phase fault. Not grounding-dependent for current, but voltage collapses on both faulted phases.
- **3 — High-impedance arcing fault (HIF):** Low current (0.5–75A), intermittent, with strong low-order harmonics (2nd, 3rd, 5th). Hardest to detect.

### 3.3 Fault Type Distribution per Facility (Non-IID)

| Facility | No-fault | SLG | LL | HIF |
|---|---|---|---|---|
| Data Center | 25% | 45% | 15% | 15% |
| Steel Plant | 20% | 30% | 35% | 15% |
| Petrochemical | 20% | 35% | 15% | 30% |
| Pharmaceutical | 25% | 40% | 20% | 15% |
| Cement Plant | 20% | 30% | 30% | 20% |

### 3.4 ML Task (Multi-task)

1. **Fault type classification** (4 classes): no-fault, SLG, LL, HIF
2. **Fault zone identification** (3 zones + no-fault): zone 0 (bus), zone 1 (near), zone 2 (far)
3. **Protection action classification** (5 classes): no-action, trip-instantaneous, trip-delayed, ZSI-block, alarm-only

### 3.5 Experiment Axes (Priority Order)

1. **Network Impairment Analysis** (must-have): Latency (1–500ms), packet loss (0–15%), bandwidth (1Mbps–unlimited)
2. **Privacy–Accuracy Trade-off** (must-have): Differential privacy ε = {0.1, 0.5, 1.0, 5.0, 10.0, ∞}
3. **Communication Efficiency + Dual-Layer Analysis** (should-have): Gradient compression, quantization, GOOSE co-existence
4. **FL Strategy Comparison** (nice-to-have): FedAvg vs. FedProx vs. SCAFFOLD

### 3.6 Baselines

- **Centralized:** All data pooled — upper bound on accuracy
- **Local-only:** No collaboration — lower bound, shows value of FL
- **Federated (ideal network):** Isolates network impairment effects

---

## 4. Data Generation — Synthetic Fault Waveform Generator

### 4.1 Overview

Physics-based analytical waveform generator in Python (`data/generator/generate.py`). No external simulation tool required. Generates 3-phase voltage and current waveforms (Va, Vb, Vc, Ia, Ib, Ic) parameterized by facility electrical characteristics.

**Output per facility:** 1,000 samples × 6 channels × 2,560 timesteps = ~61.4 MB
**Total:** 5,000 samples across 5 facilities, ~307 MB

### 4.2 Waveform Parameters

- Frequency: 60 Hz
- Samples per cycle: 256
- Sample rate: 15,360 Hz
- Pre-fault window: 3 cycles
- Post-fault window: 7 cycles
- Total window: 10 cycles (2,560 samples)
- Fault onset: sample 768 (after 3 cycles)

### 4.3 Waveform Physics

#### Pre-Fault (Steady State)

Three-phase balanced sinusoidal waveforms:

```
v_a(t) = V_peak × sin(ωt + φ_v)
v_b(t) = V_peak × sin(ωt + φ_v − 120°)
v_c(t) = V_peak × sin(ωt + φ_v + 120°)

i_a(t) = I_peak × L_pu × sin(ωt + φ_v − θ_pf)
```

Where:
- V_peak = V_kV × 1000 × √2 / √3 (phase peak from line-line kV)
- I_peak = S_fault / (√3 × V_kV) × √2 (peak of rated fault current)
- L_pu = random pre-fault load (0.3–0.9 pu)
- θ_pf = power factor angle (0.15–0.45 rad)
- φ_v = random point-on-wave inception angle
- ω = 2π × 60

#### Post-Fault: Asymmetrical Current (IEEE 551 / IEC 60909)

```
i(t) = I_f,peak × [sin(ωt + α) − sin(α) × exp(−t/τ)]
```

Where:
- α = θ − φ_z (current reference angle)
- θ = voltage angle at fault inception
- φ_z = arctan(X/R) (impedance angle)
- τ = (X/R) / ω (DC offset time constant)
- At t=0: i(0) = I_f,peak × [sin(α) − sin(α)] = 0 ✓ (satisfies initial condition)

**Verified against:** IEEE 551-2006, IEC 60909, IEEE C37.010 tables

#### Post-Fault: Voltage Sag/Swell (Grounding-Dependent)

Governed by ratio m = Z0/Z1 (zero-sequence to positive-sequence impedance):

| Grounding | m range | Faulted Phase Sag (pu) | Unfaulted Phase Swell (pu) |
|---|---|---|---|
| Solidly grounded | 1–3 | 0.10–0.50 | 1.00–1.25 |
| Low-R grounded | 3–10 | 0.15–0.55 | 1.10–1.40 |
| High-R grounded | →∞ | 0.30–0.80 | 1.40–1.73 |
| Ungrounded | ∞ | 0.40–0.90 | 1.50–1.73 |

**Verified against:** IEEE C62.92 (effectively grounded: Z0/Z1 ≤ 3), voltage-disturbance.com analysis, IEEE PSRC reports. The √3 = 1.73 pu upper limit for ungrounded/HRG systems is a well-established result.

#### High-Impedance Arcing Fault Model

Based on IEEE PSRC Working Group D15 report and Emanuel two-diode model:

```
i_arc(t) = [I_HIF × sin(ωt + φ) × asym(t)] × E(t) + h2(t) + h3(t) + h5(t)
```

Where:
- I_HIF: 0.5–5A (HRG) or 5–50A (other grounding). PSRC D15 confirms 0–75A typical range.
- asym(t): Half-cycle asymmetry (different Rp, Rn per Emanuel model). Ratio 0.6–0.9.
- E(t): Random arc extinction envelope — intervals where multiplier drops to 0.0–0.3, simulating intermittent restrike.
- h2, h3, h5: Explicit 2nd, 3rd, 5th harmonic injection (PSRC D15: harmonics 2nd–7th strongly present in HIF current). Amplitudes: h3 = 20% of fundamental, h2 = 15%, h5 = 8%.
- 3rd harmonic has near-in-phase relationship to faulted phase voltage (PSRC D15 finding).
- Total faulted phase current = i_arc(t) + i_load(t)

**Verified against:** IEEE PSRC WG D15 "High Impedance Fault Detection Technology", ScienceDirect HIF detection literature, Frontiers energy research HIF survey.

#### Post-Processing Distortions

**Harmonics** (3rd, 5th, 7th, 11th, 13th — standard 6-pulse characteristic harmonics):
```
v(t) ← v(t) + Σ w_h × THD × V_fund × sin(h×ω×t + φ_h)
```
Weights: [0.40, 0.25, 0.15, 0.12, 0.08]. Facility-specific THD: 3%–15%.

**CT Saturation** (current channels only, probabilistic):
- Clipping at C = I_max × (1 − 0.5 × S) with exponential rounding near boundary
- Severity S ∈ [0.2, 0.8], probability varies by facility (5%–25%)
- Applied only during fault conditions (fault_type > 0)

**Gaussian White Noise:**
```
x(t) ← x(t) + N(0, σ²)  where σ² = P_signal / 10^(SNR/10)
```
SNR varies: 25 dB (steel plant) to 40 dB (data center).

### 4.4 Labels

| Label | Type | Values |
|---|---|---|
| fault_type | Classification (4) | 0=no-fault, 1=SLG, 2=LL, 3=HIF |
| fault_zone | Classification (3+1) | -1=N/A, 0=bus, 1=near, 2=far |
| protection_action | Classification (5) | 0=no-action, 1=trip-instant, 2=trip-delayed, 3=ZSI-block, 4=alarm-only |

### 4.5 File Structure

```
data/generated/
├── summary.json                    # Overall dataset summary
├── facility_0/                     # Data Center
│   ├── waveforms.npy              # (1000, 6, 2560) float32
│   ├── fault_type.npy             # (1000,) int64
│   ├── fault_zone.npy             # (1000,) int64
│   ├── protection_action.npy      # (1000,) int64
│   └── metadata.json              # Facility profile + label counts
├── facility_1/                     # Steel Plant
│   └── ...
├── facility_2/                     # Petrochemical
│   └── ...
├── facility_3/                     # Pharmaceutical
│   └── ...
└── facility_4/                     # Cement Plant
    └── ...
```

---

## 5. Remaining Build Order

1. ~~Data generation~~ ✅ DONE
2. **Centralized baseline model** — 1D-CNN + MLP fault classifier in PyTorch (proves ML task works)
3. **FL framework** — Flower server + clients with configurable aggregation
4. **Network impairment layer** — Latency, packet loss, bandwidth emulation
5. **Differential privacy integration** — Opacus library or manual Gaussian mechanism
6. **Run experiments** — Sweep across axes in priority order
7. **Generate publication-quality figures** — 5–6 figures for the paper
8. **Manuscript writeup** — IEEE 2-column, 9 pages

---

## 6. Target Figures for Paper

1. **System architecture diagram** — 5 facilities, power system topologies, FL overlay, dual communication layers
2. **Accuracy vs. network impairment heatmap** — latency × packet loss → protection action classification accuracy
3. **Privacy–accuracy–communication Pareto frontier** — ε vs. accuracy vs. bytes transmitted
4. **Convergence curves** — FL strategies under ideal vs. impaired networks
5. **Dual-layer analysis** — GOOSE message delay vs. FL traffic load, showing critical threshold
6. **Per-facility accuracy comparison** — centralized vs. local-only vs. federated, per plant type

---

## 7. Paper Structure (9 pages)

1. **Abstract** (~200 words)
2. **I. Introduction** (~1 page) — Comms-first framing, data silo problem, contributions
3. **II. Related Work** (~1 page) — ML in protection, FL in power systems, communication-dependent protection
4. **III. System Model** (~1.5 pages) — Facility models (Table I), fault generation, ML formulation, FL framework
5. **IV. Experimental Setup** (~1 page) — Axes, baselines, metrics, network simulation parameters
6. **V. Results** (~2.5 pages) — Figures 2–6, analysis per axis
7. **VI. Discussion** (~0.5 pages) — Dual-layer implications, practical deployment considerations
8. **VII. Conclusion** (~0.5 pages)
9. **References** (unlimited space per MAIN 2026 rules)

---

## 8. Key References to Cite

- IEEE 551-2006 (fault current calculations)
- IEC 60909 (short-circuit currents)
- IEEE PSRC WG D15 (HIF detection technology)
- IEC 61850 / GOOSE messaging performance
- IEEE C62.92 (grounding guide)
- Frontiers review: ML in power system protection (2024)
- MDPI Energies: Distributed Learning in Power Systems (2021)
- arxiv 2509.09053: Scoping review ML in protection (2025)
- Flower FL framework
- McMahan et al. FedAvg (2017)
- Abadi et al. Deep Learning with Differential Privacy (2016)
