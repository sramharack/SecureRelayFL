# SecureRelayFL

**Privacy-Preserving Adaptive Relay Settings via Federated Learning Over Industrial Communication Networks**

[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/Flower-1.12-orange.svg)](https://flower.ai/)

---

## Overview

This repository contains the complete implementation for analyzing **federated learning (FL) traffic coexistence with IEC 61850 GOOSE protection messaging** on shared industrial Ethernet. The paper presents the first quantitative analysis of this dual communication layer interaction, using adaptive relay setting optimization as a representative safety-critical FL application.

### Key Findings

| Finding | Result |
|---------|--------|
| **GOOSE–FL coexistence** | HOL blocking = 0.121 ms on 100 Mbps (40× below IEC 61850 Type 1A 3 ms limit) |
| **Configuration embedding** | Centralized PA accuracy: 90.4% → 99.4% with 3-dim config features |
| **16-bit quantization** | +5.4 points PA over 32-bit baseline (regularization effect) |
| **Packet loss threshold** | >10% causes catastrophic collapse (PA → 0.40) |
| **Differential privacy** | Not viable for protection FL — ε=10 drops PA to 15.4% |
| **FL accuracy gap** | Local avg (97.8%) vs FedAvg (66.5%) — structural, not methodological |

---

## Architecture

### FaultClassifierV2

A multi-task 1D-CNN with temporal attention pooling and configuration embedding (118,382 parameters, 0.45 MB):

```
Input: 6 channels × 2,560 samples (10 cycles @ 256 samples/cycle)
  ├── Conv1D(32, k=7, s=2) + GroupNorm + ReLU + MaxPool
  ├── Conv1D(64, k=5, s=2) + GroupNorm + ReLU + MaxPool
  ├── Conv1D(128, k=3, s=1) + GroupNorm + ReLU + MaxPool
  ├── Conv1D(128, k=3, s=1) + GroupNorm + ReLU
  └── Temporal Attention Pooling → 128-dim

Config Input: [fault_mva/500, xr_ratio/15, source_count/3]
  └── Linear(3→32) + ReLU → 32-dim

Concat(128 + 32 = 160-dim)
  ├── Head: Fault Type    → FC(160,64) → FC(64,4)
  ├── Head: Fault Zone    → FC(160,64) → FC(64,4)
  └── Head: Prot. Action  → FC(160,64) → FC(64,5)
```

**Design decisions:**
- **GroupNorm** (8 groups/layer) instead of BatchNorm — BatchNorm diverges in FL after ~5 rounds due to non-IID batch statistics
- **Temporal attention** instead of global average pooling — learns to attend to fault inception transient vs. steady-state current
- **Configuration embedding** resolves the many-to-many mapping between waveforms and protection actions

### Five Industrial Facilities

| Facility | Voltage | Grounding | Peak MVA | X/R | Protection |
|----------|---------|-----------|----------|-----|------------|
| F1: Data Center | 13.8 kV | Solid | 250 | 6–10 | ZSI |
| F2: Steel Plant | 34.5 kV | Low-R | 500 | 10–15 | 21+87B |
| F3: Petrochemical | 13.8 kV | High-R | 150 | 3–5 | ZSI |
| F4: Pharmaceutical | 4.16 kV | R gnd | 200 | 7–10 | ZSI |
| F5: Cement Plant | 34.5 kV | Low-R | 400 | 8–12 | ZSI |

Each facility models 3 operating configurations (e.g., bus coupler open/closed, single/dual incomer, arc furnace on/off) that alter fault current characteristics and make protection action configuration-dependent.

---

## Repository Structure

```
SecureRelayFL/
├── run_all.sh                  # Run entire experiment pipeline
├── README.md
├── requirements.txt
├── src/
│   └── securerelayfl/
│       ├── __init__.py
│       ├── models/
│       │   ├── __init__.py
│       │   └── classifier.py   # FaultClassifierV2 architecture
│       ├── fl/
│       │   ├── __init__.py
│       │   ├── client.py       # Flower NumPyClient (1.12 API)
│       │   ├── server.py       # FedAvg/FedProx strategy
│       │   └── impairment.py   # Packet loss, quantization, noise
│       ├── experiments/
│       │   ├── __init__.py
│       │   ├── generate_data.py    # Physics-based EMT waveform generator
│       │   ├── train_centralized.py
│       │   ├── train_local.py
│       │   ├── train_fedavg.py
│       │   ├── train_fedprox.py
│       │   ├── sweep_impairments.py  # Axis 1: packet loss × quant × noise
│       │   └── sweep_privacy.py      # Axis 2: DP epsilon sweep
│       └── analysis/
│           ├── __init__.py
│           └── goose_timing.py       # Axis 3: GOOSE HOL analysis
├── data/                       # Generated datasets (per-facility)
├── results/                    # Experiment outputs (JSON + model checkpoints)
├── figures/                    # Publication figures
│   ├── gen_figures.py          # Figure generation script
│   ├── fig_comms.tex           # TikZ: VLAN architecture + HOL timing
│   ├── fig_fl.tex              # TikZ: FL framework overview
│   └── fig_sld.tex             # TikZ: SLD with relay zones + FL data flow
└── paper/
    └── main.tex                # Manuscript (IEEE 2-column, IEEEtran.cls)
```

---

## Quick Start

### Prerequisites

```bash
# Python 3.10+
pip install -r requirements.txt
```

### Run Everything

```bash
# Full pipeline: data generation → baselines → FL → impairments → DP → figures
chmod +x run_all.sh
./run_all.sh
```

### Run Individual Steps

```bash
# 1. Generate synthetic fault waveforms (5 facilities × 3 configs × 1000 samples)
python -m src.securerelayfl.experiments.generate_data

# 2. Baselines
python -m src.securerelayfl.experiments.train_centralized
python -m src.securerelayfl.experiments.train_local

# 3. Federated learning
python -m src.securerelayfl.experiments.train_fedavg      # FedAvg, 50 rounds
python -m src.securerelayfl.experiments.train_fedprox     # FedProx (μ=0.01)

# 4. Axis 1: Network impairment sweep
python -m src.securerelayfl.experiments.sweep_impairments

# 5. Axis 2: Differential privacy sweep
python -m src.securerelayfl.experiments.sweep_privacy

# 6. Axis 3: GOOSE timing analysis
python -m src.securerelayfl.analysis.goose_timing

# 7. Generate publication figures
python figures/gen_figures.py
```

---

## Experimental Configuration

All experiments use identical hyperparameters for reproducibility:

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning rate | 3 × 10⁻⁴ |
| Batch size | 64 |
| FL rounds | 50 |
| Local epochs/round | 1 |
| Clients | 5 (1 per facility) |
| Samples per client | 1,000 |
| Train/val split | 80/20 |
| Random seed | 42 |

### Flower 1.12 Configuration

```python
# Client uses the modern Context API (not legacy cid)
def client_fn(context: Context) -> Client:
    partition_id = context.node_config["partition-id"]
    return FlowerClient(partition_id).to_client()

# Ray backend with CPU-only clients
backend_config = {"num_gpus": 0}
client_resources = {"num_cpus": 1, "num_gpus": 0}
```

### Network Impairment Grid (Axis 1)

| Impairment | Values | Physical Scenario |
|------------|--------|-------------------|
| Packet loss | 0, 5, 10, 15, 25% | EMI → cascading fault → partial failure |
| Quantization | 32, 16, 8 bit | Full precision → shared WAN → satellite |
| Channel noise | 0, 0.001, 0.01 | Clean → lossy compression → analog artifacts |

### Differential Privacy Sweep (Axis 2)

| ε | σ (noise multiplier) | δ |
|---|----------------------|---|
| 0.5 | 11.75 | 4 × 10⁻⁸ |
| 1.0 | 5.875 | 4 × 10⁻⁸ |
| 2.0 | 2.937 | 4 × 10⁻⁸ |
| 5.0 | 1.175 | 4 × 10⁻⁸ |
| 10.0 | 0.587 | 4 × 10⁻⁸ |
| ∞ | 0 (no DP) | — |

---

## Synthetic Data Generation

Fault waveforms follow established standards:

- **Asymmetrical fault current**: IEEE 551-2006, IEC 60909 — DC offset with exponential decay
- **Grounding-dependent voltage**: IEEE C62.92 — voltage swell/sag based on Z₀/Z₁ ratio
- **High-impedance faults**: IEEE PSRC D15 — half-cycle asymmetry, harmonic content (3rd @ 20%, 2nd @ 15%), intermittent arc extinction

Each waveform: 6 channels (Vₐ, Vᵦ, V꜀, Iₐ, Iᵦ, I꜀) × 2,560 samples (10 cycles at 256 samples/cycle, 60 Hz).

Labels: fault type (4 classes) × fault zone (4 classes) × protection action (5 classes).

Configuration features normalized to [0, 1]: `[fault_mva/500, xr_ratio/15, source_count/3]`.

---

## GOOSE Communication Analysis

The dual communication layer analysis (Section VI of the paper) establishes that FL aggregation traffic coexists safely with IEC 61850 GOOSE protection messaging:

**Head-of-line blocking** (worst case, 100 Mbps):

```
τ_HOL = L_MTU / R = (1518 × 8) / 100×10⁶ = 0.121 ms
```

This is **40× below** the IEC 61850-5 Type 1A transfer time requirement of ≤ 3 ms.

**Scheduling recommendations:**
1. 802.1p priority tagging: GOOSE at priority 4, FL at priority 0 (best effort)
2. Post-fault deferral: FL rounds deferred ≥ 30 ms after GOOSE state-change events
3. Rate limiting: ≤ 60 rounds/minute on 100 Mbps links, or use 8-bit quantization to reduce traffic 75%

---

## Citation

If you use this work, please cite:

```bibtex
@misc{ramharack2026securerelayfl_repo,
  author       = {Ramharack, Shankar},
  title        = {SecureRelayFL: Privacy-Preserving Adaptive Relay Settings via Federated Learning Over Industrial Communication Networks},
  year         = {2026},
  howpublished = {\url{https://github.com/sramharack/SecureRelayFL}},
  note         = {GitHub repository}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

This work builds on the [Flower](https://flower.ai/) federated learning framework (v1.12) and uses physics-based waveform models verified against IEEE 551, IEC 60909, IEEE C62.92, and IEEE PSRC D15.
