# SecureRelayFL

**Privacy-Preserving Adaptive Relay Settings via Federated Learning Over Industrial Communication Networks**

> Paper submitted to [MAIN 2026](https://main-conference.org/) (Mediterranean Artificial Intelligence and Networking Conference), Palermo, Italy, July 1–3, 2026. IEEE ComSoc co-sponsored, IEEE Xplore indexed.

## Overview

Industrial facilities operating heterogeneous electrical protection systems face a data silo problem: optimizing relay settings requires fault data that operators refuse to share. This work applies **federated learning** to train collaborative protection models without centralizing proprietary data, and systematically evaluates how **realistic network impairments** (latency, packet loss, bandwidth) and **differential privacy** affect protection accuracy.

The key insight is a **dual communication layer** problem: FL aggregation traffic shares the same industrial Ethernet carrying time-critical IEC 61850 GOOSE protection messages (sub-4 ms requirement). We analyze how these two communication requirements interact.

## Quick Start

### Prerequisites

- Python 3.10–3.12
- (Optional) CUDA 12.x for GPU training
- ~500 MB disk for generated data

### Setup

```bash
# clone the repo
git clone https://github.com/sramharack/SecureRelayFL.git
cd SecureRelayFL

# create environment and install (pick one)
make env              # loose pins — latest compatible versions
make env-exact        # exact pins — bit-for-bit reproducibility
make env-dev          # loose pins + dev tools (pytest, ruff, jupyter)

# activate
source .venv/bin/activate
```

### Or manually without Make

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel

# option A: exact reproducibility
pip install -r requirements.lock

# option B: latest compatible
pip install -r requirements.txt

# install project as editable package
pip install -e .
```

### Generate Data

```bash
make data                          # default: 1000 samples/facility, seed=42
# or
python data/generator/generate.py --n-samples 1000 --seed 42
```

This creates `data/generated/` with 5,000 synthetic fault waveforms (6 channels × 2,560 timesteps each) across 5 industrial facility profiles.

### Run Tests

```bash
make test
# or
pytest tests/ -v
```

## Project Structure

```
SecureRelayFL/
├── data/
│   └── generator/
│       └── generate.py          # Synthetic fault waveform generator
├── models/                      # 1D-CNN + MLP baseline (TBD)
├── fl/                          # Flower FL server + clients (TBD)
├── experiments/                 # Experiment scripts per axis (TBD)
├── analysis/                    # Publication figure generation (TBD)
├── configs/                     # Experiment YAML configs (TBD)
├── results/                     # Saved metrics & checkpoints (gitignored)
├── tests/                       # Pytest smoke tests
├── pyproject.toml               # Project metadata & deps
├── requirements.txt             # Loose-pin dependencies
├── requirements.lock            # Exact-pin dependencies
├── Makefile                     # One-command workflows
└── README.md
```

## Data Generation

The waveform generator produces physics-based 3-phase voltage and current signals using analytical electromagnetic transient equations. No external simulation tool is required.

### Facility Profiles (5 FL Clients)

| Facility | Voltage | Fault MVA | X/R | Grounding | SNR | Key Protection |
|---|---|---|---|---|---|---|
| Data Center | 13.8 kV | 250 | 8 | Solidly grounded | 40 dB | ZSI (50/51) |
| Steel Plant | 34.5 kV | 500 | 15 | Low-R grounded | 25 dB | Distance (21) + 87B |
| Petrochemical | 13.8 kV | 150 | 6 | High-R grounded | 35 dB | ZSI (50/51) |
| Pharmaceutical | 4.16 kV | 100 | 5 | Resistance grounded | 38 dB | ZSI (50/51) |
| Cement Plant | 34.5 kV | 400 | 12 | Low-R grounded | 28 dB | ZSI (50/51) |

### Waveform Specifications

- **Frequency:** 60 Hz | **Sample rate:** 15,360 Hz | **Channels:** 6 (Va, Vb, Vc, Ia, Ib, Ic)
- **Window:** 10 cycles (3 pre-fault + 7 post-fault) = 2,560 samples
- **Fault types:** No-fault, SLG, LL, high-impedance arcing
- **Labels:** Fault type (4), fault zone (4), protection action (5)

### Physics References

- Asymmetrical fault current: IEEE 551-2006, IEC 60909
- Grounding-dependent voltage sag/swell: IEEE C62.92
- High-impedance arcing faults: IEEE PSRC WG D15

## Experiments

### Axes (priority order)

1. **Network impairment analysis** — latency, packet loss, bandwidth
2. **Privacy–accuracy trade-off** — differential privacy (ε sweep)
3. **Communication efficiency** — gradient compression, GOOSE co-existence
4. **FL strategy comparison** — FedAvg vs. FedProx vs. SCAFFOLD

### Baselines

- Centralized (pooled data) — upper bound
- Local-only (no collaboration) — lower bound
- Federated (ideal network) — isolates impairment effects

## License

MIT

## Citation

```bibtex
@inproceedings{securerelay2026,
  title     = {Privacy-Preserving Adaptive Relay Settings via Federated Learning Over Industrial Communication Networks},
  author    = {Shankar Ramharack},
  booktitle = {Proc. Mediterranean Artificial Intelligence and Networking
               Conference (MAIN)},
  year      = {2026},
  address   = {Palermo, Italy},
}
```
