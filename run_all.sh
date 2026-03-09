#!/usr/bin/env bash
# SecureRelayFL v2 — Full Paper Results Run
# Estimated wall time: 4-6 hours on CPU
# Run from repo root: bash run_all.sh 2>&1 | tee run_all.log
set -e

SEED=42
N_SAMPLES=1000
EPOCHS=50
ROUNDS=50
LR=3e-4
MODEL=cnn_v2

echo "========================================"
echo "SecureRelayFL v2 — Full Paper Run"
echo "Seed=$SEED  Samples=$N_SAMPLES  Rounds=$ROUNDS"
echo "Started: $(date)"
echo "========================================"

# ── 1. Generate data (~1 min) ──────────────────────────────────
echo ""
echo "[1/8] Generating data..."
python data/generator/generate.py --n-samples $N_SAMPLES --seed $SEED

# ── 2. Centralized baseline (~3 min) ──────────────────────────
echo ""
echo "[2/8] Centralized baseline..."
python -m securerelayfl.models.train_centralized \
    --model $MODEL --epochs $EPOCHS --seed $SEED

# ── 3. Local-only baselines (~10 min) ─────────────────────────
echo ""
echo "[3/8] Local-only baselines..."
for f in 0 1 2 3 4; do
    echo "  Facility $f..."
    python -m securerelayfl.models.train_centralized \
        --model $MODEL --facility $f --epochs $EPOCHS --seed $SEED
done

# ── 4. FedAvg ideal (~30 min) ─────────────────────────────────
echo ""
echo "[4/8] FedAvg ideal..."
python -m securerelayfl.fl.server \
    --model $MODEL --rounds $ROUNDS --local-epochs 1 \
    --lr $LR --seed $SEED

# ── 5. FedProx (~30 min) ──────────────────────────────────────
echo ""
echo "[5/8] FedProx (mu=0.01)..."
python -m securerelayfl.fl.server \
    --model $MODEL --rounds $ROUNDS --local-epochs 1 \
    --lr $LR --fedprox-mu 0.01 --seed $SEED \
    --output-dir results/cnn_v2_fedprox

# ── 6. Axis 1: Impairment sweep (~1.5 hrs) ────────────────────
echo ""
echo "[6/8] Axis 1 — Impairment sweep..."
python -m securerelayfl.experiments.axis1_impairment \
    --model $MODEL --rounds $ROUNDS --lr $LR --seed $SEED

# ── 7. Axis 2: DP sweep (~2 hrs) ──────────────────────────────
echo ""
echo "[7/8] Axis 2 — DP privacy sweep..."
python -m securerelayfl.experiments.axis2_privacy \
    --model $MODEL --rounds $ROUNDS --lr $LR --seed $SEED \
    --n-samples $(( N_SAMPLES * 5 ))

# ── 8. GOOSE dual-layer analysis (~seconds) ───────────────────
echo ""
echo "[8/8] GOOSE dual-layer analysis..."
python -m securerelayfl.analysis.goose_dual_layer

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "All runs complete: $(date)"
echo "========================================"
echo ""
echo "Results summary:"
echo ""

python -c "
import json
from pathlib import Path

def load_metrics(path):
    p = Path(path)
    if (p / 'metrics.json').exists():
        with open(p / 'metrics.json') as f:
            return json.load(f)
    if (p / 'round_metrics.json').exists():
        with open(p / 'round_metrics.json') as f:
            rounds = json.load(f)
            return rounds[-1] if rounds else {}
    return {}

print('─' * 65)
print(f'{\"Setting\":<30} {\"FT Acc\":>8} {\"FZ Acc\":>8} {\"PA Acc\":>8}')
print('─' * 65)

# Centralized
m = load_metrics('results/cnn_v2_centralized')
final = m.get('final', m)
print(f'{\"Centralized\":<30} {final.get(\"acc_ft\",0):>8.3f} {final.get(\"acc_fz\",0):>8.3f} {final.get(\"acc_pa\",0):>8.3f}')

# Local-only
pa_locals = []
for f in range(5):
    m = load_metrics(f'results/cnn_v2_facility_{f}')
    final = m.get('final', m)
    pa = final.get('acc_pa', 0)
    pa_locals.append(pa)
    print(f'{f\"Local F{f}\":<30} {final.get(\"acc_ft\",0):>8.3f} {final.get(\"acc_fz\",0):>8.3f} {pa:>8.3f}')
if pa_locals:
    print(f'{\"Local avg\":<30} {\"\":>8} {\"\":>8} {sum(pa_locals)/len(pa_locals):>8.3f}')

# FedAvg
m = load_metrics('results/cnn_v2_fedavg')
print(f'{\"FedAvg ideal\":<30} {m.get(\"acc_ft\",0):>8.3f} {m.get(\"acc_fz\",0):>8.3f} {m.get(\"acc_pa\",0):>8.3f}')

# FedProx
m = load_metrics('results/cnn_v2_fedprox')
print(f'{\"FedProx (mu=0.01)\":<30} {m.get(\"acc_ft\",0):>8.3f} {m.get(\"acc_fz\",0):>8.3f} {m.get(\"acc_pa\",0):>8.3f}')

print('─' * 65)

# Axis 1 summary
a1 = Path('results/axis1_impairment/all_results.json')
if a1.exists():
    with open(a1) as f:
        results = json.load(f)
    print()
    print('Axis 1 — Impairment Sweep (PA Acc):')
    print(f'{\"Scenario\":<40} {\"Q32\":>6} {\"Q16\":>6} {\"Q8\":>6}')
    print('─' * 60)
    by_pl = {}
    for r in results:
        pl = r['packet_loss']
        qb = r['quant_bits']
        ns = r['noise_scale']
        if ns == 0:
            by_pl.setdefault(pl, {})[qb] = r.get('acc_pa', 0)
    for pl in sorted(by_pl):
        label = r.get('scenario', '') if pl == r.get('packet_loss') else f'{pl:.0%} loss'
        for r2 in results:
            if r2['packet_loss'] == pl and r2['noise_scale'] == 0:
                label = r2.get('scenario', f'{pl:.0%} loss')
                break
        q32 = by_pl[pl].get(32, 0)
        q16 = by_pl[pl].get(16, 0)
        q8 = by_pl[pl].get(8, 0)
        print(f'{label:<40} {q32:>6.3f} {q16:>6.3f} {q8:>6.3f}')

# Axis 2 summary
a2 = Path('results/axis2_privacy/all_results.json')
if a2.exists():
    with open(a2) as f:
        results = json.load(f)
    print()
    print('Axis 2 — DP Privacy Sweep:')
    print(f'{\"Epsilon\":<15} {\"Noise σ\":>10} {\"PA Acc\":>8}')
    print('─' * 35)
    for r in results:
        eps = r['epsilon']
        eps_str = f'{eps:.1f}' if eps != float('inf') else '∞ (no DP)'
        print(f'{eps_str:<15} {r[\"noise_multiplier\"]:>10.4f} {r.get(\"acc_pa\",0):>8.3f}')
"