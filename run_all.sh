#!/usr/bin/env bash
# ============================================================================
# SecureRelayFL — Full Experiment Pipeline
# ============================================================================
# Runs the complete experiment from data generation through figure production.
#
# Usage:
#   chmod +x run_all.sh
#   ./run_all.sh              # Run everything
#   ./run_all.sh --skip-data  # Skip data generation (if data/ already exists)
#   ./run_all.sh --only-figures  # Only regenerate figures from existing results
#
# Requirements:
#   Python 3.10+, packages in requirements.txt
#   ~30 min on a modern CPU (no GPU required)
# ============================================================================

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────
SEED=42
FL_ROUNDS=50
LOCAL_EPOCHS=1
BATCH_SIZE=64
LR=3e-4
N_FACILITIES=5
SAMPLES_PER_FACILITY=1000

# ── Colors for terminal output ─────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'  # No color

log_step() { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; echo -e "${GREEN}[STEP]${NC} $1"; echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }
log_info() { echo -e "${YELLOW}[INFO]${NC} $1"; }
log_done() { echo -e "${GREEN}[DONE]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# ── Parse arguments ────────────────────────────────────────────
SKIP_DATA=false
ONLY_FIGURES=false

for arg in "$@"; do
    case $arg in
        --skip-data) SKIP_DATA=true ;;
        --only-figures) ONLY_FIGURES=true ;;
        --help|-h)
            echo "Usage: ./run_all.sh [--skip-data] [--only-figures]"
            echo "  --skip-data     Skip data generation (use existing data/)"
            echo "  --only-figures  Only regenerate figures from existing results/"
            exit 0 ;;
        *) log_error "Unknown argument: $arg"; exit 1 ;;
    esac
done

# ── Timing ─────────────────────────────────────────────────────
PIPELINE_START=$(date +%s)
step_timer() {
    local start=$1
    local end=$(date +%s)
    local elapsed=$((end - start))
    echo -e "${YELLOW}  Elapsed: ${elapsed}s${NC}"
}

# ── Environment check ──────────────────────────────────────────
log_step "Checking environment"

python3 -c "import torch; print(f'  PyTorch {torch.__version__}')" || {
    log_error "PyTorch not found. Install: pip install -r requirements.txt"
    exit 1
}
python3 -c "import flwr; print(f'  Flower {flwr.__version__}')" || {
    log_error "Flower not found. Install: pip install -r requirements.txt"
    exit 1
}
python3 -c "import matplotlib; print(f'  Matplotlib {matplotlib.__version__}')"

log_info "Device: $(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU')")"
log_info "Seed: ${SEED}"

# ── Create output directories ──────────────────────────────────
mkdir -p data results figures

if [ "$ONLY_FIGURES" = true ]; then
    log_step "Generating figures only (--only-figures)"
    if [ ! -d "results/axis1_impairment" ]; then
        log_error "No results found. Run full pipeline first."
        exit 1
    fi
    python3 figures/gen_figures.py
    log_done "Figures generated in figures/"
    exit 0
fi

# ════════════════════════════════════════════════════════════════
# STEP 1: Data Generation
# ════════════════════════════════════════════════════════════════
if [ "$SKIP_DATA" = false ]; then
    log_step "1/7 — Generating synthetic EMT waveforms"
    log_info "${N_FACILITIES} facilities × 3 configs × ${SAMPLES_PER_FACILITY} samples"
    STEP_START=$(date +%s)

    python3 -m src.securerelayfl.experiments.generate_data \
        --seed ${SEED} \
        --samples ${SAMPLES_PER_FACILITY} \
        --output data/

    step_timer $STEP_START
    log_done "Data saved to data/"
else
    log_info "Skipping data generation (--skip-data)"
fi

# ════════════════════════════════════════════════════════════════
# STEP 2: Centralized Baseline
# ════════════════════════════════════════════════════════════════
log_step "2/7 — Training centralized baseline"
STEP_START=$(date +%s)

python3 -m src.securerelayfl.experiments.train_centralized \
    --data data/ \
    --output results/cnn_v2_centralized \
    --epochs 50 \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED}

step_timer $STEP_START
log_done "Centralized model saved"

# ════════════════════════════════════════════════════════════════
# STEP 3: Local Baselines (per-facility)
# ════════════════════════════════════════════════════════════════
log_step "3/7 — Training local baselines (5 facilities)"
STEP_START=$(date +%s)

for i in $(seq 0 4); do
    log_info "  Facility ${i}..."
    python3 -m src.securerelayfl.experiments.train_local \
        --data data/ \
        --facility ${i} \
        --output results/cnn_v2_facility_${i} \
        --epochs 50 \
        --batch-size ${BATCH_SIZE} \
        --lr ${LR} \
        --seed ${SEED}
done

step_timer $STEP_START
log_done "Local models saved"

# ════════════════════════════════════════════════════════════════
# STEP 4: Federated Learning — FedAvg + FedProx
# ════════════════════════════════════════════════════════════════
log_step "4/7 — Federated learning (FedAvg + FedProx)"
STEP_START=$(date +%s)

log_info "  FedAvg (${FL_ROUNDS} rounds)..."
python3 -m src.securerelayfl.experiments.train_fedavg \
    --data data/ \
    --output results/cnn_v2_fedavg \
    --rounds ${FL_ROUNDS} \
    --local-epochs ${LOCAL_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --seed ${SEED}

log_info "  FedProx (μ=0.01, ${FL_ROUNDS} rounds)..."
python3 -m src.securerelayfl.experiments.train_fedprox \
    --data data/ \
    --output results/cnn_v2_fedprox \
    --rounds ${FL_ROUNDS} \
    --local-epochs ${LOCAL_EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --lr ${LR} \
    --mu 0.01 \
    --seed ${SEED}

step_timer $STEP_START
log_done "FL experiments complete"

# ════════════════════════════════════════════════════════════════
# STEP 5: Axis 1 — Network Impairment Sweep
# ════════════════════════════════════════════════════════════════
log_step "5/7 — Axis 1: Network impairment sweep"
log_info "  Packet loss: [0, 5, 10, 15, 25]%"
log_info "  Quantization: [32, 16, 8] bit"
log_info "  Channel noise: [0, 0.001, 0.01]"
STEP_START=$(date +%s)

python3 -m src.securerelayfl.experiments.sweep_impairments \
    --data data/ \
    --output results/axis1_impairment \
    --rounds ${FL_ROUNDS} \
    --seed ${SEED}

step_timer $STEP_START
log_done "Impairment sweep complete"

# ════════════════════════════════════════════════════════════════
# STEP 6: Axis 2 — Differential Privacy Sweep
# ════════════════════════════════════════════════════════════════
log_step "6/7 — Axis 2: Differential privacy sweep"
log_info "  ε ∈ {0.5, 1.0, 2.0, 5.0, 10.0, ∞}"
STEP_START=$(date +%s)

python3 -m src.securerelayfl.experiments.sweep_privacy \
    --data data/ \
    --output results/axis2_privacy \
    --rounds ${FL_ROUNDS} \
    --seed ${SEED}

step_timer $STEP_START
log_done "DP sweep complete"

# ════════════════════════════════════════════════════════════════
# STEP 7: Generate Publication Figures
# ════════════════════════════════════════════════════════════════
log_step "7/7 — Generating publication figures"
STEP_START=$(date +%s)

python3 figures/gen_figures.py

step_timer $STEP_START
log_done "Figures saved to figures/"

# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
PIPELINE_END=$(date +%s)
TOTAL_ELAPSED=$((PIPELINE_END - PIPELINE_START))

echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  SecureRelayFL — Pipeline Complete${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Total time: ${YELLOW}${TOTAL_ELAPSED}s${NC} ($((TOTAL_ELAPSED/60))m $((TOTAL_ELAPSED%60))s)"
echo ""
echo "  Results:  results/"
echo "  Figures:  figures/"
echo ""
echo "  Key outputs:"
echo "    results/cnn_v2_centralized/    Centralized baseline"
echo "    results/cnn_v2_facility_*/     Local baselines (×5)"
echo "    results/cnn_v2_fedavg/         FedAvg (50 rounds)"
echo "    results/cnn_v2_fedprox/        FedProx (μ=0.01)"
echo "    results/axis1_impairment/      Network impairment sweep"
echo "    results/axis2_privacy/         DP epsilon sweep"
echo ""
echo "    figures/fig_waveforms.pdf      Synthetic EMT waveforms"
echo "    figures/fig_convergence.pdf    FL convergence curves"
echo "    figures/fig_impairment_heatmap.pdf  Packet loss × quantization"
echo "    figures/fig_dp_analysis.pdf    DP impact analysis"
echo "    figures/fig_heterogeneity.pdf  Per-facility comparison"
echo "    figures/fig_goose_timing.pdf   GOOSE + FL timing diagram"
echo ""
