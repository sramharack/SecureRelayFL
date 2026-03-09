#!/usr/bin/env python3
"""
SecureRelayFL — Publication Figures
IEEE 2-column style, 300 dpi, clean and minimal.
Matches the aesthetic of the KESTREL EMT validation figures.
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── IEEE Style ──────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'legend.fontsize': 7.5,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
    'grid.linewidth': 0.4,
    'grid.alpha': 0.4,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '0.7',
    'axes.grid': True,
    'grid.linestyle': '--',
})

RESULTS = Path('results')
OUTDIR = Path('figures')
OUTDIR.mkdir(exist_ok=True)

# Color palette — distinct, colorblind-safe
C_CENT = '#1f77b4'   # centralized blue
C_FAVG = '#d62728'   # FedAvg red
C_FPRX = '#ff7f0e'   # FedProx orange
C_LOCAL = '#2ca02c'   # local green
FACILITY_COLORS = ['#1f77b4', '#d62728', '#ff7f0e', '#2ca02c', '#9467bd']
FACILITY_NAMES = ['F1: Data Ctr', 'F2: Steel', 'F3: Petrochem', 'F4: Pharma', 'F5: Cement']


def load_json(path):
    with open(path) as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════
# FIGURE 1: Synthetic EMT Waveforms (6-channel, 3 fault types)
# ═══════════════════════════════════════════════════════════════
def gen_waveform(fault_type, config_mva=500, config_xr=15, f0=60, fs=256*60, cycles=10):
    """Generate physics-based synthetic EMT waveform per IEEE 551 / IEC 60909."""
    t = np.arange(0, cycles / f0, 1/fs)
    omega = 2 * np.pi * f0
    tau = config_xr / omega  # DC decay time constant

    # Base pre-fault voltages (3-phase, 120° apart)
    phases = [0, -2*np.pi/3, 2*np.pi/3]
    V_base = config_mva * 1e3 / np.sqrt(3)  # rough per-phase
    I_base = config_mva * 1e6 / (13.8e3 * np.sqrt(3))

    va, vb, vc = [], [], []
    ia, ib, ic = [], [], []

    if fault_type == '3ph':
        # Three-phase bolted fault
        alpha = np.pi / 4  # inception angle
        for idx, phi in enumerate(phases):
            If = I_base * (0.8 + 0.1 * idx)
            i_sym = np.sqrt(2) * If * np.sin(omega * t + alpha + phi)
            i_dc = -np.sqrt(2) * If * np.sin(alpha + phi) * np.exp(-t / tau)
            i_total = i_sym + i_dc

            # Voltage sag during fault (onset at ~1 cycle)
            fault_onset = 1.0 / f0
            v_pre = np.sqrt(2) * (13.8e3 / np.sqrt(3)) * np.sin(omega * t + phi)
            sag = np.where(t >= fault_onset, 0.15, 1.0)
            v_total = v_pre * sag

            [ia, ib, ic][idx].extend(i_total)
            [va, vb, vc][idx].extend(v_total)

        ia, ib, ic = np.array(ia), np.array(ib), np.array(ic)
        va, vb, vc = np.array(va), np.array(vb), np.array(vc)

    elif fault_type == 'slg':
        # Single line-to-ground, phase A faulted
        alpha = np.pi / 6
        If_a = I_base * 0.6  # reduced by grounding
        ia = np.sqrt(2) * If_a * (np.sin(omega * t + alpha) -
              np.sin(alpha) * np.exp(-t / tau))
        # Unfaulted phases: small current
        ib = 0.05 * I_base * np.sqrt(2) * np.sin(omega * t - 2*np.pi/3)
        ic = 0.05 * I_base * np.sqrt(2) * np.sin(omega * t + 2*np.pi/3)

        fault_onset = 1.0 / f0
        v_nom = np.sqrt(2) * 13.8e3 / np.sqrt(3)
        va = v_nom * np.sin(omega * t) * np.where(t >= fault_onset, 0.10, 1.0)
        vb = v_nom * np.sin(omega * t - 2*np.pi/3) * np.where(t >= fault_onset, 1.45, 1.0)
        vc = v_nom * np.sin(omega * t + 2*np.pi/3) * np.where(t >= fault_onset, 1.45, 1.0)

    elif fault_type == 'hif':
        # High-impedance fault with harmonics
        alpha = np.pi / 3
        If_hif = I_base * 0.02  # very low
        ia_fund = np.sqrt(2) * If_hif * np.sin(omega * t + alpha)
        ia_3rd = 0.20 * np.sqrt(2) * If_hif * np.sin(3 * omega * t + alpha)
        ia_2nd = 0.15 * np.sqrt(2) * If_hif * np.sin(2 * omega * t)
        # Intermittent arc
        arc_mask = (np.sin(omega * t) > 0.3).astype(float)
        arc_mask = arc_mask * (1 + 0.3 * np.random.RandomState(42).randn(len(t)))
        ia = (ia_fund + ia_3rd + ia_2nd) * np.clip(arc_mask, 0, 1.5)
        ib = 0.01 * I_base * np.sqrt(2) * np.sin(omega * t - 2*np.pi/3)
        ic = 0.01 * I_base * np.sqrt(2) * np.sin(omega * t + 2*np.pi/3)

        v_nom = np.sqrt(2) * 13.8e3 / np.sqrt(3)
        va = v_nom * np.sin(omega * t) * 0.97  # tiny sag
        vb = v_nom * np.sin(omega * t - 2*np.pi/3)
        vc = v_nom * np.sin(omega * t + 2*np.pi/3)

    return t * 1e3, np.array([va, vb, vc, ia, ib, ic])  # t in ms


def fig_waveforms():
    """Fig: Synthetic EMT waveforms — 3 fault types, 6 channels each."""
    fig, axes = plt.subplots(3, 2, figsize=(7.16, 5.5), constrained_layout=True)

    fault_types = [('3ph', 'Three-Phase Bolted Fault'),
                   ('slg', 'Single Line-to-Ground Fault'),
                   ('hif', 'High-Impedance Fault')]
    phase_labels = ['$v_a$', '$v_b$', '$v_c$']
    phase_colors = ['#1f77b4', '#d62728', '#2ca02c']

    for row, (ftype, title) in enumerate(fault_types):
        t, channels = gen_waveform(ftype)

        # Voltage panel
        ax_v = axes[row, 0]
        for ch in range(3):
            ax_v.plot(t, channels[ch] / 1e3, color=phase_colors[ch],
                     linewidth=0.8, label=phase_labels[ch])
        ax_v.set_ylabel('Voltage (kV)')
        if row == 0:
            ax_v.legend(loc='upper right', ncol=3, handlelength=1.2,
                       columnspacing=0.8)
        if row == 2:
            ax_v.set_xlabel('Time (ms)')

        # Add fault type label
        ax_v.text(0.02, 0.95, f'({chr(97+row*2)}) {title}',
                 transform=ax_v.transAxes, fontsize=8, fontweight='bold',
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='0.8', alpha=0.9))

        # Current panel
        ax_i = axes[row, 1]
        i_labels = ['$i_a$', '$i_b$', '$i_c$']
        for ch in range(3):
            scale = 1e3 if ftype != 'hif' else 1.0
            unit = 'kA' if ftype != 'hif' else 'A'
            ax_i.plot(t, channels[3+ch] / scale, color=phase_colors[ch],
                     linewidth=0.8, label=i_labels[ch])
        ax_i.set_ylabel(f'Current ({unit})')
        if row == 0:
            ax_i.legend(loc='upper right', ncol=3, handlelength=1.2,
                       columnspacing=0.8)
        if row == 2:
            ax_i.set_xlabel('Time (ms)')

        ax_i.text(0.02, 0.95, f'({chr(98+row*2)})',
                 transform=ax_i.transAxes, fontsize=8, fontweight='bold',
                 va='top', ha='left',
                 bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor='0.8', alpha=0.9))

    fig.savefig(OUTDIR / 'fig_waveforms.png')
    fig.savefig(OUTDIR / 'fig_waveforms.pdf')
    plt.close(fig)
    print('  -> fig_waveforms.png/pdf')


# ═══════════════════════════════════════════════════════════════
# FIGURE 2: FL Convergence Curves (FedAvg vs FedProx vs Centralized)
# ═══════════════════════════════════════════════════════════════
def fig_convergence():
    """Fig: Training convergence — centralized vs FedAvg vs FedProx."""
    cent_hist = load_json(RESULTS / 'cnn_v2_centralized/history.json')
    favg_hist = load_json(RESULTS / 'cnn_v2_fedavg/round_metrics.json')
    fprx_hist = load_json(RESULTS / 'cnn_v2_fedprox/round_metrics.json')

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.4), constrained_layout=True)

    tasks = [('acc_ft', 'Fault Type Accuracy'),
             ('acc_fz', 'Fault Zone Accuracy'),
             ('acc_pa', 'Protection Action Accuracy')]

    for idx, (key, title) in enumerate(tasks):
        ax = axes[idx]

        # Centralized
        epochs = [d['epoch'] for d in cent_hist]
        vals = [d[key] for d in cent_hist]
        ax.plot(epochs, vals, color=C_CENT, linewidth=1.4, label='Centralized')

        # FedAvg
        rounds_fa = [d['round'] for d in favg_hist]
        vals_fa = [d[key] for d in favg_hist]
        ax.plot(rounds_fa, vals_fa, color=C_FAVG, linewidth=1.2, label='FedAvg')

        # FedProx
        rounds_fp = [d['round'] for d in fprx_hist]
        vals_fp = [d[key] for d in fprx_hist]
        ax.plot(rounds_fp, vals_fp, color=C_FPRX, linewidth=1.2,
                linestyle='--', label='FedProx')

        # Local average line
        local_vals = {
            'acc_ft': 0.976, 'acc_fz': 0.977, 'acc_pa': 0.978
        }
        ax.axhline(y=local_vals[key], color=C_LOCAL, linewidth=0.9,
                   linestyle=':', label='Local avg')

        ax.set_xlabel('Epoch / Round')
        ax.set_ylabel('Accuracy')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.set_xlim(0, 50)

        if idx == 0:
            ax.legend(loc='lower right', fontsize=7)

        ax.text(0.02, 0.02, f'({chr(97+idx)})', transform=ax.transAxes,
               fontsize=9, fontweight='bold', va='bottom')

    fig.savefig(OUTDIR / 'fig_convergence.png')
    fig.savefig(OUTDIR / 'fig_convergence.pdf')
    plt.close(fig)
    print('  -> fig_convergence.png/pdf')


# ═══════════════════════════════════════════════════════════════
# FIGURE 3: Network Impairment Heatmap (Axis 1)
# ═══════════════════════════════════════════════════════════════
def fig_impairment():
    """Fig: PA accuracy under network impairments — packet loss × quantization."""
    data = load_json(RESULTS / 'axis1_impairment/all_results.json')

    # Filter to noise_scale=0 entries only
    grid_data = [d for d in data if d['noise_scale'] == 0.0]

    pl_vals = sorted(set(d['packet_loss'] for d in grid_data))
    q_vals = sorted(set(d['quant_bits'] for d in grid_data), reverse=True)

    matrix = np.zeros((len(q_vals), len(pl_vals)))
    for d in grid_data:
        r = q_vals.index(d['quant_bits'])
        c = pl_vals.index(d['packet_loss'])
        matrix[r, c] = d['acc_pa']

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)

    im = ax.imshow(matrix, cmap='RdYlGn', vmin=0.3, vmax=0.75, aspect='auto')

    # Annotate cells
    for i in range(len(q_vals)):
        for j in range(len(pl_vals)):
            val = matrix[i, j]
            color = 'white' if val < 0.45 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                   fontsize=8, fontweight='bold', color=color)

    ax.set_xticks(range(len(pl_vals)))
    ax.set_xticklabels([f'{int(p*100)}%' for p in pl_vals])
    ax.set_yticks(range(len(q_vals)))
    ax.set_yticklabels([f'{q}-bit' for q in q_vals])
    ax.set_xlabel('Packet Loss Rate')
    ax.set_ylabel('Quantization')
    ax.set_title('Protection Action Accuracy', fontsize=9, fontweight='bold')

    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('PA Accuracy', fontsize=8)

    fig.savefig(OUTDIR / 'fig_impairment_heatmap.png')
    fig.savefig(OUTDIR / 'fig_impairment_heatmap.pdf')
    plt.close(fig)
    print('  -> fig_impairment_heatmap.png/pdf')


# ═══════════════════════════════════════════════════════════════
# FIGURE 4: Differential Privacy — Accuracy vs Epsilon
# ═══════════════════════════════════════════════════════════════
def fig_dp():
    """Fig: DP sweep — PA vs epsilon with convergence inset."""
    dp_data = load_json(RESULTS / 'axis2_privacy/all_results.json')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 2.6), constrained_layout=True)

    # (a) Bar chart: PA vs epsilon
    eps_labels = []
    pa_vals = []
    colors = []
    for d in sorted(dp_data, key=lambda x: x['epsilon']):
        eps = d['epsilon']
        if eps == float('inf'):
            eps_labels.append('$\\infty$\n(no DP)')
            colors.append(C_CENT)
        else:
            eps_labels.append(f'$\\varepsilon$={eps}')
            colors.append('#d62728' if d['acc_pa'] < 0.4 else '#ff7f0e')
        pa_vals.append(d['acc_pa'])

    bars = ax1.bar(range(len(pa_vals)), pa_vals, color=colors, edgecolor='black',
                  linewidth=0.5, width=0.7)
    ax1.set_xticks(range(len(pa_vals)))
    ax1.set_xticklabels(eps_labels, fontsize=7.5)
    ax1.set_ylabel('Protection Action Accuracy')
    ax1.set_title('(a) DP Impact on Model Utility', fontsize=9, fontweight='bold')
    ax1.set_ylim(0, 0.85)

    # Add "not viable" annotation
    ax1.axhline(y=0.4, color='gray', linewidth=0.7, linestyle=':')
    ax1.text(2.5, 0.42, 'Random baseline', fontsize=7, color='gray',
            ha='center', style='italic')

    # Annotate values on bars
    for bar, val in zip(bars, pa_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.015,
                f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # (b) Convergence curves for select epsilons
    eps_to_plot = ['inf', '10.0', '5.0', '1.0']
    eps_colors = [C_CENT, '#ff7f0e', '#d62728', '#9467bd']
    eps_labels_leg = ['$\\varepsilon=\\infty$ (no DP)', '$\\varepsilon=10$',
                      '$\\varepsilon=5$', '$\\varepsilon=1$']

    for eps_str, color, label in zip(eps_to_plot, eps_colors, eps_labels_leg):
        rnd = load_json(RESULTS / f'axis2_privacy/eps_{eps_str}/round_metrics.json')
        rounds = [d['round'] for d in rnd]
        pa = [d['acc_pa'] for d in rnd]
        ax2.plot(rounds, pa, color=color, linewidth=1.2, label=label)

    ax2.set_xlabel('FL Round')
    ax2.set_ylabel('Protection Action Accuracy')
    ax2.set_title('(b) Convergence Under DP', fontsize=9, fontweight='bold')
    ax2.set_ylim(0, 0.85)
    ax2.set_xlim(0, 50)
    ax2.legend(loc='upper left', fontsize=7)

    fig.savefig(OUTDIR / 'fig_dp_analysis.png')
    fig.savefig(OUTDIR / 'fig_dp_analysis.pdf')
    plt.close(fig)
    print('  -> fig_dp_analysis.png/pdf')


# ═══════════════════════════════════════════════════════════════
# FIGURE 5: Facility Heterogeneity — Local vs Centralized vs FL
# ═══════════════════════════════════════════════════════════════
def fig_heterogeneity():
    """Fig: Per-facility PA comparison — local, centralized, FedAvg."""
    local_pa = []
    for i in range(5):
        m = load_json(RESULTS / f'cnn_v2_facility_{i}/metrics.json')
        local_pa.append(m['final']['pa'])

    cent_pa = load_json(RESULTS / 'cnn_v2_centralized/metrics.json')['final']['pa']
    favg_pa = 0.665  # Final FedAvg PA (global)

    fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

    x = np.arange(5)
    w = 0.28

    bars1 = ax.bar(x - w, local_pa, w, label='Local only',
                   color=FACILITY_COLORS, edgecolor='black', linewidth=0.5, alpha=0.8)
    bars2 = ax.bar(x, [cent_pa]*5, w, label='Centralized',
                   color=C_CENT, edgecolor='black', linewidth=0.5, alpha=0.6)
    bars3 = ax.bar(x + w, [favg_pa]*5, w, label='FedAvg',
                   color=C_FAVG, edgecolor='black', linewidth=0.5, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(FACILITY_NAMES, fontsize=7, rotation=15)
    ax.set_ylabel('Protection Action Accuracy')
    ax.set_ylim(0.5, 1.05)
    ax.set_title('Per-Facility Accuracy Comparison', fontsize=9, fontweight='bold')
    ax.legend(loc='lower left', fontsize=7, ncol=3)

    # Annotate local PA values
    for bar, val in zip(bars1, local_pa):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
               f'{val:.2f}', ha='center', va='bottom', fontsize=6.5, fontweight='bold')

    fig.savefig(OUTDIR / 'fig_heterogeneity.png')
    fig.savefig(OUTDIR / 'fig_heterogeneity.pdf')
    plt.close(fig)
    print('  -> fig_heterogeneity.png/pdf')


# ═══════════════════════════════════════════════════════════════
# FIGURE 6: GOOSE Timing — Waveform + Protection + Comms Timeline
# ═══════════════════════════════════════════════════════════════
def fig_goose_timing():
    """Fig: Integrated timing diagram — fault waveform, protection response,
    GOOSE messaging, and FL traffic on a shared timeline."""
    fig = plt.figure(figsize=(7.16, 5.0))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1.2, 0.6, 0.8, 0.6],
                 hspace=0.08)

    t_full = np.linspace(0, 200, 4000)  # 0-200 ms
    f0 = 60
    omega = 2 * np.pi * f0

    # Fault occurs at t=16.67ms (1 cycle)
    t_fault = 16.67
    tau = 15 / omega  # X/R=15

    # ── Panel (a): Fault current waveform ──
    ax1 = fig.add_subplot(gs[0])

    If = 20.0  # kA peak (500 MVA system)
    alpha = np.pi / 4
    i_pre = 2.0 * np.sin(omega * t_full/1e3)  # small load current

    i_fault = np.where(t_full >= t_fault,
        np.sqrt(2) * If * (np.sin(omega * (t_full - t_fault)/1e3 + alpha) -
                           np.sin(alpha) * np.exp(-(t_full - t_fault)/1e3 / tau)),
        i_pre)

    ax1.plot(t_full, i_fault, color='#1f77b4', linewidth=0.9)
    ax1.axvline(x=t_fault, color='red', linewidth=0.8, linestyle='--', alpha=0.7)
    ax1.text(t_fault + 1, If * 1.15, 'Fault inception', fontsize=7,
            color='red', fontweight='bold')
    ax1.set_ylabel('Current (kA)')
    ax1.set_xlim(0, 200)
    ax1.set_ylim(-35, 35)
    ax1.tick_params(labelbottom=False)
    ax1.text(0.01, 0.92, '(a) Fault current waveform',
            transform=ax1.transAxes, fontsize=8, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))

    # ── Panel (b): Protection relay response (digital signals) ──
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Pickup at ~20ms (detection delay after inception)
    t_pickup = t_fault + 4.0  # 4ms detection
    t_zsi_block = t_pickup + 1.5  # ZSI blocking signal sent via GOOSE
    t_trip = t_pickup + 50.0  # intentional delay trip (ZSI delayed)
    t_open = t_trip + 50.0  # breaker opening time ~3 cycles

    # Pickup signal
    pickup = np.where(t_full >= t_pickup, 0.9, 0.1)
    ax2.fill_between(t_full, 0, pickup * 0.45, color='#2ca02c', alpha=0.5,
                    step='post', label='Pickup')

    # Trip command
    trip_sig = np.where((t_full >= t_trip) & (t_full < t_open), 0.9, 0.1)
    ax2.fill_between(t_full, 0.5, 0.5 + trip_sig * 0.45, color='#d62728',
                    alpha=0.5, step='post', label='Trip cmd')

    ax2.set_ylabel('Relay')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_yticks([0.25, 0.75])
    ax2.set_yticklabels(['Pickup', 'Trip'], fontsize=7)
    ax2.tick_params(labelbottom=False)
    ax2.text(0.01, 0.92, '(b) Protection relay response',
            transform=ax2.transAxes, fontsize=8, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))

    # Annotate timing
    ax2.annotate('', xy=(t_pickup, 0.1), xytext=(t_trip, 0.1),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.8))
    ax2.text((t_pickup + t_trip)/2, 0.02, 'ZSI delay',
            ha='center', fontsize=6.5, color='gray')

    # ── Panel (c): GOOSE messaging ──
    ax3 = fig.add_subplot(gs[2], sharex=ax1)

    # GOOSE burst: retransmissions with exponential backoff
    goose_times = [t_zsi_block]
    backoff = 2  # ms initial
    for _ in range(7):  # 8 retransmissions
        goose_times.append(goose_times[-1] + backoff)
        backoff = min(backoff * 2, 60)  # exponential backoff, cap at 60ms

    for gt in goose_times:
        ax3.bar(gt, 0.8, width=1.2, color='#d62728', alpha=0.7, edgecolor='black',
               linewidth=0.3)

    # HOL blocking annotation
    ax3.annotate('HOL = 0.121 ms', xy=(goose_times[0], 0.85),
                xytext=(goose_times[0] + 25, 0.9),
                fontsize=7, fontweight='bold', color='#d62728',
                arrowprops=dict(arrowstyle='->', color='#d62728', lw=0.8))

    # FL traffic blocks (background, low priority)
    fl_start_times = [5, 45, 90, 140]
    for fs_t in fl_start_times:
        ax3.bar(fs_t, 0.35, width=30, bottom=-0.4, color='#2ca02c',
               alpha=0.25, edgecolor='#2ca02c', linewidth=0.5)

    ax3.text(60, -0.25, 'FL aggregation (pri. 0, best effort)',
            fontsize=6.5, color='#2ca02c', ha='center')

    ax3.set_ylabel('GOOSE / FL')
    ax3.set_ylim(-0.5, 1.1)
    ax3.set_yticks([0.4, -0.2])
    ax3.set_yticklabels(['GOOSE\n(pri. 4)', 'FL\n(pri. 0)'], fontsize=6.5)
    ax3.tick_params(labelbottom=False)
    ax3.text(0.01, 0.95, '(c) Communication: GOOSE retx burst + FL traffic',
            transform=ax3.transAxes, fontsize=8, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))

    # Retx backoff annotation
    ax3.annotate('', xy=(goose_times[0], 0.95), xytext=(goose_times[-1], 0.95),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=0.6))
    ax3.text((goose_times[0] + goose_times[-1])/2, 1.02,
            'Retx burst (~20 ms)', ha='center', fontsize=6.5, color='gray')

    # ── Panel (d): IEC 61850 timing budget ──
    ax4 = fig.add_subplot(gs[3])

    # Timing budget bar
    budget_items = [
        (0, 0.121, '#d62728', 'HOL blocking\n0.121 ms'),
        (0.121, 0.5, '#ff7f0e', 'Switch processing\n~0.5 ms'),
        (0.621, 2.379, '#2ca02c', 'Available margin\n2.38 ms'),
    ]

    for start, dur, color, label in budget_items:
        ax4.barh(0, dur, left=start, height=0.5, color=color, alpha=0.6,
                edgecolor='black', linewidth=0.5)
        ax4.text(start + dur/2, 0, label, ha='center', va='center',
                fontsize=6, fontweight='bold')

    ax4.axvline(x=3.0, color='black', linewidth=1.5, linestyle='-')
    ax4.text(3.05, 0.35, 'IEC 61850-5\nType 1A: 3 ms', fontsize=7,
            fontweight='bold', va='center')

    ax4.set_xlabel('Time (ms)')
    ax4.set_xlim(-0.1, 5)
    ax4.set_ylim(-0.4, 0.6)
    ax4.set_yticks([])
    ax4.text(0.01, 0.92, '(d) IEC 61850 Type 1A timing budget',
            transform=ax4.transAxes, fontsize=8, fontweight='bold', va='top',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='0.8', alpha=0.9))

    # Panel d has its own x-axis (0-5ms)
    ax4.set_xlim(-0.1, 5)

    # Restore top panels x-axis
    ax1.set_xlim(0, 200)

    fig.savefig(OUTDIR / 'fig_goose_timing.png')
    fig.savefig(OUTDIR / 'fig_goose_timing.pdf')
    plt.close(fig)
    print('  -> fig_goose_timing.png/pdf')


# ═══════════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating SecureRelayFL publication figures...')
    fig_waveforms()
    fig_convergence()
    fig_impairment()
    fig_dp()
    fig_heterogeneity()
    fig_goose_timing()
    print(f'\nAll figures saved to {OUTDIR}/')