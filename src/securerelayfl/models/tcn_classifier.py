"""
SecureRelayFL — Multi-task Temporal Convolutional Network (TCN).

Architecture:
    Input: (batch, 6, 2560) — 6-channel waveform

    TCN backbone:
        4 residual blocks with dilated causal convolutions
        Dilations: 1, 2, 4, 8
        Each block: Conv1d(causal) → BN → ReLU → Dropout → Conv1d → residual add
        Final: AdaptiveAvgPool1d(1) → (backbone_dim,)

    Task-specific heads (same interface as FaultClassifier):
        fault_type:        backbone_dim → 64 → 4
        fault_zone:        backbone_dim → 64 → 4
        protection_action: backbone_dim → 64 → 5

Design rationale:
    - Causal convolutions match relay logic: decisions based on past samples only
    - Dilated stacks give long memory without recurrence (parallelizable)
    - Stable gradients → cleaner FL convergence curves
    - Moderate param count → practical for FL communication analysis
    - Same forward() output dict as FaultClassifier for drop-in comparison
"""

import torch
import torch.nn as nn


class CausalConv1d(nn.Module):
    """1D convolution with causal (left) padding — no future leakage."""

    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size,
            padding=self.pad, dilation=dilation,
        )

    def forward(self, x):
        out = self.conv(x)
        if self.pad > 0:
            out = out[:, :, :-self.pad]  # Trim right to enforce causality
        return out


class TCNResidualBlock(nn.Module):
    """
    Two-layer causal conv block with residual connection.

    Conv(causal) → BN → ReLU → Dropout → Conv(causal) → BN → ReLU → Dropout
    + residual (1x1 conv if channel mismatch)
    """

    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.downsample = (
            nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.net(x) + self.downsample(x))


class FaultClassifierTCN(nn.Module):
    """Multi-task TCN for fault waveform classification."""

    def __init__(
        self,
        in_channels: int = 6,
        tcn_channels: tuple = (32, 64, 128, 128),
        kernel_size: int = 7,
        dropout: float = 0.2,
        n_fault_types: int = 4,
        n_fault_zones: int = 4,
        n_protection_actions: int = 5,
    ):
        super().__init__()

        layers = []
        ch_in = in_channels
        for i, ch_out in enumerate(tcn_channels):
            dilation = 2 ** i
            layers.append(
                TCNResidualBlock(ch_in, ch_out, kernel_size, dilation, dropout)
            )
            ch_in = ch_out
        self.backbone = nn.Sequential(*layers)

        backbone_dim = tcn_channels[-1]
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.head_fault_type = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_fault_types),
        )

        self.head_fault_zone = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_fault_zones),
        )

        self.head_protection_action = nn.Sequential(
            nn.Linear(backbone_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_protection_actions),
        )

    def forward(self, x):
        """
        x : (batch, 6, 2560) -> dict of logits per task
        """
        features = self.pool(self.backbone(x)).squeeze(-1)
        return {
            "fault_type": self.head_fault_type(features),
            "fault_zone": self.head_fault_zone(features),
            "protection_action": self.head_protection_action(features),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self):
        return self.count_parameters() * 4 / (1024 ** 2)