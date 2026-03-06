"""
SecureRelayFL — Multi-task 1D-CNN with Configuration Embedding + Temporal Attention

Changes from v1 (FaultClassifier):
    - Configuration embedding: 3-dim config features → 32-dim embedding
      concatenated with CNN features before task heads
    - Single-head temporal attention after CNN backbone: lets the model
      weight fault inception vs steady-state fault differently per task
    - Total params: ~125K (only ~13K more than v1)

The config embedding encodes the operating configuration context that makes
protection action a proper function of (waveform, config) rather than a
many-to-many mapping from waveform alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttentionPool(nn.Module):
    """Single-head attention pooling over temporal dimension.

    Instead of global average pooling (which weights all timesteps equally),
    this learns to attend to the most informative part of the waveform
    (e.g., fault inception transient vs steady-state fault).

    Input:  (batch, channels, time)
    Output: (batch, channels)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.query = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, C, T)
        x_t = x.permute(0, 2, 1)            # (batch, T, C)
        scores = self.query(x_t)              # (batch, T, 1)
        weights = F.softmax(scores, dim=1)    # (batch, T, 1)
        pooled = (x_t * weights).sum(dim=1)   # (batch, C)
        return pooled


class FaultClassifierV2(nn.Module):
    """Multi-task 1D-CNN with config embedding + temporal attention.

    Architecture:
        waveform (6, 2560) → CNN backbone → TemporalAttentionPool → (128,)
        config (3,)        → Linear+ReLU                          → (32,)
        concat(128, 32) = (160,) → three task-specific heads
    """

    def __init__(
        self,
        in_channels: int = 6,
        backbone_dim: int = 128,
        config_dim: int = 3,           # fault_mva_norm, xr_ratio_norm, source_count_norm
        config_embed_dim: int = 32,
        n_fault_types: int = 4,
        n_fault_zones: int = 4,
        n_protection_actions: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ---- CNN backbone (same conv layers as v1 for fair comparison) ----
        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, backbone_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, backbone_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Temporal attention pooling (replaces AdaptiveAvgPool1d) ----
        self.attn_pool = TemporalAttentionPool(backbone_dim)

        # ---- Configuration embedding ----
        self.config_embed = nn.Sequential(
            nn.Linear(config_dim, config_embed_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Combined feature dimension ----
        combined_dim = backbone_dim + config_embed_dim  # 128 + 32 = 160

        # ---- Task-specific heads ----
        self.head_fault_type = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_fault_types),
        )

        self.head_fault_zone = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_fault_zones),
        )

        self.head_protection_action = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, n_protection_actions),
        )

    def forward(
        self,
        waveform: torch.Tensor,
        config_features: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            waveform:        (batch, 6, 2560)
            config_features: (batch, 3)  — [fault_mva_norm, xr_ratio_norm, source_count_norm]

        Returns:
            dict with keys 'fault_type', 'fault_zone', 'protection_action',
            each mapping to logits tensor (batch, n_classes).
        """
        # CNN backbone
        conv_out = self.backbone(waveform)          # (batch, 128, T')
        wf_features = self.attn_pool(conv_out)      # (batch, 128)

        # Config embedding
        cfg_features = self.config_embed(config_features)  # (batch, 32)

        # Concatenate
        combined = torch.cat([wf_features, cfg_features], dim=1)  # (batch, 160)

        return {
            "fault_type": self.head_fault_type(combined),
            "fault_zone": self.head_fault_zone(combined),
            "protection_action": self.head_protection_action(combined),
        }

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / (1024 ** 2)