"""
SecureRelayFL — Multi-task 1D-CNN for fault waveform classification.

Architecture:
    Input: (batch, 6, 2560) — 6-channel waveform, 10 cycles at 256 samples/cycle

    Shared backbone (1D-CNN):
        Conv1d(6, 32, k=7, s=2)  → BatchNorm → ReLU → MaxPool(2)  → (32, 638)
        Conv1d(32, 64, k=5, s=2) → BatchNorm → ReLU → MaxPool(2)  → (64, 158)
        Conv1d(64, 128, k=3, s=1)→ BatchNorm → ReLU → MaxPool(2)  → (128, 78)
        Conv1d(128, 128, k=3, s=1)→ BatchNorm → ReLU → AdaptiveAvgPool(1) → (128,)

    Task-specific heads (MLP each):
        fault_type:        128 → 64 → 4
        fault_zone:        128 → 64 → 4   (N/A, bus, near, far)
        protection_action: 128 → 64 → 5

Design rationale:
    - Lightweight (~200K params, ~800 KB per FL round)
    - Conv layers learn temporal fault signatures (DC offset, harmonics, arc patterns)
    - Shared backbone + separate heads = standard multi-task approach
    - BatchNorm for stable training across heterogeneous facility data
"""

import torch
import torch.nn as nn


class FaultClassifier(nn.Module):
    """Multi-task 1D-CNN for fault waveform classification."""

    def __init__(
        self,
        in_channels: int = 6,
        backbone_dim: int = 128,
        n_fault_types: int = 4,
        n_fault_zones: int = 4,
        n_protection_actions: int = 5,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(128, backbone_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(backbone_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

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
        features = self.backbone(x).squeeze(-1)  # (batch, backbone_dim)
        return {
            "fault_type": self.head_fault_type(features),
            "fault_zone": self.head_fault_zone(features),
            "protection_action": self.head_protection_action(features),
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self):
        return self.count_parameters() * 4 / (1024 ** 2)


class MultiTaskLoss(nn.Module):
    """
    Weighted cross-entropy loss across three classification tasks.
    Loss = w1*CE(fault_type) + w2*CE(fault_zone) + w3*CE(protection_action)
    """

    def __init__(self, weights: tuple = (1.0, 0.5, 1.0)):
        super().__init__()
        self.w_ft, self.w_fz, self.w_pa = weights
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: dict, fault_type, fault_zone, protection_action):
        loss_ft = self.ce(preds["fault_type"], fault_type)
        loss_fz = self.ce(preds["fault_zone"], fault_zone)
        loss_pa = self.ce(preds["protection_action"], protection_action)

        total = self.w_ft * loss_ft + self.w_fz * loss_fz + self.w_pa * loss_pa
        return total, {
            "fault_type": loss_ft.item(),
            "fault_zone": loss_fz.item(),
            "protection_action": loss_pa.item(),
            "total": total.item(),
        }


def build_model(device: str = "cpu"):
    """Convenience: create model + loss, move to device."""
    model = FaultClassifier().to(device)
    criterion = MultiTaskLoss().to(device)
    return model, criterion
