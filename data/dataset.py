"""
SecureRelayFL — PyTorch Dataset (v2)

v2 changes:
    - Loads config_features.npy (N, 3) alongside existing arrays
    - __getitem__ returns (waveform, config_features, fault_type, fault_zone, protection_action)
    - Backward-compatible: if config_features.npy missing, returns zeros
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class FaultWaveformDataset(Dataset):
    """Multi-facility fault waveform dataset with operating configuration features.

    Each sample is a tuple:
        waveform:        (6, 2560) float32 — 3-phase V + I
        config_features: (3,)     float32 — [fault_mva_norm, xr_ratio_norm, source_count_norm]
        fault_type:      int64    — {0: no_fault, 1: SLG, 2: LL, 3: HIF}
        fault_zone:      int64    — {0: bus, 1: near, 2: far, 3: none}
        protection_action: int64  — {0: no_action, 1: trip_inst, 2: trip_delayed,
                                      3: ZSI_block, 4: alarm_only}
    """

    def __init__(
        self,
        data_dir: str | Path = "data/generated",
        facility_ids: list[int] | None = None,
        normalize_waveforms: bool = True,
    ):
        data_dir = Path(data_dir)
        if facility_ids is None:
            facility_ids = list(range(5))

        wf_list, cf_list, ft_list, fz_list, pa_list = [], [], [], [], []

        for fid in facility_ids:
            fdir = data_dir / f"facility_{fid}"
            if not fdir.exists():
                raise FileNotFoundError(f"Facility directory not found: {fdir}")

            wf_list.append(np.load(fdir / "waveforms.npy"))
            ft_list.append(np.load(fdir / "fault_type.npy"))
            fz_list.append(np.load(fdir / "fault_zone.npy"))
            pa_list.append(np.load(fdir / "protection_action.npy"))

            # v2: config features (backward-compatible fallback)
            cf_path = fdir / "config_features.npy"
            if cf_path.exists():
                cf_list.append(np.load(cf_path))
            else:
                # Fallback for v1 data: zeros
                n = wf_list[-1].shape[0]
                cf_list.append(np.zeros((n, 3), dtype=np.float32))

        self.waveforms = np.concatenate(wf_list, axis=0)        # (N, 6, 2560)
        self.config_features = np.concatenate(cf_list, axis=0)   # (N, 3)
        self.fault_type = np.concatenate(ft_list, axis=0)        # (N,)
        self.fault_zone = np.concatenate(fz_list, axis=0)        # (N,)
        self.protection_action = np.concatenate(pa_list, axis=0) # (N,)

        # Per-channel normalization (zero-mean, unit-variance)
        if normalize_waveforms:
            mean = self.waveforms.mean(axis=(0, 2), keepdims=True)  # (1, 6, 1)
            std = self.waveforms.std(axis=(0, 2), keepdims=True) + 1e-8
            self.waveforms = (self.waveforms - mean) / std

    def __len__(self) -> int:
        return len(self.fault_type)

    def __getitem__(self, idx: int):
        wf = torch.from_numpy(self.waveforms[idx].astype(np.float32))
        cf = torch.from_numpy(self.config_features[idx].astype(np.float32))
        ft = torch.tensor(self.fault_type[idx], dtype=torch.long)
        fz = torch.tensor(self.fault_zone[idx], dtype=torch.long)
        pa = torch.tensor(self.protection_action[idx], dtype=torch.long)
        return wf, cf, ft, fz, pa