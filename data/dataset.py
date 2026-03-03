"""
SecureRelayFL — PyTorch Dataset for fault waveforms.

Loads .npy files produced by the generator and serves (waveform, labels) tuples.
Supports loading a single facility or pooling multiple facilities (centralized).
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional


class FaultWaveformDataset(Dataset):
    """
    Loads fault waveform data from .npy files.

    Each sample: (waveform, fault_type, fault_zone, protection_action)
        waveform:          float32 tensor, shape (6, 2560)
        fault_type:        int64, {0: no-fault, 1: SLG, 2: LL, 3: HIF}
        fault_zone:        int64, {0: N/A, 1: bus, 2: near, 3: far}
        protection_action: int64, {0: no-action, 1: trip-instant, 2: trip-delayed,
                                    3: ZSI-block, 4: alarm-only}

    Parameters
    ----------
    data_dir : str or Path
        Root data directory (e.g. "data/generated").
    facility_ids : list[int] or None
        Which facilities to load. None = all (0–4).
    normalize : bool
        If True, z-score normalize each channel independently.
    """

    def __init__(
        self,
        data_dir: str = "data/generated",
        facility_ids: Optional[list[int]] = None,
        normalize: bool = True,
    ):
        data_dir = Path(data_dir)
        if facility_ids is None:
            facility_ids = list(range(5))

        waveforms_list = []
        fault_type_list = []
        fault_zone_list = []
        protection_action_list = []
        facility_label_list = []

        for fid in facility_ids:
            fdir = data_dir / f"facility_{fid}"
            wf = np.load(fdir / "waveforms.npy")            # (N, 6, 2560)
            ft = np.load(fdir / "fault_type.npy")            # (N,)
            fz = np.load(fdir / "fault_zone.npy")            # (N,)
            pa = np.load(fdir / "protection_action.npy")      # (N,)

            waveforms_list.append(wf)
            fault_type_list.append(ft)
            fault_zone_list.append(fz)
            protection_action_list.append(pa)
            facility_label_list.append(np.full(len(wf), fid, dtype=np.int64))

        self.waveforms = np.concatenate(waveforms_list, axis=0)
        self.fault_type = np.concatenate(fault_type_list, axis=0)
        self.fault_zone = np.concatenate(fault_zone_list, axis=0)
        self.protection_action = np.concatenate(protection_action_list, axis=0)
        self.facility_id = np.concatenate(facility_label_list, axis=0)

        # Remap fault_zone: {-1, 0, 1, 2} -> {0, 1, 2, 3} for cross-entropy
        self.fault_zone = self.fault_zone + 1

        if normalize:
            self._normalize()

    def _normalize(self):
        """Z-score normalize each channel across all samples."""
        mean = self.waveforms.mean(axis=(0, 2), keepdims=True)  # (1, 6, 1)
        std = self.waveforms.std(axis=(0, 2), keepdims=True) + 1e-8
        self.waveforms = (self.waveforms - mean) / std
        self.channel_mean = mean.squeeze()
        self.channel_std = std.squeeze()

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        wf = torch.from_numpy(self.waveforms[idx].astype(np.float32))
        ft = torch.tensor(self.fault_type[idx], dtype=torch.long)
        fz = torch.tensor(self.fault_zone[idx], dtype=torch.long)
        pa = torch.tensor(self.protection_action[idx], dtype=torch.long)
        return wf, ft, fz, pa


def get_dataloaders(
    data_dir: str = "data/generated",
    facility_ids: Optional[list[int]] = None,
    batch_size: int = 64,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    num_workers: int = 0,
):
    """
    Create train/val/test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader, dataset
    """
    dataset = FaultWaveformDataset(data_dir, facility_ids, normalize=True)

    n = len(dataset)
    n_test = int(n * test_split)
    n_val = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test], generator=generator
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader, dataset
