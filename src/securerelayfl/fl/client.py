"""
SecureRelayFL — Flower FL Client (v2)

v2 changes:
    - Unpacks config_features from dataloader, passes to model
    - Supports both cnn_v1 and cnn_v2 models via model_name parameter
    - FedProx proximal term support (mu > 0)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.dataset import FaultWaveformDataset


def get_model(model_name: str, device: torch.device) -> nn.Module:
    if model_name == "cnn_v1":
        from securerelayfl.models.fault_classifier import FaultClassifier
        return FaultClassifier().to(device)
    elif model_name == "cnn_v2":
        from securerelayfl.models.fault_classifier_v2 import FaultClassifierV2
        return FaultClassifierV2().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def set_parameters(model: nn.Module, parameters: list[np.ndarray]) -> None:
    """Set model parameters from a list of NumPy arrays."""
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = {
        k: torch.from_numpy(np.copy(v)) for k, v in params_dict
    }
    model.load_state_dict(state_dict, strict=True)


def get_parameters(model: nn.Module) -> list[np.ndarray]:
    """Get model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class FaultClient(fl.client.NumPyClient):
    """Flower client for one facility.

    Args:
        facility_id: Which facility this client represents (0-4).
        model_name:  'cnn_v1' or 'cnn_v2'.
        data_dir:    Path to generated data.
        local_epochs: Number of local training epochs per FL round.
        batch_size:  Training batch size.
        lr:          Learning rate.
        fedprox_mu:  FedProx proximal term weight. 0 = pure FedAvg.
        device:      Torch device.
    """

    def __init__(
        self,
        facility_id: int,
        model_name: str = "cnn_v2",
        data_dir: str = "data/generated",
        local_epochs: int = 1,
        batch_size: int = 64,
        lr: float = 3e-4,
        fedprox_mu: float = 0.0,
        device: torch.device | None = None,
    ):
        self.facility_id = facility_id
        self.model_name = model_name
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.fedprox_mu = fedprox_mu
        self.device = device or torch.device("cpu")

        # Model
        self.model = get_model(model_name, self.device)

        # Data — single facility
        dataset = FaultWaveformDataset(
            data_dir=data_dir, facility_ids=[facility_id]
        )

        # Train/val split (80/20)
        n_val = int(len(dataset) * 0.2)
        n_train = len(dataset) - n_val
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(42 + facility_id),
        )
        self.train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, num_workers=0
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False, num_workers=0
        )
        self.n_train = n_train
        self.n_val = n_val

    def get_parameters(self, config: dict[str, Any]) -> list[np.ndarray]:
        return get_parameters(self.model)

    def fit(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[list[np.ndarray], int, dict[str, Any]]:
        """Local training round."""
        set_parameters(self.model, parameters)

        # Save global model for FedProx proximal term
        if self.fedprox_mu > 0:
            global_params = copy.deepcopy(
                [p.data.clone() for p in self.model.parameters()]
            )

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()
        self.model.train()

        for _ in range(self.local_epochs):
            for batch in self.train_loader:
                wf, cf, ft, fz, pa = batch
                wf, cf = wf.to(self.device), cf.to(self.device)
                ft, fz, pa = ft.to(self.device), fz.to(self.device), pa.to(self.device)

                if self.model_name == "cnn_v2":
                    preds = self.model(wf, cf)
                else:
                    preds = self.model(wf)

                loss = (
                    loss_fn(preds["fault_type"], ft)
                    + loss_fn(preds["fault_zone"], fz)
                    + loss_fn(preds["protection_action"], pa)
                )

                # FedProx proximal term
                if self.fedprox_mu > 0:
                    prox = 0.0
                    for p_local, p_global in zip(self.model.parameters(), global_params):
                        prox += ((p_local - p_global) ** 2).sum()
                    loss += (self.fedprox_mu / 2.0) * prox

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return get_parameters(self.model), self.n_train, {}

    def evaluate(
        self, parameters: list[np.ndarray], config: dict[str, Any]
    ) -> tuple[float, int, dict[str, Any]]:
        """Local evaluation."""
        set_parameters(self.model, parameters)
        self.model.eval()
        loss_fn = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct_ft, correct_fz, correct_pa = 0, 0, 0
        total = 0

        with torch.no_grad():
            for batch in self.val_loader:
                wf, cf, ft, fz, pa = batch
                wf, cf = wf.to(self.device), cf.to(self.device)
                ft, fz, pa = ft.to(self.device), fz.to(self.device), pa.to(self.device)

                if self.model_name == "cnn_v2":
                    preds = self.model(wf, cf)
                else:
                    preds = self.model(wf)

                loss = (
                    loss_fn(preds["fault_type"], ft)
                    + loss_fn(preds["fault_zone"], fz)
                    + loss_fn(preds["protection_action"], pa)
                )
                total_loss += loss.item() * ft.size(0)

                correct_ft += (preds["fault_type"].argmax(1) == ft).sum().item()
                correct_fz += (preds["fault_zone"].argmax(1) == fz).sum().item()
                correct_pa += (preds["protection_action"].argmax(1) == pa).sum().item()
                total += ft.size(0)

        metrics = {
            "acc_ft": correct_ft / max(total, 1),
            "acc_fz": correct_fz / max(total, 1),
            "acc_pa": correct_pa / max(total, 1),
            "facility_id": self.facility_id,
        }

        return total_loss / max(total, 1), self.n_val, metrics