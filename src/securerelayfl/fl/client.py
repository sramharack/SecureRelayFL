"""
SecureRelayFL — Flower FL Client

Each client represents one industrial facility, training on its local data only
and communicating model updates to the aggregation server.
"""

import flwr as fl
import torch
import torch.optim as optim
import numpy as np
from collections import OrderedDict

from data.dataset import FaultWaveformDataset
from securerelayfl.models.fault_classifier import FaultClassifier, MultiTaskLoss
from securerelayfl.models.tcn_classifier import FaultClassifierTCN


class RelayClient(fl.client.NumPyClient):
    """Flower client for one industrial facility."""

    def __init__(
        self,
        facility_id: int,
        model_name: str = "cnn",
        data_dir: str = "data/generated",
        local_epochs: int = 3,
        batch_size: int = 64,
        lr: float = 1e-3,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.facility_id = facility_id
        self.device = device
        self.local_epochs = local_epochs
        self.lr = lr

        # ---- Model ----
        if model_name == "cnn":
            self.model = FaultClassifier().to(device)
        elif model_name == "tcn":
            self.model = FaultClassifierTCN().to(device)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        self.criterion = MultiTaskLoss().to(device)

        # ---- Data (this facility only) ----
        dataset = FaultWaveformDataset(
            data_dir=data_dir,
            facility_ids=[facility_id],
            normalize=True,
        )

        n = len(dataset)
        n_val = int(n * 0.15)
        n_train = n - n_val
        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds = torch.utils.data.random_split(
            dataset, [n_train, n_val], generator=generator,
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        )
        self.val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
        )
        self.n_train = n_train
        self.n_val = n_val

    def get_parameters(self, config):
        """Return model parameters as a list of numpy arrays."""
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters):
        """Set model parameters from a list of numpy arrays."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train on local data for local_epochs, return updated parameters."""
        self.set_parameters(parameters)

        # Allow server to override local_epochs via config
        local_epochs = config.get("local_epochs", self.local_epochs)

        optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4,
        )
        self.model.train()

        for _ in range(local_epochs):
            for wf, ft, fz, pa in self.train_loader:
                wf = wf.to(self.device)
                ft = ft.to(self.device)
                fz = fz.to(self.device)
                pa = pa.to(self.device)

                optimizer.zero_grad()
                preds = self.model(wf)
                loss, _ = self.criterion(preds, ft, fz, pa)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config={}), self.n_train, {}

    def evaluate(self, parameters, config):
        """Evaluate on local validation data."""
        self.set_parameters(parameters)
        self.model.eval()

        total_loss = 0.0
        correct_ft, correct_fz, correct_pa = 0, 0, 0
        n_samples = 0

        with torch.no_grad():
            for wf, ft, fz, pa in self.val_loader:
                wf = wf.to(self.device)
                ft = ft.to(self.device)
                fz = fz.to(self.device)
                pa = pa.to(self.device)

                preds = self.model(wf)
                loss, _ = self.criterion(preds, ft, fz, pa)

                batch_size = wf.size(0)
                total_loss += loss.item() * batch_size
                correct_ft += (preds["fault_type"].argmax(1) == ft).sum().item()
                correct_fz += (preds["fault_zone"].argmax(1) == fz).sum().item()
                correct_pa += (preds["protection_action"].argmax(1) == pa).sum().item()
                n_samples += batch_size

        avg_loss = total_loss / max(n_samples, 1)
        metrics = {
            "fault_type_acc": correct_ft / max(n_samples, 1),
            "fault_zone_acc": correct_fz / max(n_samples, 1),
            "protection_action_acc": correct_pa / max(n_samples, 1),
        }
        return float(avg_loss), n_samples, metrics


def client_fn(context) -> fl.client.Client:
    """Factory function for Flower simulation. Reads config from context."""
    cfg = context.run_config
    facility_id = int(context.node_config["partition-id"])

    return RelayClient(
        facility_id=facility_id,
        model_name=cfg.get("model", "cnn"),
        data_dir=cfg.get("data_dir", "data/generated"),
        local_epochs=int(cfg.get("local_epochs", 3)),
        batch_size=int(cfg.get("batch_size", 64)),
        lr=float(cfg.get("lr", 1e-3)),
        device=cfg.get("device", "cpu"),
        seed=int(cfg.get("seed", 42)),
    ).to_client()