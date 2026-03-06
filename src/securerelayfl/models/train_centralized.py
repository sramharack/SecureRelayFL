"""
SecureRelayFL — Centralized training baseline (v2)

Supports both v1 (FaultClassifier) and v2 (FaultClassifierV2) models.
v2 adds config_features as a model input.

Usage:
    # Centralized (all facilities)
    python -m securerelayfl.models.train_centralized --model cnn_v2 --seed 42

    # Local-only (single facility)
    python -m securerelayfl.models.train_centralized --model cnn_v2 --facility 0 --seed 42
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

# Relative imports when run as module
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from data.dataset import FaultWaveformDataset


def get_model(model_name: str, device: torch.device) -> nn.Module:
    """Instantiate model by name."""
    if model_name == "cnn_v1":
        from securerelayfl.models.fault_classifier import FaultClassifier
        return FaultClassifier().to(device)
    elif model_name == "cnn_v2":
        from securerelayfl.models.fault_classifier_v2 import FaultClassifierV2
        return FaultClassifierV2().to(device)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
) -> dict[str, float]:
    """Train for one epoch, return average losses."""
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    running = {"ft": 0.0, "fz": 0.0, "pa": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        wf, cf, ft, fz, pa = batch
        wf, cf = wf.to(device), cf.to(device)
        ft, fz, pa = ft.to(device), fz.to(device), pa.to(device)

        # Forward — v2 passes config features, v1 ignores them
        if model_name == "cnn_v2":
            preds = model(wf, cf)
        else:
            preds = model(wf)

        loss_ft = loss_fn(preds["fault_type"], ft)
        loss_fz = loss_fn(preds["fault_zone"], fz)
        loss_pa = loss_fn(preds["protection_action"], pa)
        loss = loss_ft + loss_fz + loss_pa

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running["ft"] += loss_ft.item()
        running["fz"] += loss_fz.item()
        running["pa"] += loss_pa.item()
        running["total"] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    model_name: str,
) -> dict[str, float]:
    """Evaluate model, return accuracy per task."""
    model.eval()
    correct = {"ft": 0, "fz": 0, "pa": 0}
    total = 0

    for batch in loader:
        wf, cf, ft, fz, pa = batch
        wf, cf = wf.to(device), cf.to(device)
        ft, fz, pa = ft.to(device), fz.to(device), pa.to(device)

        if model_name == "cnn_v2":
            preds = model(wf, cf)
        else:
            preds = model(wf)

        correct["ft"] += (preds["fault_type"].argmax(1) == ft).sum().item()
        correct["fz"] += (preds["fault_zone"].argmax(1) == fz).sum().item()
        correct["pa"] += (preds["protection_action"].argmax(1) == pa).sum().item()
        total += ft.size(0)

    return {k: v / max(total, 1) for k, v in correct.items()}


def main():
    parser = argparse.ArgumentParser(description="Centralized training baseline")
    parser.add_argument("--model", type=str, default="cnn_v2", choices=["cnn_v1", "cnn_v2"])
    parser.add_argument("--facility", type=int, default=None,
                        help="Train on single facility (local-only baseline). None = all.")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    fids = [args.facility] if args.facility is not None else list(range(5))
    dataset = FaultWaveformDataset(data_dir=args.data_dir, facility_ids=fids)

    # Train/val split (80/20)
    n_val = int(len(dataset) * 0.2)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Model
    model = get_model(args.model, device)
    print(f"Model: {args.model} | Params: {model.count_parameters():,} | "
          f"Size: {model.model_size_mb():.2f} MB")
    print(f"Facilities: {fids} | Train: {n_train} | Val: {n_val}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Output directory
    if args.output_dir is None:
        tag = f"facility_{args.facility}" if args.facility is not None else "centralized"
        args.output_dir = f"results/{args.model}_{tag}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_pa = 0.0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        losses = train_one_epoch(model, train_loader, optimizer, device, args.model)
        accs = evaluate(model, val_loader, device, args.model)
        scheduler.step()

        row = {"epoch": epoch, **{f"loss_{k}": v for k, v in losses.items()}, **{f"acc_{k}": v for k, v in accs.items()}}
        history.append(row)

        if accs["pa"] > best_pa:
            best_pa = accs["pa"]
            torch.save(model.state_dict(), out_dir / "best_model.pt")

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}  |  loss={losses['total']:.4f}  |  "
                  f"FT={accs['ft']:.3f}  FZ={accs['fz']:.3f}  PA={accs['pa']:.3f}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Best PA acc: {best_pa:.4f}")
    print(f"Saved to {out_dir}")

    # Save history
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save final metrics
    final_accs = evaluate(model, val_loader, device, args.model)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"model": args.model, "facilities": fids, "best_pa": best_pa, "final": final_accs}, f, indent=2)


if __name__ == "__main__":
    main()