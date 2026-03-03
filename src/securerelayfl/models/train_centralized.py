"""
SecureRelayFL — Centralized Baseline Training

Usage:
    python -m securerelayfl.models.train_centralized --model cnn --seed 42
    python -m securerelayfl.models.train_centralized --model tcn --seed 42
    python -m securerelayfl.models.train_centralized --model cnn --facility 0  # local-only
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)

from data.dataset import get_dataloaders
from securerelayfl.models.fault_classifier import FaultClassifier, MultiTaskLoss
from securerelayfl.models.tcn_classifier import FaultClassifierTCN


FAULT_TYPE_NAMES = ["no-fault", "SLG", "LL", "HIF"]
FAULT_ZONE_NAMES = ["N/A", "bus", "near", "far"]
PROTECTION_ACTION_NAMES = ["no-action", "trip-inst", "trip-delay", "ZSI-block", "alarm"]


def build_model(model_name: str, device: str):
    if model_name == "cnn":
        model = FaultClassifier()
    elif model_name == "tcn":
        model = FaultClassifierTCN()
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose 'cnn' or 'tcn'.")
    return model.to(device)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for wf, ft, fz, pa in loader:
        wf = wf.to(device)
        ft, fz, pa = ft.to(device), fz.to(device), pa.to(device)

        optimizer.zero_grad()
        preds = model(wf)
        loss, loss_dict = criterion(preds, ft, fz, pa)
        loss.backward()
        optimizer.step()

        running_loss += loss_dict["total"]
        n_batches += 1

    return running_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    n_batches = 0

    all_ft_true, all_ft_pred = [], []
    all_fz_true, all_fz_pred = [], []
    all_pa_true, all_pa_pred = [], []

    for wf, ft, fz, pa in loader:
        wf = wf.to(device)
        ft, fz, pa = ft.to(device), fz.to(device), pa.to(device)

        preds = model(wf)
        _, loss_dict = criterion(preds, ft, fz, pa)

        running_loss += loss_dict["total"]
        n_batches += 1

        all_ft_true.append(ft.cpu().numpy())
        all_ft_pred.append(preds["fault_type"].argmax(1).cpu().numpy())
        all_fz_true.append(fz.cpu().numpy())
        all_fz_pred.append(preds["fault_zone"].argmax(1).cpu().numpy())
        all_pa_true.append(pa.cpu().numpy())
        all_pa_pred.append(preds["protection_action"].argmax(1).cpu().numpy())

    avg_loss = running_loss / max(n_batches, 1)

    ft_true = np.concatenate(all_ft_true)
    ft_pred = np.concatenate(all_ft_pred)
    fz_true = np.concatenate(all_fz_true)
    fz_pred = np.concatenate(all_fz_pred)
    pa_true = np.concatenate(all_pa_true)
    pa_pred = np.concatenate(all_pa_pred)

    metrics = {
        "loss": avg_loss,
        "fault_type_acc": accuracy_score(ft_true, ft_pred),
        "fault_type_f1": f1_score(ft_true, ft_pred, average="weighted"),
        "fault_zone_acc": accuracy_score(fz_true, fz_pred),
        "fault_zone_f1": f1_score(fz_true, fz_pred, average="weighted"),
        "protection_action_acc": accuracy_score(pa_true, pa_pred),
        "protection_action_f1": f1_score(pa_true, pa_pred, average="weighted"),
    }

    detailed = {
        "fault_type": {
            "true": ft_true, "pred": ft_pred, "names": FAULT_TYPE_NAMES,
        },
        "fault_zone": {
            "true": fz_true, "pred": fz_pred, "names": FAULT_ZONE_NAMES,
        },
        "protection_action": {
            "true": pa_true, "pred": pa_pred, "names": PROTECTION_ACTION_NAMES,
        },
    }
    return metrics, detailed


def main():
    parser = argparse.ArgumentParser(description="SecureRelayFL Centralized Baseline")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "tcn"],
                        help="Model architecture: 'cnn' or 'tcn'")
    parser.add_argument("--data-dir", type=str, default="data/generated")
    parser.add_argument("--facility", type=int, default=None,
                        help="Train on single facility (0-4) for local-only baseline. "
                             "Default: None = all facilities pooled (centralized).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    if args.output_dir is None:
        if args.facility is not None:
            args.output_dir = f"results/{args.model}_local_{args.facility}"
        else:
            args.output_dir = f"results/{args.model}_centralized"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Data ----
    facility_ids = [args.facility] if args.facility is not None else None
    facility_desc = f"facility {args.facility}" if args.facility is not None else "all 5 facilities (pooled)"
    print(f"Loading data: {facility_desc}...")

    train_loader, val_loader, test_loader, dataset = get_dataloaders(
        data_dir=args.data_dir,
        facility_ids=facility_ids,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, "
          f"Test: {len(test_loader.dataset)}")

    # ---- Model ----
    model = build_model(args.model, device)
    criterion = MultiTaskLoss().to(device)
    print(f"  Model: {args.model.upper()} | Params: {model.count_parameters():,} "
          f"({model.model_size_mb():.2f} MB)")

    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- Training loop ----
    best_val_loss = float("inf")
    patience_counter = 0
    history = []

    print(f"\nTraining for up to {args.epochs} epochs (patience={args.patience})...\n")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_ft_acc": val_metrics["fault_type_acc"],
            "val_fz_acc": val_metrics["fault_zone_acc"],
            "val_pa_acc": val_metrics["protection_action_acc"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(record)

        improved = val_metrics["loss"] < best_val_loss
        if improved:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
        else:
            patience_counter += 1

        marker = " *" if improved else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train {train_loss:.4f} | Val {val_metrics['loss']:.4f} | "
            f"FT {val_metrics['fault_type_acc']:.3f} | "
            f"FZ {val_metrics['fault_zone_acc']:.3f} | "
            f"PA {val_metrics['protection_action_acc']:.3f} | "
            f"LR {optimizer.param_groups[0]['lr']:.2e}{marker}"
        )

        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} "
                  f"(no improvement for {args.patience} epochs)")
            break

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.1f}s")

    # ---- Test evaluation ----
    print("\n" + "=" * 70)
    print("TEST SET EVALUATION (best model)")
    print("=" * 70)

    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, "best_model.pt"), weights_only=True)
    )
    test_metrics, detailed = evaluate(model, test_loader, criterion, device)

    for task_name in ["fault_type", "fault_zone", "protection_action"]:
        d = detailed[task_name]
        print(f"\n--- {task_name} ---")
        print(f"  Accuracy: {test_metrics[f'{task_name}_acc']:.4f}")
        print(f"  F1 (weighted): {test_metrics[f'{task_name}_f1']:.4f}")

        present_labels = sorted(set(d["true"]) | set(d["pred"]))
        present_names = [d["names"][i] for i in present_labels if i < len(d["names"])]
        print(classification_report(
            d["true"], d["pred"],
            labels=present_labels,
            target_names=present_names,
            zero_division=0,
        ))

    # ---- Save results ----
    results = {
        "args": vars(args),
        "model_params": model.count_parameters(),
        "model_size_mb": model.model_size_mb(),
        "training_time_s": elapsed,
        "best_epoch": len(history) - patience_counter,
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
        "history": history,
    }
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()