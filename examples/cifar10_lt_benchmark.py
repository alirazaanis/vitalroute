"""CIFAR-10 long-tail benchmark with transfer-pick comparison.

Includes a scarce-data trial (200 samples) comparing cold start, manual transfer,
and VitalRoute controller with a parent pool.

Command:
    python examples/cifar10_lt_benchmark.py --mode full
    python examples/cifar10_lt_benchmark.py --mode transfer --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.benchmark_cnn_common import (
    evaluate,
    imbalanced_class_indices,
    print_results_table,
    run_vitalroute_condition,
    train_epoch,
)
from examples.cifar10_resnet_benchmark import (
    DEVICE,
    MAJORITY,
    MAJORITY_N,
    MINORITY,
    MINORITY_N,
    NUM_CLASSES,
    _get_cifar10,
    _make_resnet18,
)
from vitalroute.torch_transfer import pick_transfer_parent_torch, warm_start_from_parent
from vitalroute.torch_data import stratified_probe_batch


def _pretrain_parent(epochs: int = 5) -> nn.Module:
    train_full, _ = _get_cifar10()
    model = _make_resnet18()
    loader = DataLoader(train_full, batch_size=128, shuffle=True, num_workers=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    for _ in range(epochs):
        train_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE)
    return model


def run_full_benchmark(epochs: int, trials: int, batch: int, probe_zone: str):
    from examples.cifar10_resnet_benchmark import run_trial

    all_results = []
    for seed in range(trials):
        print(f"\n--- Full LT trial {seed + 1}/{trials} ---")
        all_results.append(run_trial(seed, epochs, batch, probe_zone))
    print_results_table(
        all_results,
        "CIFAR-10-LT full benchmark",
        MAJORITY_N,
        MINORITY_N,
        epochs,
    )


def run_transfer_benchmark(epochs: int, seed: int, n_scarce: int = 200):
    train_full, val_ds = _get_cifar10()
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    rng = np.random.default_rng(seed)
    all_idx = rng.choice(len(train_full), n_scarce, replace=False)
    train_sub = Subset(train_full, all_idx.tolist())
    targets_np = np.array(train_full.targets)[all_idx]

    X_probe, y_probe, _ = stratified_probe_batch(
        train_sub, targets_np, per_class=20, num_classes=NUM_CLASSES, seed=seed, device=DEVICE
    )

    print("Pretraining parent pool …")
    parent_a = _pretrain_parent(epochs=3)
    parent_b = _pretrain_parent(epochs=8)
    pool = [("parent_short", parent_a), ("parent_long", parent_b)]

    name, parent, scores = pick_transfer_parent_torch(pool, X_probe, probe_zone="head")
    print(f"Transfer pick: {name}  stasis={scores[0].stasis:.4f}")

    results = {}
    # cold start
    model = _make_resnet18()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    loader = DataLoader(train_sub, batch_size=64, shuffle=True, num_workers=0)
    for _ in range(epochs):
        train_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE)
    results["cold_start"] = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)[:2]

    # manual transfer warm-start
    model = _make_resnet18()
    warm_start_from_parent(model, parent, fresh_head=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    for _ in range(epochs):
        train_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE)
    results[f"transfer_{name}"] = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)[:2]

    # vitalroute controller with parent pool
    model = run_vitalroute_condition(
        model_factory=_make_resnet18,
        train_sub=train_sub,
        targets_np=targets_np,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        epochs=epochs,
        batch=64,
        seed=seed,
        probe_zone="head",
        parent_pool=pool,
    )
    results["vitalroute_transfer"] = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)[:2]

    print_results_table([results], f"CIFAR-10 scarce ({n_scarce} samples)", 1, 1, epochs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["full", "transfer", "both"], default="both")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--probe-zone", choices=["head", "trunk", "all"], default="all")
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    if args.mode in ("full", "both"):
        run_full_benchmark(args.epochs, args.trials, args.batch, args.probe_zone)
    if args.mode in ("transfer", "both"):
        run_transfer_benchmark(args.epochs, seed=0)


if __name__ == "__main__":
    main()
