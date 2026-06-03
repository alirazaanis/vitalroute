"""Benchmark: VitalRoute vs baselines on imbalanced CIFAR-10 with ResNet18.

Long-tail CIFAR-10 setup:
  - Majority classes 0-4: 1000 training samples each
  - Minority classes 5-9:  100 training samples each  (10:1 ratio)

Methods compared: uniform, inv_freq, focal, vitalroute.

Command:
    python examples/cifar10_resnet_benchmark.py
    python examples/cifar10_resnet_benchmark.py --epochs 15 --trials 2 --probe-zone all
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
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.benchmark_cnn_common import (
    FocalLoss,
    evaluate,
    imbalanced_class_indices,
    print_results_table,
    run_vitalroute_condition,
    train_epoch,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAJORITY_N = 1000
MINORITY_N = 100
NUM_CLASSES = 10
MINORITY = list(range(5, 10))
MAJORITY = list(range(0, 5))


def _get_cifar10():
    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])
    root = str(Path(__file__).resolve().parents[1] / ".data" / "cifar10")
    train = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    val = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    return train, val


def _make_resnet18() -> nn.Module:
    return torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES).to(DEVICE)


def run_trial(seed: int, epochs: int, batch: int, probe_zone: str):
    train_full, val_ds = _get_cifar10()
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    targets_all = np.array(train_full.targets)
    idx = imbalanced_class_indices(
        targets_all, NUM_CLASSES, MAJORITY, MAJORITY_N, MINORITY_N, seed
    )
    train_sub = Subset(train_full, idx)
    targets_np = targets_all[idx]
    results = {}

    # uniform
    model = _make_resnet18()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)
    for _ in range(epochs):
        train_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE)
        sch.step()
    results["uniform"] = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)[:2]

    # inv_freq
    model = _make_resnet18()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(targets_np, minlength=NUM_CLASSES).astype(np.float32)
    weights = torch.tensor((1.0 / np.maximum(counts, 1))[targets_np], dtype=torch.float32)
    wrs = WeightedRandomSampler(weights, num_samples=len(idx), replacement=True)
    loader = DataLoader(train_sub, batch_size=batch, sampler=wrs, num_workers=0)
    for _ in range(epochs):
        train_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE)
        sch.step()
    results["inv_freq"] = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)[:2]

    # focal
    model = _make_resnet18()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)
    for _ in range(epochs):
        train_epoch(model, loader, opt, FocalLoss(gamma=2.0), DEVICE)
        sch.step()
    results["focal"] = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)[:2]

    # vitalroute
    model = run_vitalroute_condition(
        model_factory=_make_resnet18,
        train_sub=train_sub,
        targets_np=targets_np,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        epochs=epochs,
        batch=batch,
        seed=seed,
        probe_zone=probe_zone,
    )
    results["vitalroute"] = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)[:2]
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--probe-zone", choices=["head", "trunk", "all"], default="all")
    args = p.parse_args()

    print(f"Device: {DEVICE}  probe_zone={args.probe_zone}")
    all_results = []
    for seed in range(args.trials):
        print(f"\n--- Trial {seed + 1}/{args.trials} ---")
        all_results.append(run_trial(seed, args.epochs, args.batch, args.probe_zone))
    print_results_table(
        all_results,
        "CIFAR-10 ResNet18 long-tail benchmark",
        MAJORITY_N,
        MINORITY_N,
        args.epochs,
    )


if __name__ == "__main__":
    main()
