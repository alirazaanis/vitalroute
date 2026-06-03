"""Benchmark: VitalRoute vs baselines on imbalanced CIFAR-10 with ResNet18.

Creates a long-tail version of CIFAR-10:
  - Majority classes 0-4: 4 000 training samples each
  - Minority classes 5-9:   400 training samples each  (10:1 ratio)

Compares four conditions for 15 epochs:
  1. uniform       — no sampler
  2. inv_freq      — WeightedRandomSampler with inverse class frequency
  3. vitalroute    — TorchTrainingController (vitality class sampler)
  4. focal         — focal loss (gamma=2) with uniform sampling

Metrics: overall val accuracy + minority-class accuracy.

Command:
    python examples/cifar10_resnet_benchmark.py
    python examples/cifar10_resnet_benchmark.py --epochs 20 --trials 2
"""

from __future__ import annotations

import argparse
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from vitalroute.torch_controller import torch_adaptive_controller

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAJORITY_N = 1000
MINORITY_N = 100
NUM_CLASSES = 10
MINORITY = list(range(5, 10))
MAJORITY = list(range(0, 5))


# ── data ─────────────────────────────────────────────────────────────────────

def _get_cifar10():
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262))])
    train = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=True,  download=True, transform=tfm)
    val   = torchvision.datasets.CIFAR10(root="/tmp/cifar10", train=False, download=True, transform=tfm)
    return train, val


def _imbalanced_indices(dataset, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    keep = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(targets == c)
        n = MAJORITY_N if c in MAJORITY else MINORITY_N
        keep.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    return keep


# ── model ─────────────────────────────────────────────────────────────────────

def _make_resnet18() -> nn.Module:
    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES)
    return model.to(DEVICE)


# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        p  = torch.exp(-ce)
        loss = (1 - p) ** self.gamma * ce
        return loss.mean() if self.reduction == "mean" else loss


# ── training ──────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimizer, criterion):
    model.train()
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        criterion(model(X), y).backward()
        optimizer.step()


def _evaluate(model, val_loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            all_preds.append(model(X.to(DEVICE)).argmax(1).cpu())
            all_labels.append(y)
    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    overall = float((preds == labels).mean())
    per_class = np.array([float((preds[labels == c] == c).mean()) if (labels == c).any() else 0.0
                          for c in range(NUM_CLASSES)])
    minority_acc = per_class[MINORITY].mean()
    return overall, minority_acc, per_class


# ── run conditions ────────────────────────────────────────────────────────────

def run_trial(seed: int, epochs: int, batch: int = 128):
    train_full, val_ds = _get_cifar10()
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    idx = _imbalanced_indices(train_full, seed)
    train_sub = Subset(train_full, idx)
    targets_np = np.array(train_full.targets)[idx]

    results = {}

    # ── 1. uniform ────────────────────────────────────────────────────────
    model = _make_resnet18()
    opt   = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)
    for _ in range(epochs):
        _train_epoch(model, loader, opt, nn.CrossEntropyLoss())
        sch.step()
    results["uniform"] = _evaluate(model, val_loader)[:2]

    # ── 2. inv_freq ───────────────────────────────────────────────────────
    model = _make_resnet18()
    opt   = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    counts = np.bincount(targets_np, minlength=NUM_CLASSES).astype(np.float32)
    sample_weights = torch.tensor((1.0 / np.maximum(counts, 1))[targets_np], dtype=torch.float32)
    wrs = WeightedRandomSampler(sample_weights, num_samples=len(idx), replacement=True)
    loader = DataLoader(train_sub, batch_size=batch, sampler=wrs, num_workers=0)
    for _ in range(epochs):
        _train_epoch(model, loader, opt, nn.CrossEntropyLoss())
        sch.step()
    results["inv_freq"] = _evaluate(model, val_loader)[:2]

    # ── 3. focal loss ─────────────────────────────────────────────────────
    model = _make_resnet18()
    opt   = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)
    for _ in range(epochs):
        _train_epoch(model, loader, opt, FocalLoss(gamma=2.0))
        sch.step()
    results["focal"] = _evaluate(model, val_loader)[:2]

    # ── 4. vitalroute ─────────────────────────────────────────────────────
    model = _make_resnet18()
    opt   = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    sch   = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    ctrl = torch_adaptive_controller(targets_np, NUM_CLASSES, verbose=True)
    # Small representative batch seeds the probe (avoids a full-dataset forward pass)
    seed_idx = np.random.default_rng(seed).choice(len(idx), min(512, len(idx)), replace=False)
    X_seed = torch.stack([train_sub[i][0] for i in seed_idx]).to(DEVICE)
    y_seed = torch.tensor(targets_np[seed_idx])

    sampler = ctrl.setup(model, X_seed, y_seed,
                         y_full=y_seed,
                         num_classes=NUM_CLASSES, seed=seed)
    if sampler is not None:
        vr_loader = DataLoader(train_sub, batch_size=batch, sampler=sampler, num_workers=0)
    else:
        vr_loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        ctrl.on_epoch_start(model, X_seed, opt, epoch)
        _train_epoch(model, vr_loader, opt, nn.CrossEntropyLoss())
        sch.step()
        if sampler is not None and epoch % 2 == 0:
            sampler.refresh(X_seed, y_seed, model)
        ctrl.after_epoch(model, X_seed, y_seed)

    ctrl.detach()
    results["vitalroute"] = _evaluate(model, val_loader)[:2]

    return results


def print_table(all_results: list, epochs: int):
    names = list(all_results[0].keys())
    n = len(all_results)
    print(f"\n{'=' * 62}")
    print(f"  CIFAR-10 ResNet18 — {n} trial(s), {epochs} epochs")
    print(f"  Majority 0-4 ({MAJORITY_N} each) | Minority 5-9 ({MINORITY_N} each)  [{MAJORITY_N//MINORITY_N}:1]")
    print(f"{'=' * 62}")
    print(f"{'Method':<16}  {'Overall':>8}  {'Minority':>8}")
    print(f"{'-' * 38}")
    for name in names:
        ov = [all_results[t][name][0] for t in range(n)]
        mn = [all_results[t][name][1] for t in range(n)]
        print(f"{name:<16}  {np.mean(ov):>7.1%}±{np.std(ov):.1%}  {np.mean(mn):>7.1%}±{np.std(mn):.1%}")
    print(f"{'=' * 62}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch",  type=int, default=128)
    args = p.parse_args()

    print(f"Device: {DEVICE}")
    all_results = []
    for seed in range(args.trials):
        print(f"\n--- Trial {seed + 1}/{args.trials} ---")
        all_results.append(run_trial(seed=seed, epochs=args.epochs, batch=args.batch))
    print_table(all_results, args.epochs)


if __name__ == "__main__":
    main()
