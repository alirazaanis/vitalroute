"""Benchmark: VitalRoute vs baselines on imbalanced Fashion-MNIST (PyTorch).

Uses a small MLP so it finishes in ~2 minutes on CPU.
ResNet18 / CIFAR-10: `examples/cifar10_resnet_benchmark.py` (GPU recommended).

Long-tail setup:
  - Majority classes 0-4: 1 000 samples each
  - Minority classes 5-9:   100 samples each  (10:1 ratio)

Four conditions, 20 epochs:
  1. uniform       — no sampler
  2. inv_freq      — WeightedRandomSampler with inverse class frequency
  3. vitalroute    — TorchTrainingController (vitality class sampler)
  4. focal         — focal loss (gamma=2), uniform sampler

Command:
    python examples/torch_benchmark_fmnist.py
    python examples/torch_benchmark_fmnist.py --epochs 25 --trials 3
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


# ── data ─────────────────────────────────────────────────────────────────────

def _get_fmnist():
    tfm = T.Compose([T.ToTensor(), T.Normalize((0.286,), (0.353,))])
    train = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=True,  download=True, transform=tfm)
    val   = torchvision.datasets.FashionMNIST(root="/tmp/fmnist", train=False, download=True, transform=tfm)
    return train, val


def _imbalanced_indices(dataset, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)
    keep = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(targets == c)
        n = MAJORITY_N if c < 5 else MINORITY_N
        keep.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    return keep


# ── model (small MLP, fast on CPU) ───────────────────────────────────────────

def _make_mlp() -> nn.Module:
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, NUM_CLASSES),
    ).to(DEVICE)


# ── focal loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        p  = torch.exp(-ce)
        return ((1 - p) ** self.gamma * ce).mean()


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
    per_class = np.array(
        [float((preds[labels == c] == c).mean()) if (labels == c).any() else 0.0
         for c in range(NUM_CLASSES)]
    )
    return overall, per_class[MINORITY].mean(), per_class


# ── run trial ─────────────────────────────────────────────────────────────────

def run_trial(seed: int, epochs: int, batch: int = 64):
    train_full, val_ds = _get_fmnist()
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    idx = _imbalanced_indices(train_full, seed)
    train_sub = Subset(train_full, idx)
    targets_np = np.array(train_full.targets)[idx]

    results = {}

    # 1. uniform
    model = _make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        _train_epoch(model, DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0), opt, nn.CrossEntropyLoss())
    results["uniform"] = _evaluate(model, val_loader)[:2]
    print(f"  uniform done: overall={results['uniform'][0]:.3f}  minority={results['uniform'][1]:.3f}")

    # 2. inv_freq
    model = _make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = np.bincount(targets_np, minlength=NUM_CLASSES).astype(np.float32)
    sample_w = torch.tensor((1.0 / np.maximum(counts, 1))[targets_np], dtype=torch.float32)
    wrs = WeightedRandomSampler(sample_w, num_samples=len(idx), replacement=True)
    for _ in range(epochs):
        _train_epoch(model, DataLoader(train_sub, batch_size=batch, sampler=wrs, num_workers=0), opt, nn.CrossEntropyLoss())
    results["inv_freq"] = _evaluate(model, val_loader)[:2]
    print(f"  inv_freq done: overall={results['inv_freq'][0]:.3f}  minority={results['inv_freq'][1]:.3f}")

    # 3. focal
    model = _make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        _train_epoch(model, DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0), opt, FocalLoss())
    results["focal"] = _evaluate(model, val_loader)[:2]
    print(f"  focal done: overall={results['focal'][0]:.3f}  minority={results['focal'][1]:.3f}")

    # 4. vitalroute
    model = _make_mlp()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    ctrl = torch_adaptive_controller(targets_np, NUM_CLASSES, verbose=True)
    # Stratified seed: 50 samples per class so the probe sees all classes
    rng_seed = np.random.default_rng(seed + 42)
    seed_idx = np.concatenate([
        rng_seed.choice(np.flatnonzero(targets_np == c), min(50, (targets_np == c).sum()), replace=False)
        for c in range(NUM_CLASSES)
    ])
    X_seed = torch.stack([train_sub[i][0] for i in seed_idx]).to(DEVICE)
    y_seed = torch.from_numpy(targets_np[seed_idx])

    sampler = ctrl.setup(model, X_seed, y_seed,
                         y_full=torch.from_numpy(targets_np),
                         num_classes=NUM_CLASSES, seed=seed)
    if sampler is not None:
        vr_loader = DataLoader(train_sub, batch_size=batch, sampler=sampler, num_workers=0)
    else:
        vr_loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        ctrl.on_epoch_start(model, X_seed, opt, epoch)
        _train_epoch(model, vr_loader, opt, nn.CrossEntropyLoss())
        ctrl.after_epoch(model, X_seed, y_seed)

    ctrl.detach()
    results["vitalroute"] = _evaluate(model, val_loader)[:2]
    print(f"  vitalroute done: overall={results['vitalroute'][0]:.3f}  minority={results['vitalroute'][1]:.3f}")

    return results


def print_table(all_results: list, epochs: int):
    names = list(all_results[0].keys())
    n = len(all_results)
    print(f"\n{'=' * 62}")
    print(f"  Fashion-MNIST MLP — {n} trial(s), {epochs} epochs")
    print(f"  Majority 0-4 ({MAJORITY_N} each) | Minority 5-9 ({MINORITY_N} each)  [10:1]")
    print(f"{'=' * 62}")
    print(f"{'Method':<16}  {'Overall':>10}  {'Minority':>10}")
    print(f"{'-' * 42}")
    for name in names:
        ov = [all_results[t][name][0] for t in range(n)]
        mn = [all_results[t][name][1] for t in range(n)]
        print(f"{name:<16}  {np.mean(ov):>8.1%}±{np.std(ov):.1%}  {np.mean(mn):>8.1%}±{np.std(mn):.1%}")
    print(f"{'=' * 62}")
    print()
    print("ResNet18 / CIFAR-10 benchmark (GPU recommended):")
    print("  python examples/cifar10_resnet_benchmark.py --epochs 30")
    print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch",  type=int, default=64)
    args = p.parse_args()
    print(f"Device: {DEVICE}  |  {args.trials} trial(s), {args.epochs} epochs")
    all_results = []
    for seed in range(args.trials):
        print(f"\n--- Trial {seed + 1}/{args.trials} ---")
        all_results.append(run_trial(seed=seed, epochs=args.epochs, batch=args.batch))
    print_table(all_results, args.epochs)


if __name__ == "__main__":
    main()
