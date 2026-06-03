"""Shared utilities for CNN long-tail benchmark scripts."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from vitalroute.torch_controller import torch_adaptive_controller
from vitalroute.torch_data import stratified_probe_batch


def imbalanced_class_indices(
    targets: np.ndarray,
    num_classes: int,
    majority_classes: List[int],
    majority_n: int,
    minority_n: int,
    seed: int,
) -> List[int]:
    rng = np.random.default_rng(seed)
    keep: List[int] = []
    for c in range(num_classes):
        idx = np.flatnonzero(targets == c)
        n = majority_n if c in majority_classes else minority_n
        keep.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    return keep


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = nn.functional.cross_entropy(logits, targets, reduction="none")
        p = torch.exp(-ce)
        return ((1 - p) ** self.gamma * ce).mean()


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        criterion(model(X), y).backward()
        optimizer.step()


def evaluate(model, val_loader, device, num_classes: int, minority_classes: List[int]):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            preds.append(model(X.to(device)).argmax(1).cpu())
            labels.append(y)
    preds_np = torch.cat(preds).numpy()
    labels_np = torch.cat(labels).numpy()
    overall = float((preds_np == labels_np).mean())
    per_class = np.array([
        float((preds_np[labels_np == c] == c).mean()) if (labels_np == c).any() else 0.0
        for c in range(num_classes)
    ])
    minority_acc = float(per_class[minority_classes].mean())
    return overall, minority_acc, per_class


def run_vitalroute_condition(
    *,
    model_factory: Callable[[], nn.Module],
    train_sub: Subset,
    targets_np: np.ndarray,
    num_classes: int,
    device: str,
    epochs: int,
    batch: int,
    seed: int,
    probe_zone: str = "head",
    parent_pool=None,
    lr: float = 0.05,
) -> nn.Module:
    model = model_factory()
    ctrl = torch_adaptive_controller(
        targets_np,
        num_classes,
        parent_pool=parent_pool,
        architecture="cnn",
        probe_zone=probe_zone,  # type: ignore[arg-type]
        verbose=True,
    )
    X_probe, y_probe, _ = stratified_probe_batch(
        train_sub,
        targets_np,
        per_class=50,
        num_classes=num_classes,
        seed=seed,
        device=device,
    )
    sampler = ctrl.setup(
        model, X_probe, y_probe,
        y_full=targets_np,
        num_classes=num_classes,
        seed=seed,
    )
    opt = ctrl.make_optimizer(
        model, torch.optim.SGD, lr=lr, momentum=0.9, weight_decay=1e-4
    )
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    if sampler is not None:
        loader = DataLoader(train_sub, batch_size=batch, sampler=sampler, num_workers=0)
    else:
        loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)

    for epoch in range(epochs):
        ctrl.on_epoch_start(model, X_probe, opt, epoch)
        train_epoch(model, loader, opt, nn.CrossEntropyLoss(), device)
        sch.step()
        if sampler is not None and epoch % 2 == 0:
            sampler.refresh(X_probe, y_probe, model)
        ctrl.after_epoch(model, X_probe, y_probe)
    ctrl.detach()
    return model


def print_results_table(
    all_results: List[Dict[str, Tuple[float, float]]],
    title: str,
    majority_n: int,
    minority_n: int,
    epochs: int,
) -> None:
    names = list(all_results[0].keys())
    n = len(all_results)
    print(f"\n{'=' * 64}")
    print(f"  {title}")
    print(f"  {n} trial(s), {epochs} epochs  |  {majority_n}:{minority_n} long-tail")
    print(f"{'=' * 64}")
    print(f"{'Method':<18}  {'Overall':>8}  {'Minority':>8}")
    print(f"{'-' * 40}")
    for name in names:
        ov = [all_results[t][name][0] for t in range(n)]
        mn = [all_results[t][name][1] for t in range(n)]
        print(f"{name:<18}  {np.mean(ov):>7.1%}±{np.std(ov):.1%}  {np.mean(mn):>7.1%}±{np.std(mn):.1%}")
    print(f"{'=' * 64}\n")
