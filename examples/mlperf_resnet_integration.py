"""MLPerf-style ResNet training with VitalRoute callback.

Demonstrates ``VitalRouteMLPerfCallback`` on imbalanced CIFAR-10. Tags from
``log_mlperf_tags()`` follow MLPerf result-metadata conventions.

Command:
    python examples/mlperf_resnet_integration.py
    python examples/mlperf_resnet_integration.py --epochs 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.benchmark_cnn_common import evaluate, imbalanced_class_indices, train_epoch
from vitalroute.mlperf_hooks import VitalRouteMLPerfCallback

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 10
MINORITY = list(range(5, 10))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=128)
    args = p.parse_args()

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.491, 0.482, 0.447), (0.247, 0.243, 0.262)),
    ])
    root = str(Path(__file__).resolve().parents[1] / ".data" / "cifar10")
    train_full = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=tfm)
    val_ds = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=tfm)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

    targets_all = np.array(train_full.targets)
    idx = imbalanced_class_indices(targets_all, NUM_CLASSES, list(range(5)), 1000, 100, seed=0)
    train_sub = Subset(train_full, idx)
    targets_np = targets_all[idx]

    model = torchvision.models.resnet18(weights=None, num_classes=NUM_CLASSES).to(DEVICE)

    callback = VitalRouteMLPerfCallback(
        targets_np,
        NUM_CLASSES,
        architecture="cnn",
        probe_zone="head",
        per_class_probe=50,
        batch_size=args.batch,
        verbose=True,
    )
    callback.before_train(model, train_sub, device=DEVICE, labels=targets_np)
    optimizer = callback.make_optimizer(
        model, torch.optim.SGD, lr=0.05, momentum=0.9, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print("\nMLPerf tags:", json.dumps(callback.log_mlperf_tags(), indent=2))
    print(f"Route: {callback.routing_label}\n")

    for epoch in range(args.epochs):
        callback.on_epoch_begin(epoch, model, optimizer)
        loader = callback.get_train_loader(train_sub)
        train_epoch(model, loader, optimizer, nn.CrossEntropyLoss(), DEVICE)
        scheduler.step()
        metrics = callback.on_epoch_end(epoch, model)
        print(f"  epoch {epoch}: mean_stasis={metrics.get('mean_stasis', 0):.4f}")

    callback.after_train()
    overall, minority, _ = evaluate(model, val_loader, DEVICE, NUM_CLASSES, MINORITY)
    print(f"\nVal accuracy: overall={overall:.1%}  minority={minority:.1%}")


if __name__ == "__main__":
    main()
