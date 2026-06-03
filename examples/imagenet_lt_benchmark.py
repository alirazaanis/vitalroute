"""ImageNet-LT style benchmark.

Uses CIFAR-100 as a local dataset when ``--data-dir`` is omitted. Pass
``--data-dir /path/to/imagenet/train`` for ImageNet-LT with ResNet50.

Command:
    python examples/imagenet_lt_benchmark.py --backend cifar100 --epochs 10
    python examples/imagenet_lt_benchmark.py --data-dir /data/imagenet/train --model resnet50
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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets import ImageFolder

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


def _build_datasets(backend: str, data_dir: str | None):
    if backend == "imagenet":
        if not data_dir:
            raise ValueError("--data-dir required for imagenet backend")
        tfm = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        val_tfm = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        train = ImageFolder(data_dir, transform=tfm)
        val_root = str(Path(data_dir).parent / "val")
        val = ImageFolder(val_root, transform=val_tfm) if Path(val_root).exists() else None
        num_classes = len(train.classes)
        return train, val, num_classes

    tfm = T.Compose([
        T.ToTensor(),
        T.Normalize((0.507, 0.487, 0.441), (0.267, 0.256, 0.276)),
    ])
    root = str(Path(__file__).resolve().parents[1] / ".data" / "cifar100")
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=True, transform=tfm)
    val = torchvision.datasets.CIFAR100(root=root, train=False, download=True, transform=tfm)
    return train, val, 100


def _make_model(name: str, num_classes: int) -> nn.Module:
    if name == "resnet50":
        m = torchvision.models.resnet50(weights=None, num_classes=num_classes)
    else:
        m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    return m.to(DEVICE)


def run_trial(
    backend: str,
    data_dir: str | None,
    model_name: str,
    seed: int,
    epochs: int,
    batch: int,
    majority_n: int,
    minority_n: int,
    probe_zone: str,
):
    train_full, val_ds, num_classes = _build_datasets(backend, data_dir)
    if val_ds is None:
        raise RuntimeError("Validation set not found for ImageNet backend")

    val_loader = DataLoader(val_ds, batch_size=min(batch * 2, 256), shuffle=False, num_workers=0)
    targets_all = np.array(getattr(train_full, "targets", None) or [s[1] for s in train_full.samples])
    half = num_classes // 2
    majority = list(range(half))
    minority = list(range(half, num_classes))

    idx = imbalanced_class_indices(
        targets_all, num_classes, majority, majority_n, minority_n, seed
    )
    train_sub: Dataset = Subset(train_full, idx)
    targets_np = targets_all[idx]
    results = {}

    def factory():
        return _make_model(model_name, num_classes)

    # uniform
    model = factory()
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=1e-4)
    loader = DataLoader(train_sub, batch_size=batch, shuffle=True, num_workers=0)
    for _ in range(epochs):
        train_epoch(model, loader, opt, nn.CrossEntropyLoss(), DEVICE)
    results["uniform"] = evaluate(model, val_loader, DEVICE, num_classes, minority)[:2]

    # vitalroute
    model = run_vitalroute_condition(
        model_factory=factory,
        train_sub=train_sub,
        targets_np=targets_np,
        num_classes=num_classes,
        device=DEVICE,
        epochs=epochs,
        batch=batch,
        seed=seed,
        probe_zone=probe_zone,
        lr=0.05 if backend == "cifar100" else 0.1,
    )
    results["vitalroute"] = evaluate(model, val_loader, DEVICE, num_classes, minority)[:2]

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cifar100", "imagenet"], default="cifar100")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--trials", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--majority-n", type=int, default=500)
    p.add_argument("--minority-n", type=int, default=50)
    p.add_argument("--probe-zone", choices=["head", "trunk", "all"], default="all")
    args = p.parse_args()

    print(f"Device: {DEVICE}  backend={args.backend}  model={args.model}")
    all_results = []
    for seed in range(args.trials):
        print(f"\n--- Trial {seed + 1}/{args.trials} ---")
        all_results.append(
            run_trial(
                args.backend,
                args.data_dir,
                args.model,
                seed,
                args.epochs,
                args.batch,
                args.majority_n,
                args.minority_n,
                args.probe_zone,
            )
        )
    title = f"{'ImageNet-LT' if args.backend == 'imagenet' else 'CIFAR-100-LT'} ({args.model})"
    print_results_table(all_results, title, args.majority_n, args.minority_n, args.epochs)


if __name__ == "__main__":
    main()
