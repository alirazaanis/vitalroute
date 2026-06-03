"""Per-layer optimizer param groups for vitality-scaled learning rates."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Type

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_lr_scale. "
        "Install it with: pip install torch"
    ) from exc

from .torch_probe import VitalityProbe


def _named_modules_map(model: nn.Module) -> Dict[str, nn.Module]:
    return dict(model.named_modules())


def params_for_layer(model: nn.Module, layer_name: str) -> List[torch.nn.Parameter]:
    mod = _named_modules_map(model).get(layer_name)
    if mod is None:
        return []
    return list(mod.parameters())


def build_layer_param_groups(
    model: nn.Module,
    probe: VitalityProbe,
    base_lr: float,
    *,
    weight_decay: float = 0.0,
) -> List[dict]:
    """One optimizer param group per probed layer; remainder in ``_rest``."""
    covered: set = set()
    groups: List[dict] = []
    layer_names = probe.layer_names()

    for name in layer_names:
        params = params_for_layer(model, name)
        if not params:
            continue
        for p in params:
            covered.add(id(p))
        groups.append({
            "params": params,
            "lr": base_lr,
            "base_lr": base_lr,
            "name": name,
            "weight_decay": weight_decay,
        })

    rest = [p for p in model.parameters() if id(p) not in covered]
    if rest:
        groups.append({
            "params": rest,
            "lr": base_lr,
            "base_lr": base_lr,
            "name": "_rest",
            "weight_decay": weight_decay,
        })
    return groups


def make_vitality_optimizer(
    model: nn.Module,
    probe: VitalityProbe,
    optimizer_cls: Type[torch.optim.Optimizer],
    lr: float,
    **optimizer_kwargs,
) -> torch.optim.Optimizer:
    """Build an optimizer with named per-layer param groups."""
    wd = float(optimizer_kwargs.pop("weight_decay", 0.0))
    groups = build_layer_param_groups(model, probe, lr, weight_decay=wd)
    return optimizer_cls(groups, **optimizer_kwargs)


def apply_lr_scales(
    optimizer: torch.optim.Optimizer,
    probe: VitalityProbe,
    *,
    alpha: float = 4.0,
    lr_min: float = 0.1,
) -> Dict[str, float]:
    """Scale each named param group's LR by ``1 / (1 + alpha * stasis)``."""
    rates = probe.stasis_rates()
    names = probe.layer_names()
    scale_map: Dict[str, float] = {}
    for name, rate in zip(names, rates):
        scale_map[name] = max(lr_min, 1.0 / (1.0 + alpha * float(rate)))

    for pg in optimizer.param_groups:
        name = pg.get("name")
        if name is None or name not in scale_map:
            continue
        if "base_lr" not in pg:
            pg["base_lr"] = pg["lr"]
        pg["lr"] = pg["base_lr"] * scale_map[name]
    return scale_map
