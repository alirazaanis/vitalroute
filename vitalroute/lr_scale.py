"""Per-layer learning rate scaling from layer stasis rates."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from .backbone.mlp import MLP, SGD, _Layer
from .vitality import _resolve_core, layer_stasis_rates_any


@dataclass
class VitalityScaledAdam:
    """Adam with per-layer LR scaled by 1 / (1 + alpha * stasis_rate)."""

    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    alpha: float = 4.0
    min_scale: float = 0.1
    _t: int = 0
    _layer_scale: Dict[int, float] = field(default_factory=dict)

    def refresh_health(self, model: object, X: np.ndarray) -> Dict[int, float]:
        rates = layer_stasis_rates_any(model, X)
        mlp, _ = _resolve_core(model, X)
        scales = {}
        for layer, rate in zip(mlp._layers, rates):
            scales[id(layer)] = max(self.min_scale, 1.0 / (1.0 + self.alpha * float(rate)))
        self._layer_scale = scales
        return scales

    def step(self, layer: _Layer, gW: np.ndarray, gb: np.ndarray) -> None:
        if layer.mW.size == 0:
            layer.mW = np.zeros_like(layer.W)
            layer.vW = np.zeros_like(layer.W)
            layer.mb = np.zeros_like(layer.b)
            layer.vb = np.zeros_like(layer.b)
        self._t += 1
        if self.weight_decay:
            gW = gW + self.weight_decay * layer.W
        layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * gW
        layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (gW * gW)
        layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * gb
        layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (gb * gb)
        mW_hat = layer.mW / (1 - self.beta1 ** self._t)
        vW_hat = layer.vW / (1 - self.beta2 ** self._t)
        mb_hat = layer.mb / (1 - self.beta1 ** self._t)
        vb_hat = layer.vb / (1 - self.beta2 ** self._t)
        scale = self._layer_scale.get(id(layer), 1.0)
        eff_lr = self.lr * scale
        layer.W -= eff_lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= eff_lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


@dataclass
class VitalityScaledSGD:
    """SGD-momentum with per-layer LR scaled by stasis rate."""

    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    alpha: float = 4.0
    min_scale: float = 0.1
    _layer_scale: Dict[int, float] = field(default_factory=dict)

    def refresh_health(self, model: object, X: np.ndarray) -> Dict[int, float]:
        rates = layer_stasis_rates_any(model, X)
        mlp, _ = _resolve_core(model, X)
        scales = {}
        for layer, rate in zip(mlp._layers, rates):
            scales[id(layer)] = max(self.min_scale, 1.0 / (1.0 + self.alpha * float(rate)))
        self._layer_scale = scales
        return scales

    def step(self, layer: _Layer, gW: np.ndarray, gb: np.ndarray) -> None:
        if layer.velW.size == 0:
            layer.velW = np.zeros_like(layer.W)
            layer.velb = np.zeros_like(layer.b)
        if self.weight_decay:
            gW = gW + self.weight_decay * layer.W
        scale = self._layer_scale.get(id(layer), 1.0)
        eff_lr = self.lr * scale
        layer.velW = self.momentum * layer.velW - eff_lr * gW
        layer.velb = self.momentum * layer.velb - eff_lr * gb
        layer.W += layer.velW
        layer.b += layer.velb


def make_vitality_scaled_optimizer(
    name: str = "adam",
    *,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    momentum: float = 0.9,
    alpha: float = 4.0,
    min_scale: float = 0.1,
) -> VitalityScaledAdam | VitalityScaledSGD:
    """Factory for vitality-scaled optimizers (drop-in for backbone Adam/SGD)."""
    if name == "adam":
        return VitalityScaledAdam(
            lr=lr, weight_decay=weight_decay, alpha=alpha, min_scale=min_scale,
        )
    if name == "sgd":
        return VitalityScaledSGD(
            lr=lr, momentum=momentum, weight_decay=weight_decay,
            alpha=alpha, min_scale=min_scale,
        )
    raise ValueError(f"unknown optimizer: {name!r}")


def refresh_lr_scales(
    optimizer: object,
    model: object,
    X: np.ndarray,
    *,
    epoch: int,
    refresh_every: int = 1,
) -> Optional[Dict[int, float]]:
    """Invoked at epoch start when using a vitality-scaled optimizer."""
    if not hasattr(optimizer, "refresh_health"):
        return None
    if epoch % refresh_every != 0:
        return None
    return optimizer.refresh_health(model, X)
