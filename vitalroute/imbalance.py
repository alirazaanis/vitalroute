"""Vitality-weighted class sampling for imbalanced training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from .backbone.mlp import MLP
from .vitality import per_class_health, per_class_stress

StressMode = Literal["stasis", "composite"]


@dataclass
class VitalitySampler:
    """Oversample classes where the network shows high per-class stress.

    ``stress_mode='composite'`` uses all four vitality signals (stasis,
    weak weights, weak input, saturation). ``'stasis'`` uses dead-unit
    rate only (legacy behaviour).
    """

    num_classes: int
    strength: float = 0.7
    beta: float = 4.0
    refresh_every: int = 2
    stress_mode: StressMode = "composite"
    seed: Optional[int] = 0
    _last_class_probs: Optional[np.ndarray] = None
    _last_scores: Optional[np.ndarray] = None
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _class_scores(self, model: MLP, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.stress_mode == "stasis":
            return per_class_health(model, X, y, self.num_classes)
        return per_class_stress(model, X, y, self.num_classes)

    def _compute_class_probs(self, model: MLP, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        scores = self._class_scores(model, X, y)
        if scores.max() == 0:
            weighted = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)
        else:
            z = self.beta * scores
            z = z - z.max()
            weighted = np.exp(z)
            weighted = weighted / weighted.sum()
        uniform = np.full(self.num_classes, 1.0 / self.num_classes, dtype=np.float32)
        probs = (1.0 - self.strength) * uniform + self.strength * weighted
        self._last_class_probs = probs
        self._last_scores = scores
        return probs

    def sample_indices(
        self, epoch: int, model: MLP, X: np.ndarray, y: np.ndarray, n: int,
    ) -> np.ndarray:
        if self._last_class_probs is None or (epoch % self.refresh_every == 0):
            self._compute_class_probs(model, X, y)
        probs = self._last_class_probs
        chosen = self._rng.choice(self.num_classes, size=n, p=probs)
        pools = {c: np.flatnonzero(y == c) for c in range(self.num_classes)}
        out = np.empty(n, dtype=np.int64)
        for i, c in enumerate(chosen):
            pool = pools[int(c)]
            out[i] = pool[self._rng.integers(0, pool.size)] if pool.size else self._rng.integers(0, len(y))
        return out

    def describe_latest(self) -> str:
        if self._last_scores is None:
            return "(sampler: not yet computed)"
        s = self._last_scores
        p = self._last_class_probs
        worst = int(np.argmax(s))
        mode = self.stress_mode
        return (
            f"mode={mode}  stress=[{', '.join(f'{x:.2f}' for x in s)}]  worst=class{worst}  "
            f"prob=[{', '.join(f'{x:.2f}' for x in p)}]"
        )
