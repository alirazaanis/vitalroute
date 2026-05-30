"""Per-sample difficulty sampling (hard-example mining)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .backbone.mlp import MLP
from .vitality import per_sample_stress


@dataclass
class HardSampleSampler:
    """Oversample individual examples with high composite stress scores.

    Mixes uniform draws with a softmax over per-sample stress (stasis,
    weak coupling, saturation, low confidence).
    """

    strength: float = 0.6
    beta: float = 3.0
    refresh_every: int = 1
    max_score_samples: int = 4096
    seed: Optional[int] = 0
    _last_probs: Optional[np.ndarray] = None
    _last_mean_score: float = 0.0
    _rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def _compute_probs(self, model: MLP, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n = len(y)
        scores = per_sample_stress(
            model, X, y, sample_size=min(n, self.max_score_samples),
        )
        if scores.max() <= 0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        z = self.beta * scores.astype(np.float64)
        z = z - z.max()
        weighted = np.exp(z)
        weighted /= weighted.sum()
        uniform = np.full(n, 1.0 / n, dtype=np.float64)
        probs = (1.0 - self.strength) * uniform + self.strength * weighted
        self._last_probs = probs.astype(np.float32)
        self._last_mean_score = float(scores.mean())
        return probs

    def sample_indices(
        self, epoch: int, model: MLP, X: np.ndarray, y: np.ndarray, n: int,
    ) -> np.ndarray:
        if self._last_probs is None or (epoch % self.refresh_every == 0):
            self._compute_probs(model, X, y)
        return self._rng.choice(len(y), size=n, replace=True, p=self._last_probs)

    def describe_latest(self) -> str:
        if self._last_probs is None:
            return "(hard sampler: not yet computed)"
        top = float(self._last_probs.max())
        return f"mean_stress={self._last_mean_score:.3f}  max_sample_p={top:.4f}"
