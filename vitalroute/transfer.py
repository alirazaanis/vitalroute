"""Label-free transfer model selection by lowest stasis on target data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np

from .backbone.mlp import MLP
from .vitality import health_fingerprint, layer_clotting_rates, layer_stasis_rates_any


@dataclass
class TransferScore:
    name: str
    score: float
    stasis: float
    fingerprint: np.ndarray


def _mean_stasis(model: object, X: np.ndarray, *, sample_size: int = 1024) -> float:
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(0)
        X = X[rng.choice(n, sample_size, replace=False)]
    rates = layer_stasis_rates_any(model, X)
    if len(rates) > 1:
        return float(rates[:-1].mean())
    return float(rates.mean())


def score_transfer_candidates(
    parents: Iterable[Tuple[str, object]],
    X_new: np.ndarray,
    *,
    sample_size: int = 1024,
) -> List[TransferScore]:
    out: List[TransferScore] = []
    for name, parent in parents:
        stasis = _mean_stasis(parent, X_new, sample_size=sample_size)
        if hasattr(parent, "head"):
            fp = health_fingerprint(parent.head, parent.flatten_features(X_new), sample_size=sample_size)  # type: ignore[union-attr]
        else:
            fp = health_fingerprint(parent, X_new, sample_size=sample_size)  # type: ignore[arg-type]
        out.append(TransferScore(name=name, score=1.0 - stasis, stasis=stasis, fingerprint=fp))
    out.sort(key=lambda s: -s.score)
    return out


def pick_transfer_parent(
    parents: Iterable[Tuple[str, object]],
    X_new: np.ndarray,
) -> Tuple[str, object, List[TransferScore]]:
    parents_list = list(parents)
    scores = score_transfer_candidates(parents_list, X_new)
    name_to_parent = {n: p for n, p in parents_list}
    best = scores[0]
    return best.name, name_to_parent[best.name], scores
