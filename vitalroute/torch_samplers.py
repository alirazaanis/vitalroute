"""PyTorch-native samplers driven by VitalityProbe stress scores.

Two samplers, mirroring the NumPy originals but working with any
``torch.nn.Module`` via ``VitalityProbe``:

``TorchVitalitySampler``
    For imbalanced data. Oversamples classes where the model shows high
    composite stress. Drop-in for PyTorch ``DataLoader(sampler=...)``.

``TorchHardSampleSampler``
    For scarce / balanced data. Oversamples individual examples that
    score high on per-sample stress (stasis + low confidence).

Both implement ``__iter__`` and ``__len__`` so they work directly as
a PyTorch ``Sampler``.

Example
-------
::

    from vitalroute.torch_samplers import TorchVitalitySampler
    from vitalroute.torch_probe import VitalityProbe

    probe   = VitalityProbe(model)
    sampler = TorchVitalitySampler(y_train, num_classes=10, probe=probe)

    for epoch in range(epochs):
        probe.observe(X_train)          # refresh activations once per epoch
        sampler.refresh(X_train, y_train, model)   # recompute class weights
        loader = DataLoader(dataset, sampler=sampler, batch_size=64)
        for X_batch, y_batch in loader:
            ...
"""

from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

try:
    import torch
    from torch.utils.data import Sampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_samplers. "
        "Install it with: pip install torch"
    ) from exc

from .torch_probe import VitalityProbe


class TorchVitalitySampler(Sampler):
    """Oversample classes with high composite vitality stress.

    Parameters
    ----------
    labels:
        1-D array / tensor of integer class labels for the full training set.
    num_classes:
        Total number of classes.
    probe:
        A ``VitalityProbe`` already attached to the model.
    n:
        Number of indices to yield per epoch. Defaults to ``len(labels)``.
    strength:
        Mix between uniform (0.0) and fully stress-weighted (1.0) sampling.
    beta:
        Softmax temperature applied to stress scores (higher = sharper).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        labels: "torch.Tensor | np.ndarray",
        num_classes: int,
        probe: VitalityProbe,
        *,
        n: Optional[int] = None,
        strength: float = 0.7,
        beta: float = 4.0,
        seed: int = 0,
    ):
        if isinstance(labels, torch.Tensor):
            self._labels = labels.cpu().numpy().astype(np.int64)
        else:
            self._labels = np.asarray(labels, dtype=np.int64)
        self._num_classes = num_classes
        self._probe = probe
        self._n = n if n is not None else len(self._labels)
        self._strength = strength
        self._beta = beta
        self._rng = np.random.default_rng(seed)
        self._class_probs: Optional[np.ndarray] = None
        # index pool per class
        self._pools = {
            c: np.flatnonzero(self._labels == c)
            for c in range(num_classes)
        }

    def refresh(
        self,
        X: "torch.Tensor | np.ndarray",
        y: "torch.Tensor | np.ndarray",
        model: "torch.nn.Module",
    ) -> np.ndarray:
        """Recompute class sampling probabilities from current vitality stress.

        Call once per epoch after ``probe.observe()``.
        Returns the raw stress scores (one per class).
        """
        scores = self._probe.per_class_stress(X, y, self._num_classes)
        self._class_probs = self._stress_to_probs(scores)
        return scores

    def _stress_to_probs(self, scores: np.ndarray) -> np.ndarray:
        if scores.max() == 0:
            return np.full(self._num_classes, 1.0 / self._num_classes, dtype=np.float64)
        z = self._beta * scores.astype(np.float64)
        z -= z.max()
        weighted = np.exp(z)
        weighted /= weighted.sum()
        uniform = np.full(self._num_classes, 1.0 / self._num_classes, dtype=np.float64)
        probs = (1.0 - self._strength) * uniform + self._strength * weighted
        probs /= probs.sum()
        return probs

    def __iter__(self) -> Iterator[int]:
        probs = (
            self._class_probs
            if self._class_probs is not None
            else np.full(self._num_classes, 1.0 / self._num_classes)
        )
        chosen_classes = self._rng.choice(self._num_classes, size=self._n, p=probs)
        indices = np.empty(self._n, dtype=np.int64)
        for i, c in enumerate(chosen_classes):
            pool = self._pools[int(c)]
            indices[i] = pool[self._rng.integers(0, pool.size)] if pool.size else self._rng.integers(0, len(self._labels))
        yield from indices.tolist()

    def __len__(self) -> int:
        return self._n

    def describe(self) -> str:
        if self._class_probs is None:
            return "(TorchVitalitySampler: not yet refreshed)"
        worst = int(np.argmax(self._class_probs))
        return (
            f"class_probs=[{', '.join(f'{p:.2f}' for p in self._class_probs)}]  "
            f"highest_p=class{worst}"
        )


class TorchHardSampleSampler(Sampler):
    """Oversample individual examples with high per-sample stress.

    Parameters
    ----------
    n_samples:
        Size of the training set.
    probe:
        A ``VitalityProbe`` already attached to the model.
    n:
        Number of indices to yield per epoch. Defaults to ``n_samples``.
    strength:
        Mix between uniform (0.0) and fully stress-weighted (1.0).
    beta:
        Softmax temperature for stress scores.
    seed:
        Random seed.
    """

    def __init__(
        self,
        n_samples: int,
        probe: VitalityProbe,
        *,
        n: Optional[int] = None,
        strength: float = 0.6,
        beta: float = 3.0,
        seed: int = 0,
    ):
        self._n_samples = n_samples
        self._probe = probe
        self._n = n if n is not None else n_samples
        self._strength = strength
        self._beta = beta
        self._rng = np.random.default_rng(seed)
        self._sample_probs: Optional[np.ndarray] = None

    def refresh(
        self,
        X: "torch.Tensor | np.ndarray",
        y: "torch.Tensor | np.ndarray",
        model: "torch.nn.Module",
    ) -> np.ndarray:
        """Recompute per-sample probabilities. Call after ``probe.observe()``."""
        scores = self._probe.per_sample_stress(X, y)
        n = len(scores)
        if scores.max() <= 0:
            self._sample_probs = np.full(self._n_samples, 1.0 / self._n_samples)
            return scores
        z = self._beta * scores.astype(np.float64)
        z -= z.max()
        weighted = np.exp(z)
        weighted /= weighted.sum()
        uniform = np.full(n, 1.0 / n, dtype=np.float64)
        probs = (1.0 - self._strength) * uniform + self._strength * weighted
        # Expand to full dataset size if probe was subsampled
        if n < self._n_samples:
            full_probs = np.full(self._n_samples, probs.mean() / self._n_samples, dtype=np.float64)
            full_probs /= full_probs.sum()
            self._sample_probs = full_probs
        else:
            self._sample_probs = (probs / probs.sum()).astype(np.float64)
        return scores

    def __iter__(self) -> Iterator[int]:
        probs = (
            self._sample_probs
            if self._sample_probs is not None
            else np.full(self._n_samples, 1.0 / self._n_samples)
        )
        yield from self._rng.choice(self._n_samples, size=self._n, replace=True, p=probs).tolist()

    def __len__(self) -> int:
        return self._n
