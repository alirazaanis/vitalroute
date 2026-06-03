"""PyTorch data helpers for VitalRoute probe batches."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_data. "
        "Install it with: pip install torch"
    ) from exc


def stratified_probe_indices(
    labels: Union[np.ndarray, "torch.Tensor", Sequence[int]],
    *,
    per_class: int = 50,
    num_classes: Optional[int] = None,
    seed: int = 0,
) -> np.ndarray:
    """Return dataset indices with up to ``per_class`` examples per class.

    Every class present in ``labels`` is represented. Raises ``ValueError`` if
    any expected class (0 .. num_classes-1) has zero samples when
    ``num_classes`` is set.
    """
    if isinstance(labels, torch.Tensor):
        y = labels.cpu().numpy().astype(np.int64)
    else:
        y = np.asarray(labels, dtype=np.int64)

    rng = np.random.default_rng(seed)
    classes = (
        list(range(num_classes))
        if num_classes is not None
        else sorted(int(c) for c in np.unique(y))
    )
    picks: List[int] = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            if num_classes is not None:
                raise ValueError(f"class {c} has no samples in labels")
            continue
        n = min(per_class, int(idx.size))
        picks.extend(rng.choice(idx, n, replace=False).tolist())
    if not picks:
        raise ValueError("stratified_probe_indices: no samples selected")
    return np.asarray(picks, dtype=np.int64)


def stratified_probe_batch(
    dataset,
    labels: Union[np.ndarray, "torch.Tensor", Sequence[int]],
    *,
    per_class: int = 50,
    num_classes: Optional[int] = None,
    seed: int = 0,
    device: Optional[Union[str, "torch.device"]] = None,
) -> Tuple["torch.Tensor", "torch.Tensor", np.ndarray]:
    """Build a stratified probe batch from a PyTorch ``Dataset``.

    Returns ``(X_probe, y_probe, indices)`` where ``X_probe`` is stacked
    tensors from ``dataset[i][0]`` and ``y_probe`` uses ``labels[indices]``.
    """
    idx = stratified_probe_indices(
        labels, per_class=per_class, num_classes=num_classes, seed=seed
    )
    if isinstance(labels, torch.Tensor):
        y_np = labels.cpu().numpy().astype(np.int64)
    else:
        y_np = np.asarray(labels, dtype=np.int64)

    xs = []
    for i in idx:
        sample = dataset[int(i)]
        xs.append(sample[0] if isinstance(sample, (tuple, list)) else sample)
    X = torch.stack(xs)
    y = torch.from_numpy(y_np[idx].astype(np.int64))
    if device is not None:
        X = X.to(device)
        y = y.to(device)
    return X, y, idx
