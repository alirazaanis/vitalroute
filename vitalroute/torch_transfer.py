"""Label-free transfer model selection for PyTorch models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Literal, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_transfer. "
        "Install it with: pip install torch"
    ) from exc

from .torch_probes import make_probe, ProbeZone

TransferScore = "TorchTransferScore"


@dataclass
class TorchTransferScore:
    name: str
    score: float
    stasis: float
    n_layers: int


def _mean_stasis_torch(
    model: nn.Module,
    X: "torch.Tensor | np.ndarray",
    *,
    probe_zone: ProbeZone = "head",
    sample_size: int = 512,
) -> float:
    probe = make_probe(model, probe_zone=probe_zone, sample_size=sample_size)
    probe.observe(X)
    rates = probe.stasis_rates()
    probe.detach()
    if rates.size > 1:
        return float(rates[:-1].mean())
    return float(rates.mean()) if rates.size else 0.0


def score_transfer_candidates_torch(
    parents: Iterable[Tuple[str, nn.Module]],
    X_new: "torch.Tensor | np.ndarray",
    *,
    probe_zone: ProbeZone = "head",
    sample_size: int = 512,
) -> List[TorchTransferScore]:
    """Rank pretrained parents by lowest stasis on unlabeled target inputs."""
    out: List[TorchTransferScore] = []
    for name, parent in parents:
        probe = make_probe(parent, probe_zone=probe_zone, sample_size=sample_size)
        probe.observe(X_new)
        stasis = float(probe.mean_stasis())
        n_layers = len(probe.layer_names())
        probe.detach()
        out.append(
            TorchTransferScore(name=name, score=1.0 - stasis, stasis=stasis, n_layers=n_layers)
        )
    out.sort(key=lambda s: -s.score)
    return out


def pick_transfer_parent_torch(
    parents: Iterable[Tuple[str, nn.Module]],
    X_new: "torch.Tensor | np.ndarray",
    *,
    probe_zone: ProbeZone = "head",
    sample_size: int = 512,
) -> Tuple[str, nn.Module, List[TorchTransferScore]]:
    parents_list = list(parents)
    scores = score_transfer_candidates_torch(
        parents_list, X_new, probe_zone=probe_zone, sample_size=sample_size
    )
    name_to_parent = {n: p for n, p in parents_list}
    best = scores[0]
    return best.name, name_to_parent[best.name], scores


def _classifier_keys(state: dict, model: Optional[nn.Module] = None) -> List[str]:
    """Parameter keys belonging to the classifier head."""
    by_name = [k for k in state if any(x in k.lower() for x in ("fc.", "classifier.", "head."))]
    if by_name:
        return by_name
    if model is not None:
        last_linear = None
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear):
                last_linear = name
        if last_linear:
            prefix = last_linear + "."
            return [k for k in state if k.startswith(prefix)]
    return [k for k in state if ".weight" in k and k.split(".")[-2].isdigit()]


def warm_start_from_parent(
    child: nn.Module,
    parent: nn.Module,
    *,
    fresh_head: bool = True,
    strict: bool = False,
) -> dict:
    """Copy compatible weights from ``parent`` into ``child``.

    When ``fresh_head=True``, classifier / fc layers are left at their
    current (random) initialization.
    """
    parent_state = parent.state_dict()
    child_state = child.state_dict()
    skip = set(_classifier_keys(parent_state, parent)) if fresh_head else set()
    loaded = {}
    for k, v in parent_state.items():
        if k in skip:
            continue
        if k in child_state and child_state[k].shape == v.shape:
            loaded[k] = v
    child_state.update(loaded)
    child.load_state_dict(child_state, strict=strict)
    return {
        "n_loaded": len(loaded),
        "n_skipped_head": len(skip),
        "fresh_head": fresh_head,
    }
