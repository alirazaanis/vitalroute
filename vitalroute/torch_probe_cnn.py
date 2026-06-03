"""BatchNorm-aware CNN vitality probes.

``CNNVitalityProbe`` tracks Conv→BatchNorm→ReLU blocks and linear classifier
layers. Probe zones: ``head`` (classifier), ``trunk`` (conv stack), ``all``.
"""

from __future__ import annotations

from typing import List, Literal, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_probe_cnn. "
        "Install it with: pip install torch"
    ) from exc

from .torch_probe import (
    VitalityProbe,
    _ACT_TYPES,
    _four_stress_rates,
    _stasis_rate,
    _to_numpy,
)

ProbeZone = Literal["head", "trunk", "all"]
_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
_CONV_TYPES = (nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _spatial_stasis_rate(
    act: np.ndarray,
    activation_type: str,
    threshold: float,
) -> float:
    """Channel stasis for conv activations ``(N, C, *spatial)``."""
    if act.ndim <= 2:
        return _stasis_rate(act, activation_type, threshold)

    n = act.shape[0]
    flat = act.reshape(n, act.shape[1], -1)  # (N, C, S)
    if activation_type == "relu":
        fires = flat > 0
    elif activation_type == "linear":
        fires = np.abs(flat) > 1e-6
    else:
        fires = np.abs(flat) > 1e-3

    activity_rate = fires.mean(axis=(0, 2))   # per channel
    variance = flat.var(axis=(0, 2))
    dead = ((activity_rate < threshold) | (variance < 1e-6)).sum()
    return float(dead) / max(1, act.shape[1])


def _spatial_four_stress(
    act: np.ndarray,
    W: Optional[np.ndarray],
    incoming_mean: np.ndarray,
    activation_type: str,
    *,
    dead_threshold: float = 0.01,
) -> Tuple[float, float, float, float]:
    """Four stress signals aggregated over conv channels."""
    if act.ndim <= 2:
        return _four_stress_rates(
            act, W, incoming_mean, activation_type, dead_threshold=dead_threshold
        )

    n = act.shape[0]
    flat = act.reshape(n, act.shape[1], -1)
    channel_means = flat.mean(axis=2)          # (N, C)
    return _four_stress_rates(
        channel_means, W, incoming_mean, activation_type, dead_threshold=dead_threshold
    )


class _BnConvUnit:
    """Conv → optional BatchNorm → optional activation."""

    def __init__(
        self,
        name: str,
        conv_mod: nn.Module,
        bn_mod: Optional[nn.Module] = None,
        act_mod: Optional[nn.Module] = None,
    ):
        self.name = name
        self.conv_mod = conv_mod
        self.bn_mod = bn_mod
        self.act_mod = act_mod
        self._post_act: Optional[np.ndarray] = None
        self._pre_act: Optional[np.ndarray] = None
        self._input_mean: Optional[np.ndarray] = None
        self._handles: List = []

    def attach(self) -> None:
        def _conv_hook(mod, inp, out):
            a = _to_numpy(out)
            self._pre_act = a
            if inp and inp[0] is not None:
                x = inp[0]
                if x.dim() > 2:
                    self._input_mean = np.abs(_to_numpy(x)).mean(axis=(0, 2, 3) if x.dim() == 4 else (0, 2))
                else:
                    self._input_mean = np.abs(_to_numpy(x)).mean(axis=0)

        def _bn_hook(mod, inp, out):
            self._pre_act = _to_numpy(out)

        def _act_hook(mod, inp, out):
            self._post_act = _to_numpy(out)

        self._handles.append(self.conv_mod.register_forward_hook(_conv_hook))
        if self.bn_mod is not None:
            self._handles.append(self.bn_mod.register_forward_hook(_bn_hook))
        if self.act_mod is not None:
            self._handles.append(self.act_mod.register_forward_hook(_act_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    @property
    def activation(self) -> Optional[np.ndarray]:
        return self._post_act if self._post_act is not None else self._pre_act

    def get_weight(self) -> Optional[np.ndarray]:
        W = getattr(self.conv_mod, "weight", None)
        if W is None:
            return None
        w = _to_numpy(W)
        if w.ndim == 4:
            return w.reshape(w.shape[0], -1)
        if w.ndim == 3:
            return w.reshape(w.shape[0], -1)
        return w

    def activation_type(self) -> str:
        if self.act_mod is not None:
            cls = type(self.act_mod).__name__.lower()
        else:
            cls = type(self.conv_mod).__name__.lower()
        if any(k in cls for k in ("relu", "gelu", "silu", "swish", "elu", "leaky")):
            return "relu"
        return "linear"


class CNNVitalityProbe(VitalityProbe):
    """Vitality probe for CNN classifiers with BatchNorm-aware conv tracking.

    Parameters
    ----------
    probe_zone:
        ``"head"`` — linear classifier layers only.
        ``"trunk"`` — Conv+BN+ReLU blocks only.
        ``"all"`` — trunk and head.
    """

    def __init__(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
        *,
        probe_zone: ProbeZone = "all",
        dead_threshold: float = 0.01,
        sample_size: int = 1024,
    ):
        self._probe_zone = probe_zone
        self._conv_units: List[_BnConvUnit] = []
        self._model = model
        self._dead_threshold = dead_threshold
        self._sample_size = sample_size
        self._units = []
        self._attach_cnn(layer_names)

    def _attach_head(self, layer_names: Optional[List[str]]) -> None:
        """Attach Linear classifier layers only (skip Conv2d)."""
        name_set = set(layer_names) if layer_names else None
        all_named = list(self._model.named_modules())
        i = 0
        while i < len(all_named):
            name, mod = all_named[i]
            if not isinstance(mod, nn.Linear):
                i += 1
                continue
            if name_set is not None and name not in name_set:
                i += 1
                continue
            act_mod = None
            if i + 1 < len(all_named):
                _, next_mod = all_named[i + 1]
                if isinstance(next_mod, _ACT_TYPES):
                    act_mod = next_mod
                    i += 1
            from .torch_probe import _LayerUnit
            unit = _LayerUnit(name, mod, act_mod)
            unit.attach()
            self._units.append(unit)
            i += 1

    def _attach_cnn(self, layer_names: Optional[List[str]]) -> None:
        if self._probe_zone in ("head", "all"):
            self._attach_head(layer_names)

        if self._probe_zone not in ("trunk", "all"):
            return

        name_set = set(layer_names) if layer_names else None
        all_named = list(self._model.named_modules())
        i = 0
        while i < len(all_named):
            name, mod = all_named[i]
            if not isinstance(mod, _CONV_TYPES):
                i += 1
                continue
            if name_set is not None and name not in name_set:
                i += 1
                continue

            bn_mod = act_mod = None
            j = i + 1
            while j < len(all_named):
                _, next_mod = all_named[j]
                next_name = all_named[j][0]
                if not next_name.startswith(name + ".") and "." in next_name[len(name):]:
                    break
                if isinstance(next_mod, _BN_TYPES) and bn_mod is None:
                    bn_mod = next_mod
                    j += 1
                    continue
                if isinstance(next_mod, _ACT_TYPES):
                    act_mod = next_mod
                    j += 1
                    break
                if isinstance(next_mod, _CONV_TYPES):
                    break
                j += 1

            unit = _BnConvUnit(name, mod, bn_mod, act_mod)
            unit.attach()
            self._conv_units.append(unit)
            i += 1

    def detach(self) -> None:
        for u in self._conv_units:
            u.detach()
        self._conv_units.clear()
        super().detach()

    def layer_names(self) -> List[str]:
        trunk = [u.name for u in self._conv_units]
        head = super().layer_names()
        if self._probe_zone == "trunk":
            return trunk
        if self._probe_zone == "head":
            return head
        return trunk + head

    def stasis_rates(self) -> np.ndarray:
        rates: List[float] = []
        if self._probe_zone in ("trunk", "all"):
            for u in self._conv_units:
                act = u.activation
                if act is None:
                    rates.append(0.0)
                else:
                    rates.append(
                        _spatial_stasis_rate(act, u.activation_type(), self._dead_threshold)
                    )
        if self._probe_zone in ("head", "all"):
            rates.extend(super().stasis_rates().tolist())
        return np.asarray(rates, dtype=np.float32)

    def composite_stress(
        self,
        weights: Tuple[float, float, float, float] = (1.0, 0.6, 0.6, 0.5),
    ) -> np.ndarray:
        w_st, w_ww, w_wi, w_sat = weights
        denom = w_st + w_ww + w_wi + w_sat
        scores: List[float] = []

        if self._probe_zone in ("trunk", "all"):
            for u in self._conv_units:
                act = u.activation
                if act is None:
                    scores.append(0.0)
                    continue
                if act.ndim > 2:
                    n = act.shape[0]
                    channel_means = act.reshape(n, act.shape[1], -1).mean(axis=2)
                    act_for_stress = channel_means
                else:
                    act_for_stress = act
                W = u.get_weight()
                incoming = (
                    u._input_mean
                    if u._input_mean is not None
                    else np.abs(act_for_stress).mean(axis=0)
                )
                s0, s1, s2, s3 = _spatial_four_stress(
                    act if act.ndim > 2 else act_for_stress,
                    W,
                    incoming,
                    u.activation_type(),
                    dead_threshold=self._dead_threshold,
                )
                scores.append((w_st * s0 + w_ww * s1 + w_wi * s2 + w_sat * s3) / denom)

        if self._probe_zone in ("head", "all"):
            scores.extend(super().composite_stress(weights).tolist())
        return np.asarray(scores, dtype=np.float32)

    def summary(self) -> str:
        stasis = self.stasis_rates()
        composite = self.composite_stress()
        names = self.layer_names()
        lines = [
            f"  {names[i]:<40}  stasis={stasis[i]:.3f}  composite={composite[i]:.3f}"
            for i in range(len(names))
        ]
        zone = f"zone={self._probe_zone}"
        header = f"CNNVitalityProbe ({zone}, {len(names)} layers)"
        body = "\n".join(lines) if lines else "(no layers tracked)"
        return f"{header}\n{body}"
