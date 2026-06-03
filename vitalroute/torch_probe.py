"""PyTorch integration for VitalRoute vitality probes.

Attaches to any ``torch.nn.Module`` via forward hooks. No changes to
the model, optimizer, or training loop are required beyond the three hook
calls shown below.

Usage
-----
::

    from vitalroute.torch_probe import VitalityProbe

    probe = VitalityProbe(model)          # attach once before training
    probe.observe(X_batch)               # once per epoch (or on demand)

    # Read composite stress per layer
    rates = probe.stasis_rates()         # np.ndarray, one value per tracked layer
    mean  = probe.mean_stasis()          # scalar float

    # Full per-class and per-sample stress (needs labels)
    class_scores  = probe.per_class_stress(X, y, num_classes)
    sample_scores = probe.per_sample_stress(X, y)

    probe.detach()                        # remove hooks when done

Compatible with any ``Linear`` or ``Conv2d``-based classifier.
Tested with PyTorch >= 2.0 on CPU and CUDA.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_probe. "
        "Install it with: pip install torch"
    ) from exc


# ── internal helpers ──────────────────────────────────────────────────────────

def _to_numpy(t: "torch.Tensor") -> np.ndarray:
    return t.detach().cpu().float().numpy()


def _stasis_rate(act: np.ndarray, activation_type: str, threshold: float) -> float:
    """Fraction of units that are near-dead on this batch."""
    if activation_type == "relu":
        fires = act > 0
    elif activation_type == "linear":
        fires = np.abs(act) > 1e-6
    else:
        fires = np.abs(act) > 1e-3
    activity_rate = fires.mean(axis=0)         # (units,)
    variance = act.var(axis=0)
    dead = ((activity_rate < threshold) | (variance < 1e-6)).sum()
    return float(dead) / max(1, act.shape[1])


def _four_stress_rates(
    act: np.ndarray,
    W: Optional[np.ndarray],
    incoming_mean: np.ndarray,
    activation_type: str,
    *,
    dead_threshold: float = 0.01,
    variance_threshold: float = 1e-6,
    weight_threshold: float = 0.05,
    headache_fraction: float = 0.8,
) -> Tuple[float, float, float, float]:
    """Return (stasis, weak_weights, weak_input, saturation) rates.

    W shape: (out_features, in_features)  — PyTorch convention.
    incoming_mean shape: (in_features,).
    act shape: (N, out_features).
    """
    if activation_type == "relu":
        fires = act > 0
    elif activation_type == "linear":
        fires = np.abs(act) > 1e-6
    else:
        fires = np.abs(act) > 1e-3

    activity_rate = fires.mean(axis=0)
    variance = act.var(axis=0)
    mean_abs = np.abs(act.mean(axis=0))
    col_norm = np.linalg.norm(W, axis=1) if W is not None else np.ones(act.shape[1])

    units = act.shape[1]
    stasis  = float(((activity_rate < dead_threshold) | (variance < variance_threshold)).sum()) / units
    weak_w  = float((col_norm < weight_threshold).sum()) / units
    sat     = float(((variance < variance_threshold) & (mean_abs > 1e-3)).sum()) / units

    if W is not None and incoming_mean.shape[0] == W.shape[1]:
        abs_w = np.abs(W)                              # (out, in)
        ratio = (incoming_mean[None, :] < abs_w).mean(axis=1)  # (out,)
        weak_in = float((ratio > headache_fraction).sum()) / units
    else:
        weak_in = 0.0

    return stasis, weak_w, weak_in, sat


# ── layer unit ────────────────────────────────────────────────────────────────

_ACT_TYPES = (nn.ReLU, nn.GELU, nn.SiLU, nn.Tanh, nn.Sigmoid, nn.LeakyReLU, nn.ELU)
_LINEAR_TYPES = (nn.Linear, nn.Conv2d)


class _LayerUnit:
    """One logical hidden unit = a linear/conv module + its following activation.

    Hooks the linear module to capture input statistics and weight info.
    Hooks the activation module (if present) to capture post-activation output.
    Falls back to pre-activation output when there is no explicit activation module.
    """

    def __init__(self, name: str, linear_mod: "nn.Module", act_mod: Optional["nn.Module"] = None):
        self.name = name
        self.linear_mod = linear_mod
        self.act_mod = act_mod

        self._post_act: Optional[np.ndarray] = None
        self._pre_act: Optional[np.ndarray] = None
        self._input_mean: Optional[np.ndarray] = None
        self._linear_handle = None
        self._act_handle = None

    def attach(self):
        def _linear_hook(mod, inp, out):
            a = out
            if a.dim() > 2:
                a = a.flatten(1)
            self._pre_act = _to_numpy(a)
            if inp and inp[0] is not None:
                x = inp[0]
                if x.dim() > 2:
                    x = x.flatten(1)
                self._input_mean = np.abs(_to_numpy(x)).mean(axis=0)

        def _act_hook(mod, inp, out):
            a = out
            if a.dim() > 2:
                a = a.flatten(1)
            self._post_act = _to_numpy(a)

        self._linear_handle = self.linear_mod.register_forward_hook(_linear_hook)
        if self.act_mod is not None:
            self._act_handle = self.act_mod.register_forward_hook(_act_hook)

    def detach(self):
        for h in (self._linear_handle, self._act_handle):
            if h is not None:
                h.remove()
        self._linear_handle = self._act_handle = None

    @property
    def activation(self) -> Optional[np.ndarray]:
        return self._post_act if self._post_act is not None else self._pre_act

    def get_weight(self) -> Optional[np.ndarray]:
        W = getattr(self.linear_mod, "weight", None)
        if W is None:
            return None
        w = _to_numpy(W)
        if w.ndim == 4:
            return w.reshape(w.shape[0], -1)   # Conv2d → (out, in*kH*kW)
        return w                               # Linear → (out, in)

    def activation_type(self) -> str:
        if self.act_mod is not None:
            cls = type(self.act_mod).__name__.lower()
        else:
            cls = type(self.linear_mod).__name__.lower()
        if any(k in cls for k in ("relu", "gelu", "silu", "swish", "elu", "leaky")):
            return "relu"
        if "tanh" in cls:
            return "tanh"
        if "sigmoid" in cls:
            return "sigmoid"
        return "linear"


# ── main probe ───────────────────────────────────────────────────────────────

class VitalityProbe:
    """Vitality probes for a PyTorch model via forward hooks.

    Parameters
    ----------
    model:
        Any ``nn.Module``. Each ``Linear`` / ``Conv2d`` layer is paired
        with the activation function that immediately follows it in the
        module list. Pass ``layer_names`` to restrict to specific linear
        submodules.
    layer_names:
        Optional list of submodule names (must be Linear/Conv2d) to track.
    dead_threshold:
        Activity rate below which a unit counts as in stasis.
    sample_size:
        Maximum number of rows used when scoring stress.
    """

    def __init__(
        self,
        model: "nn.Module",
        layer_names: Optional[List[str]] = None,
        *,
        dead_threshold: float = 0.01,
        sample_size: int = 1024,
    ):
        self._model = model
        self._dead_threshold = dead_threshold
        self._sample_size = sample_size
        self._units: List[_LayerUnit] = []
        self._attach(layer_names)

    # ── setup ─────────────────────────────────────────────────────────────

    def _attach(self, layer_names: Optional[List[str]]):
        name_set = set(layer_names) if layer_names else None
        all_named = list(self._model.named_modules())
        i = 0
        while i < len(all_named):
            name, mod = all_named[i]
            if name_set is not None:
                if name not in name_set or not isinstance(mod, _LINEAR_TYPES):
                    i += 1
                    continue
            elif not isinstance(mod, _LINEAR_TYPES):
                i += 1
                continue
            act_mod = None
            if i + 1 < len(all_named):
                _, next_mod = all_named[i + 1]
                if isinstance(next_mod, _ACT_TYPES):
                    act_mod = next_mod
                    i += 1
            unit = _LayerUnit(name, mod, act_mod)
            unit.attach()
            self._units.append(unit)
            i += 1

    def detach(self):
        """Remove all forward hooks after training is complete."""
        for u in self._units:
            u.detach()
        self._units.clear()

    # ── observe ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def observe(self, X: "torch.Tensor | np.ndarray"):
        """Forward pass to capture activations. No gradients computed."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        n = X.shape[0]
        if n > self._sample_size:
            idx = np.random.default_rng(0).choice(n, self._sample_size, replace=False)
            X = X[idx]
        X = X.to(next(self._model.parameters()).device)
        self._model.eval()
        self._model(X)

    # ── stress signals ────────────────────────────────────────────────────

    def stasis_rates(self) -> np.ndarray:
        """One stasis rate in [0,1] per tracked layer."""
        return np.array([
            _stasis_rate(u.activation, u.activation_type(), self._dead_threshold)
            if u.activation is not None else 0.0
            for u in self._units
        ], dtype=np.float32)

    def mean_stasis(self) -> float:
        """Mean stasis rate across all tracked layers (scalar)."""
        r = self.stasis_rates()
        return float(r.mean()) if r.size else 0.0

    def layer_names(self) -> List[str]:
        return [u.name for u in self._units]

    def composite_stress(
        self,
        weights: Tuple[float, float, float, float] = (1.0, 0.6, 0.6, 0.5),
    ) -> np.ndarray:
        """Composite stress score per layer using all four vitality signals."""
        w_st, w_ww, w_wi, w_sat = weights
        denom = w_st + w_ww + w_wi + w_sat
        scores = []
        for u in self._units:
            if u.activation is None:
                scores.append(0.0)
                continue
            act = u.activation
            W = u.get_weight()
            incoming = u._input_mean if u._input_mean is not None else np.abs(act).mean(axis=0)
            s0, s1, s2, s3 = _four_stress_rates(
                act, W, incoming, u.activation_type(),
                dead_threshold=self._dead_threshold,
            )
            scores.append((w_st * s0 + w_ww * s1 + w_wi * s2 + w_sat * s3) / denom)
        return np.array(scores, dtype=np.float32)

    # ── class / sample stress (need labels) ───────────────────────────────

    @torch.no_grad()
    def per_class_stress(
        self,
        X: "torch.Tensor | np.ndarray",
        y: "torch.Tensor | np.ndarray",
        num_classes: int,
        *,
        sample_size_per_class: int = 256,
    ) -> np.ndarray:
        """Composite stress per class in [0, 1]. Higher = model struggles more."""
        X_np = _to_numpy(X) if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32)
        y_np = _to_numpy(y).astype(np.int64) if isinstance(y, torch.Tensor) else np.asarray(y, np.int64)
        device = next(self._model.parameters()).device
        rng = np.random.default_rng(0)
        scores = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            idx = np.flatnonzero(y_np == c)
            if idx.size == 0:
                continue
            if idx.size > sample_size_per_class:
                idx = rng.choice(idx, sample_size_per_class, replace=False)
            self._model(torch.from_numpy(X_np[idx]).to(device))
            scores[c] = float(self.composite_stress().mean())
        return scores

    @torch.no_grad()
    def per_sample_stress(
        self,
        X: "torch.Tensor | np.ndarray",
        y: "torch.Tensor | np.ndarray",
        *,
        sample_size: int = 2048,
    ) -> np.ndarray:
        """Per-example stress in [0, 1]. Higher = harder example."""
        X_np = _to_numpy(X) if isinstance(X, torch.Tensor) else np.asarray(X, dtype=np.float32)
        y_np = _to_numpy(y).astype(np.int64) if isinstance(y, torch.Tensor) else np.asarray(y, np.int64)
        n = X_np.shape[0]
        cap = min(n, sample_size)
        if cap < n:
            sel = np.random.default_rng(0).choice(n, cap, replace=False)
            Xw, yw = X_np[sel], y_np[sel]
        else:
            sel, Xw, yw = None, X_np, y_np

        device = next(self._model.parameters()).device
        Xt = torch.from_numpy(Xw).to(device)
        out = self._model(Xt)

        if out.shape[-1] > 1:
            probs = torch.softmax(out, dim=-1)
            true_p = probs[torch.arange(len(yw)), torch.from_numpy(yw).to(device)]
            conf_gap = _to_numpy(1.0 - true_p)
        else:
            conf_gap = np.zeros(len(yw), dtype=np.float32)

        mean_stasis = self.mean_stasis()
        scores = np.clip(0.5 * mean_stasis + 0.5 * conf_gap, 0.0, 1.0)
        if sel is not None:
            full = np.full(n, float(scores.mean()), dtype=np.float32)
            full[sel] = scores
            return full
        return scores

    # ── summary ───────────────────────────────────────────────────────────

    def summary(self) -> str:
        """Human-readable health summary, one line per tracked layer."""
        stasis = self.stasis_rates()
        composite = self.composite_stress()
        lines = [
            f"  {u.name:<40}  [{u.activation_type():<6}]  stasis={stasis[i]:.3f}  composite={composite[i]:.3f}"
            for i, u in enumerate(self._units)
        ]
        return "\n".join(lines) if lines else "(no layers tracked)"
