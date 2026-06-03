"""Layer vitality monitoring for feed-forward classifiers.

Reads four stress signals on hidden units during training:

    stasis          — unit barely activates on the data (dead ReLU, etc.)
    weak_weights    — weight column norm collapsed
    weak_input      — incoming signal much smaller than weights
    saturation      — near-constant activation with large mean

These signals drive the VitalRoute training controller (sampling,
transfer pick, conditional reset). Learning remains on the existing
optimizer (e.g. Adam + backprop).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .backbone.mlp import MLP, _ACTIVATIONS, _softmax


@dataclass
class LayerHealth:
    layer_index: int
    units: int
    activation: str
    clotting: int
    less_immunity: int
    headache: int
    forgetfulness: int

    @property
    def total_sick(self) -> int:
        return self.clotting + self.less_immunity + self.headache + self.forgetfulness


@dataclass
class HealthReport:
    layers: List[LayerHealth]
    resurrected: int = 0

    def summary(self) -> str:
        parts = []
        for h in self.layers:
            parts.append(
                f"L{h.layer_index}({h.units}u): "
                f"clot={h.clotting} less_imm={h.less_immunity} "
                f"head={h.headache} forg={h.forgetfulness}"
            )
        return "  ".join(parts)


# ----------------------------------------------------------------------
# Internal: compute per-layer activations on a sample of inputs
# ----------------------------------------------------------------------

def _layer_activations(mlp: MLP, X: np.ndarray) -> List[np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    a = X
    out: List[np.ndarray] = []
    for i, layer in enumerate(mlp._layers):
        z = a @ layer.W + layer.b
        is_last = i == len(mlp._layers) - 1
        if is_last and mlp.output == "softmax":
            a = _softmax(z)
        else:
            act_fn, _ = _ACTIVATIONS[layer.activation]
            a = act_fn(z)
        out.append(a)
    return out


# ----------------------------------------------------------------------
# Diagnose
# ----------------------------------------------------------------------

def diagnose(
    mlp: MLP,
    X: np.ndarray,
    *,
    dead_activity_threshold: float = 0.01,
    variance_threshold: float = 1e-6,
    weight_threshold: float = 0.05,
    headache_fraction: float = 0.8,
    sample_size: int = 1024,
) -> HealthReport:
    """Diagnose each layer's neurons. Sub-samples X for speed."""
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, sample_size, replace=False)
        X = X[idx]

    activations = _layer_activations(mlp, X)
    abs_x_mean = np.abs(X).mean(axis=0)  # (in_dim,)

    layer_reports: List[LayerHealth] = []
    for li, (layer, act) in enumerate(zip(mlp._layers, activations)):
        # "fires" means non-zero output (true for relu, near-zero detect for tanh/sig)
        if layer.activation == "relu":
            fires = act > 0
        elif layer.activation == "linear":
            fires = np.abs(act) > 1e-6
        else:
            # tanh / sigmoid: "active" when the output is informative (not extreme)
            fires = np.abs(act) > 1e-3
        activity_rate = fires.mean(axis=0)
        variance = act.var(axis=0)
        mean_abs = np.abs(act.mean(axis=0))

        # clotting: dead / silent unit
        clotting = ((activity_rate < dead_activity_threshold) |
                    (variance < variance_threshold)).sum()

        # less_immunity: weight column norm near zero
        col_norm = np.linalg.norm(layer.W, axis=0)
        less_immunity = (col_norm < weight_threshold).sum()

        # headache: most inputs into this unit are smaller in magnitude
        # than the corresponding weights -> "bad signal reception"
        abs_w = np.abs(layer.W)               # (fan_in, units)
        # incoming feature stats: use either raw X (first layer) or previous activation
        if li == 0:
            incoming_mean = abs_x_mean
        else:
            incoming_mean = np.abs(activations[li - 1]).mean(axis=0)
        ratio = (incoming_mean[:, None] < abs_w).mean(axis=0)
        headache = (ratio > headache_fraction).sum()

        # forgetfulness: low variance + non-trivial mean -> stuck at a constant
        forgetfulness = ((variance < variance_threshold) & (mean_abs > 1e-3)).sum()

        layer_reports.append(LayerHealth(
            layer_index=li,
            units=int(layer.W.shape[1]),
            activation=layer.activation,
            clotting=int(clotting),
            less_immunity=int(less_immunity),
            headache=int(headache),
            forgetfulness=int(forgetfulness),
        ))

    return HealthReport(layers=layer_reports)


# ----------------------------------------------------------------------
# Resurrect: re-initialise clotting / starving hidden units.
# The output layer is never modified (its scale matters for softmax).
# ----------------------------------------------------------------------

def _resolve_core(model: object, X: np.ndarray) -> Tuple[MLP, np.ndarray]:
    """MLP passthrough; models with `.head` + `.flatten_features` use the head."""
    if hasattr(model, "head") and hasattr(model, "flatten_features"):
        return model.head, model.flatten_features(X)  # type: ignore[union-attr]
    return model, X  # type: ignore[return-value]


def resurrect_dead(
    mlp: object,
    X: np.ndarray,
    *,
    dead_activity_threshold: float = 0.01,
    variance_threshold: float = 1e-6,
    weight_threshold: float = 0.05,
    sample_size: int = 1024,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """Re-initialise all hidden-layer units that look clotting or starving.
    Returns the count of units resurrected. Output layer is untouched."""
    if rng is None:
        rng = np.random.default_rng()

    n = X.shape[0]
    if n > sample_size:
        idx = rng.choice(n, sample_size, replace=False)
        X = X[idx]

    mlp, X = _resolve_core(mlp, X)
    activations = _layer_activations(mlp, X)
    total = 0
    for li, (layer, act) in enumerate(zip(mlp._layers, activations)):
        # Skip the output layer: re-initialising its weights would
        # destroy what classification ability has been learned.
        if li == len(mlp._layers) - 1:
            continue

        if layer.activation == "relu":
            fires = act > 0
        elif layer.activation == "linear":
            fires = np.abs(act) > 1e-6
        else:
            fires = np.abs(act) > 1e-3
        activity_rate = fires.mean(axis=0)
        variance = act.var(axis=0)
        col_norm = np.linalg.norm(layer.W, axis=0)

        sick = np.flatnonzero(
            (activity_rate < dead_activity_threshold) |
            (variance < variance_threshold) |
            (col_norm < weight_threshold)
        )
        if sick.size == 0:
            continue

        fan_in = layer.W.shape[0]
        if layer.activation == "relu":
            std_in = np.sqrt(2.0 / fan_in)
        else:
            std_in = np.sqrt(1.0 / fan_in)

        for u in sick:
            layer.W[:, u] = rng.normal(0.0, std_in, fan_in).astype(np.float32)
            layer.b[u] = 0.0
            # Also reset the *outgoing* row in the next layer (small init so
            # the freshly-randomised unit doesn't dominate immediately).
            next_layer = mlp._layers[li + 1]
            std_out = np.sqrt(0.5 / next_layer.W.shape[0])
            next_layer.W[u, :] = rng.normal(0.0, std_out, next_layer.W.shape[1]).astype(np.float32)

            # Wipe optimiser momentum for the rewritten column so
            # stale gradients don't immediately undo the resurrection.
            for arr in (layer.mW, layer.vW, layer.velW):
                if arr.size:
                    arr[:, u] = 0.0
            for arr in (layer.mb, layer.vb, layer.velb):
                if arr.size:
                    arr[u] = 0.0

        total += int(sick.size)
    return total


# ======================================================================
# PATHS B & C primitives
# ======================================================================
# The functions below promote the disease detector from a passive monitor
# to an active *signal* that other parts of the system can act on:
#   - per-LAYER stasis rates can feed optional LR scaling (not in router by default)
#   - per-MODEL fingerprints feed transfer parent selection
#   - per-CLASS and per-SAMPLE health feed vitality-weighted sampling
# ======================================================================


def layer_clotting_rates(
    mlp: MLP, X: np.ndarray, *, dead_threshold: float = 0.01,
    sample_size: int = 1024,
) -> np.ndarray:
    """Return a vector of clotting *rates* per layer (0..1).

    A layer's clotting rate is `dead_units / total_units`. The output
    layer is included but typically 0.
    """
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(0)
        X = X[rng.choice(n, sample_size, replace=False)]
    acts = _layer_activations(mlp, X)
    rates = []
    for layer, act in zip(mlp._layers, acts):
        if layer.activation == "relu":
            fires = act > 0
        elif layer.activation == "linear":
            fires = np.abs(act) > 1e-6
        else:
            fires = np.abs(act) > 1e-3
        activity_rate = fires.mean(axis=0)
        dead = (activity_rate < dead_threshold).sum()
        rates.append(float(dead) / float(layer.W.shape[1]))
    return np.array(rates, dtype=np.float32)


def per_class_health(
    mlp: MLP,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    *,
    sample_size_per_class: int = 256,
    dead_threshold: float = 0.01,
) -> np.ndarray:
    """For each class, the *aggregate* clotting rate across all hidden
    layers when only samples of that class are run through the model.

    Returns a (num_classes,) vector where larger = sicker. Classes the
    model handles healthily score near 0; classes that produce many dead
    units score near 1.
    """
    rng = np.random.default_rng(0)
    scores = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            scores[c] = 0.0
            continue
        if idx.size > sample_size_per_class:
            idx = rng.choice(idx, sample_size_per_class, replace=False)
        Xc = X[idx]
        acts = _layer_activations(mlp, Xc)
        # average clotting rate over hidden layers (skip output)
        layer_rates = []
        for li, (layer, act) in enumerate(zip(mlp._layers, acts)):
            if li == len(mlp._layers) - 1:
                continue
            if layer.activation == "relu":
                fires = act > 0
            elif layer.activation == "linear":
                fires = np.abs(act) > 1e-6
            else:
                fires = np.abs(act) > 1e-3
            activity_rate = fires.mean(axis=0)
            dead = (activity_rate < dead_threshold).mean()
            layer_rates.append(float(dead))
        scores[c] = float(np.mean(layer_rates)) if layer_rates else 0.0
    return scores


def _layer_stress_rates(
    mlp: MLP,
    X: np.ndarray,
    *,
    dead_threshold: float = 0.01,
    variance_threshold: float = 1e-6,
    weight_threshold: float = 0.05,
    headache_fraction: float = 0.8,
) -> Tuple[float, float, float, float]:
    """Aggregate hidden-layer fractions for stasis, weak weights, weak input, saturation."""
    acts = _layer_activations(mlp, X)
    abs_x_mean = np.abs(X).mean(axis=0)
    n_hidden_units = 0
    stasis = weak_w = weak_in = sat = 0.0
    for li, (layer, act) in enumerate(zip(mlp._layers, acts)):
        if li == len(mlp._layers) - 1:
            continue
        units = int(layer.W.shape[1])
        n_hidden_units += units
        if layer.activation == "relu":
            fires = act > 0
        elif layer.activation == "linear":
            fires = np.abs(act) > 1e-6
        else:
            fires = np.abs(act) > 1e-3
        activity_rate = fires.mean(axis=0)
        variance = act.var(axis=0)
        mean_abs = np.abs(act.mean(axis=0))
        col_norm = np.linalg.norm(layer.W, axis=0)
        stasis += float(
            ((activity_rate < dead_threshold) | (variance < variance_threshold)).sum()
        )
        weak_w += float((col_norm < weight_threshold).sum())
        abs_w = np.abs(layer.W)
        if li == 0:
            incoming_mean = abs_x_mean
        else:
            incoming_mean = np.abs(acts[li - 1]).mean(axis=0)
        ratio = (incoming_mean[:, None] < abs_w).mean(axis=0)
        weak_in += float((ratio > headache_fraction).sum())
        sat += float(((variance < variance_threshold) & (mean_abs > 1e-3)).sum())
    if n_hidden_units == 0:
        return 0.0, 0.0, 0.0, 0.0
    inv = 1.0 / n_hidden_units
    return stasis * inv, weak_w * inv, weak_in * inv, sat * inv


def per_class_stress(
    mlp: MLP,
    X: np.ndarray,
    y: np.ndarray,
    num_classes: int,
    *,
    sample_size_per_class: int = 256,
    weights: Tuple[float, float, float, float] = (1.0, 0.6, 0.6, 0.5),
) -> np.ndarray:
    """Per-class composite stress in [0, 1] using all four vitality signals.

    ``weights`` apply to (stasis, weak_weights, weak_input, saturation).
    """
    rng = np.random.default_rng(0)
    w_st, w_ww, w_wi, w_sat = weights
    denom = w_st + w_ww + w_wi + w_sat
    scores = np.zeros(num_classes, dtype=np.float32)
    for c in range(num_classes):
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        if idx.size > sample_size_per_class:
            idx = rng.choice(idx, sample_size_per_class, replace=False)
        s0, s1, s2, s3 = _layer_stress_rates(mlp, X[idx])
        scores[c] = (w_st * s0 + w_ww * s1 + w_wi * s2 + w_sat * s3) / denom
    return scores


def per_sample_difficulty(
    mlp: MLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_size: int = 2048,
    dead_threshold: float = 0.01,
) -> np.ndarray:
    """A per-sample difficulty score in [0, 1].

    Combines two health signals per sample:
      - fraction of hidden units that did NOT fire for this sample
        (clotting from the sample's perspective)
      - 1 - probability the model assigned to the true class (confidence)

    Returns a vector of length `min(len(X), sample_size)`. Used by
    curriculum sampling (higher score = harder = oversample).
    """
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, sample_size, replace=False)
        X = X[idx]; y = y[idx]
    acts = _layer_activations(mlp, X)
    # clotting from the sample's perspective: fraction of hidden units that
    # produced ~0 activation for this sample (averaged across hidden layers)
    sample_clot = np.zeros(X.shape[0], dtype=np.float32)
    hidden_layers = 0
    for li, (layer, act) in enumerate(zip(mlp._layers, acts)):
        if li == len(mlp._layers) - 1:
            continue
        if layer.activation == "relu":
            silent = (act <= 0).mean(axis=1)
        elif layer.activation == "linear":
            silent = (np.abs(act) < 1e-6).mean(axis=1)
        else:
            silent = (np.abs(act) < 1e-3).mean(axis=1)
        sample_clot += silent
        hidden_layers += 1
    if hidden_layers > 0:
        sample_clot /= hidden_layers

    # confidence (only meaningful for softmax)
    if mlp.output == "softmax":
        probs = acts[-1]
        true_prob = probs[np.arange(X.shape[0]), y]
        wrong = 1.0 - true_prob
    else:
        wrong = np.zeros(X.shape[0], dtype=np.float32)

    return (0.5 * sample_clot + 0.5 * wrong).astype(np.float32)


def per_sample_stress(
    mlp: MLP,
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_size: Optional[int] = 4096,
    weights: Tuple[float, float, float, float, float] = (0.3, 0.15, 0.15, 0.1, 0.3),
) -> np.ndarray:
    """Per-example stress in [0, 1]: stasis + weak weights/input + saturation + low confidence.

    ``weights`` = (stasis, weak_weights, weak_input, saturation, confidence_gap).
    Returns one score per row of ``X`` (subsamples when ``sample_size`` < n).
    """
    n = X.shape[0]
    cap = n if sample_size is None else min(n, sample_size)
    if cap < n:
        rng = np.random.default_rng(0)
        sel = rng.choice(n, cap, replace=False)
        Xw, yw = X[sel], y[sel]
    else:
        sel = None
        Xw, yw = X, y

    w_st, w_ww, w_wi, w_sat, w_conf = weights
    wsum = w_st + w_ww + w_wi + w_sat + w_conf
    g_st, g_ww, g_wi, g_sat = _layer_stress_rates(mlp, Xw)

    acts = _layer_activations(mlp, Xw)
    n_rows = Xw.shape[0]
    per_st = np.zeros(n_rows, dtype=np.float32)
    per_wi = np.zeros(n_rows, dtype=np.float32)
    per_sat = np.zeros(n_rows, dtype=np.float32)
    n_hidden = 0
    for li, (layer, act) in enumerate(zip(mlp._layers, acts)):
        if li == len(mlp._layers) - 1:
            continue
        n_hidden += 1
        if layer.activation == "relu":
            silent = (act <= 0).mean(axis=1)
        elif layer.activation == "linear":
            silent = (np.abs(act) < 1e-6).mean(axis=1)
        else:
            silent = (np.abs(act) < 1e-3).mean(axis=1)
        per_st += silent
        abs_w = np.abs(layer.W)
        if li == 0:
            incoming = np.abs(Xw)
        else:
            incoming = np.abs(acts[li - 1])
        # per-sample weak input: share of (input, weight) pairs that look starved
        per_wi += (incoming @ (abs_w < incoming.mean() + 1e-6)).mean(axis=1) / max(1, abs_w.shape[1])
        var = act.var(axis=1)
        mean_abs = np.abs(act.mean(axis=1))
        per_sat += ((var < 1e-6) & (mean_abs > 1e-3)).astype(np.float32)
    if n_hidden > 0:
        per_st /= n_hidden
        per_wi /= n_hidden
        per_sat /= n_hidden

    if mlp.output == "softmax":
        probs = acts[-1]
        conf_gap = 1.0 - probs[np.arange(n_rows), yw]
    else:
        conf_gap = np.zeros(n_rows, dtype=np.float32)

    local = (
        w_st * per_st + w_wi * per_wi + w_sat * per_sat + w_conf * conf_gap
        + w_ww * g_ww + w_wi * 0.5 * (per_wi + g_wi) + w_sat * 0.5 * (per_sat + g_sat)
    )
    scores = np.clip(local / wsum, 0.0, 1.0).astype(np.float32)
    if sel is not None:
        full = np.full(n, float(scores.mean()), dtype=np.float32)
        full[sel] = scores
        return full
    return scores


def health_fingerprint(
    mlp: MLP, X: np.ndarray, *, dead_threshold: float = 0.01,
    sample_size: int = 1024,
) -> np.ndarray:
    """A compact (12,) descriptor of a model's health on a dataset.

    Used by transfer pick to compare models by their
    response to the same data without needing to compare weights directly.
    Layout (all in [0, 1]):
      [0] mean activation rate across hidden layers
      [1] std  activation rate across hidden layers
      [2] mean activation magnitude
      [3] std  activation magnitude
      [4] fraction of hidden units that are "dead" (clotting rate)
      [5] mean absolute weight magnitude across hidden layers
      [6] std  absolute weight magnitude across hidden layers
      [7] mean output entropy (softmax) or 0 for regression
      [8] mean L2 of the output activations
      [9] mean variance of hidden activations
      [10] fraction of units near saturation in tanh/sigmoid layers
      [11] global "vital signs" score (lower = sicker)
    """
    n = X.shape[0]
    if n > sample_size:
        rng = np.random.default_rng(0)
        X = X[rng.choice(n, sample_size, replace=False)]
    acts = _layer_activations(mlp, X)
    hidden_acts = acts[:-1] if len(acts) > 1 else acts

    if hidden_acts:
        all_h = np.concatenate([a.ravel() for a in hidden_acts])
        act_rates = []
        for layer, act in zip(mlp._layers[:-1], hidden_acts):
            if layer.activation == "relu":
                fires = act > 0
            else:
                fires = np.abs(act) > 1e-3
            act_rates.append(fires.mean())
        act_rates = np.array(act_rates)
        clot = 0.0
        sat = 0.0
        for layer, act in zip(mlp._layers[:-1], hidden_acts):
            if layer.activation == "relu":
                fires_per_unit = (act > 0).mean(axis=0)
                clot += (fires_per_unit < dead_threshold).mean()
            elif layer.activation in ("tanh", "sigmoid"):
                sat += (np.abs(act) > 0.95).mean()
        clot /= max(1, len(hidden_acts))
        sat /= max(1, len(hidden_acts))
        var_h = float(np.mean([a.var() for a in hidden_acts]))
    else:
        all_h = np.zeros(1)
        act_rates = np.zeros(1)
        clot = 0.0
        sat = 0.0
        var_h = 0.0

    w_mags = [float(np.abs(l.W).mean()) for l in mlp._layers[:-1]] or [0.0]

    out = acts[-1]
    if mlp.output == "softmax":
        ent = float((-out * np.log(np.clip(out, 1e-12, 1.0))).sum(axis=1).mean())
    else:
        ent = 0.0
    out_l2 = float(np.linalg.norm(out, axis=1).mean())

    fp = np.array([
        float(act_rates.mean()),
        float(act_rates.std()),
        float(np.abs(all_h).mean()),
        float(np.abs(all_h).std()),
        float(clot),
        float(np.mean(w_mags)),
        float(np.std(w_mags)),
        ent,
        out_l2,
        var_h,
        float(sat),
        1.0 - float(clot),
    ], dtype=np.float32)
    return fp


def fingerprint_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Lower = more similar. Combines cosine distance with L2 to be
    sensitive to both shape and scale of the two fingerprints."""
    a = np.asarray(a, dtype=np.float32); b = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    cos_dist = 1.0 - float(np.dot(a, b) / (na * nb))
    l2 = float(np.linalg.norm(a - b))
    return 0.7 * cos_dist + 0.3 * l2 / max(1.0, na)


def diagnose_any(model: object, X: np.ndarray, **kwargs) -> HealthReport:
    """Diagnose layer health on an MLP or ConvNet (head layers for CNN)."""
    mlp, Xh = _resolve_core(model, X)
    return diagnose(mlp, Xh, **kwargs)


def layer_clotting_rates_any(model: object, X: np.ndarray, **kwargs) -> np.ndarray:
    mlp, Xh = _resolve_core(model, X)
    return layer_clotting_rates(mlp, Xh, **kwargs)


# Public name: stasis rate per layer (same probe as legacy "clotting" in research code)
layer_stasis_rates_any = layer_clotting_rates_any
