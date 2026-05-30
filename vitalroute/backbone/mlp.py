"""A pure-NumPy multilayer perceptron with backprop.

Supports:
  * arbitrary number of hidden layers, configurable widths
  * activations: relu, tanh, sigmoid, linear
  * output heads: softmax (classification) or linear (regression)
  * losses: cross_entropy (with softmax) or mse
  * He / Glorot initialisation
  * Optimizers: SGD with momentum, and Adam
  * L2 weight decay

Designed for clarity over raw speed. Handles tens of thousands of
samples and dimensions in the low thousands comfortably on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def _relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def _tanh_grad(x: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(x) ** 2


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable sigmoid
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _sigmoid_grad(x: np.ndarray) -> np.ndarray:
    s = _sigmoid(x)
    return s * (1.0 - s)


def _linear(x: np.ndarray) -> np.ndarray:
    return x


def _linear_grad(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


_ACTIVATIONS = {
    "relu": (_relu, _relu_grad),
    "tanh": (_tanh, _tanh_grad),
    "sigmoid": (_sigmoid, _sigmoid_grad),
    "linear": (_linear, _linear_grad),
}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------
# Layer
# ----------------------------------------------------------------------

@dataclass
class LayerSpec:
    units: int
    activation: str = "relu"


@dataclass
class _Layer:
    W: np.ndarray
    b: np.ndarray
    activation: str
    # Adam state
    mW: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vW: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    vb: np.ndarray = field(default_factory=lambda: np.zeros(0))
    # SGD-momentum state
    velW: np.ndarray = field(default_factory=lambda: np.zeros(0))
    velb: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ----------------------------------------------------------------------
# Optimizers
# ----------------------------------------------------------------------

@dataclass
class SGD:
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0

    def step(self, layer: _Layer, gW: np.ndarray, gb: np.ndarray) -> None:
        if layer.velW.size == 0:
            layer.velW = np.zeros_like(layer.W)
            layer.velb = np.zeros_like(layer.b)
        if self.weight_decay:
            gW = gW + self.weight_decay * layer.W
        layer.velW = self.momentum * layer.velW - self.lr * gW
        layer.velb = self.momentum * layer.velb - self.lr * gb
        layer.W += layer.velW
        layer.b += layer.velb


@dataclass
class Adam:
    lr: float = 1e-3
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.0
    _t: int = 0

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
        layer.W -= self.lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
        layer.b -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)


# ----------------------------------------------------------------------
# MLP
# ----------------------------------------------------------------------

class MLP:
    """A simple feed-forward multilayer perceptron.

    Examples
    --------
    >>> mlp = MLP(input_dim=4, layers=[LayerSpec(8, "relu"), LayerSpec(3, "linear")], output="softmax")
    >>> probs = mlp.forward(X)            # shape (N, 3) summing to 1
    """

    def __init__(
        self,
        input_dim: int,
        layers: Sequence[LayerSpec],
        output: str = "softmax",
        *,
        seed: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        if not layers:
            raise ValueError("need at least one layer")
        if output not in {"softmax", "linear"}:
            raise ValueError("output must be 'softmax' or 'linear'")
        self.input_dim = input_dim
        self.output = output
        self.dropout = float(dropout)
        self._dropout_mask: Optional[np.ndarray] = None
        rng = np.random.default_rng(seed)
        self._layers: List[_Layer] = []
        prev = input_dim
        for spec in layers:
            # He init for relu-like, Glorot for tanh/sigmoid
            if spec.activation == "relu":
                std = np.sqrt(2.0 / prev)
            else:
                std = np.sqrt(1.0 / prev)
            W = rng.normal(0.0, std, size=(prev, spec.units)).astype(np.float32)
            b = np.zeros(spec.units, dtype=np.float32)
            self._layers.append(_Layer(W=W, b=b, activation=spec.activation))
            prev = spec.units
        self._cache: List[Tuple[np.ndarray, np.ndarray]] = []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray, *, train: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        cache: List[Tuple[np.ndarray, np.ndarray]] = []
        a = X
        for i, layer in enumerate(self._layers):
            z = a @ layer.W + layer.b
            is_last = i == len(self._layers) - 1
            if is_last and self.output == "softmax":
                a_next = _softmax(z)
            else:
                act_fn, _ = _ACTIVATIONS[layer.activation]
                a_next = act_fn(z)
                if train and self.dropout > 0 and not is_last:
                    keep = 1.0 - self.dropout
                    mask = (np.random.rand(*a_next.shape) < keep).astype(a_next.dtype) / keep
                    a_next = a_next * mask
            if train:
                cache.append((a, z))
            a = a_next
        if train:
            self._cache = cache
        return a

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = self.forward(X)
        if self.output == "softmax":
            return out.argmax(axis=1)
        return out

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.output != "softmax":
            raise ValueError("predict_proba only valid for softmax output")
        return self.forward(X)

    # ------------------------------------------------------------------
    # Loss + backward
    # ------------------------------------------------------------------

    @staticmethod
    def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
        oh = np.zeros((y.size, num_classes), dtype=np.float32)
        oh[np.arange(y.size), y] = 1.0
        return oh

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        out = self.forward(X)
        if self.output == "softmax":
            y = np.asarray(y, dtype=np.int64)
            # categorical cross-entropy
            p = np.clip(out[np.arange(y.size), y], 1e-12, 1.0)
            return float(-np.log(p).mean())
        y = np.asarray(y, dtype=np.float32)
        return float(((out - y) ** 2).mean())

    def _backward(
        self,
        y: np.ndarray,
        optimizer,
    ) -> None:
        """Compute gradients from the most recent forward(train=True) and
        apply the optimizer step to every layer."""
        if not self._cache:
            raise RuntimeError("call forward(..., train=True) before _backward")
        # Output gradient
        last_layer = self._layers[-1]
        a_last, z_last = self._cache[-1]
        if self.output == "softmax":
            y = np.asarray(y, dtype=np.int64)
            out = _softmax(z_last)
            n = y.shape[0]
            # softmax + cross-entropy gradient simplifies to (p - y_onehot)/N
            dZ = out.copy()
            dZ[np.arange(n), y] -= 1.0
            dZ /= n
        else:
            y = np.asarray(y, dtype=np.float32)
            act_fn, act_grad = _ACTIVATIONS[last_layer.activation]
            out = act_fn(z_last)
            dA = 2.0 * (out - y) / max(1, y.shape[0])
            dZ = dA * act_grad(z_last)

        # Walk backwards
        grads: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(len(self._layers) - 1, -1, -1):
            layer = self._layers[i]
            a_prev, z = self._cache[i]
            gW = a_prev.T @ dZ
            gb = dZ.sum(axis=0)
            grads.append((gW, gb))
            if i > 0:
                dA_prev = dZ @ layer.W.T
                prev_layer = self._layers[i - 1]
                _, prev_grad = _ACTIVATIONS[prev_layer.activation]
                _, z_prev = self._cache[i - 1]
                dZ = dA_prev * prev_grad(z_prev)
        grads.reverse()
        for layer, (gW, gb) in zip(self._layers, grads):
            optimizer.step(layer, gW, gb)
        self._cache = []

    # ------------------------------------------------------------------
    # Convenience: single training step
    # ------------------------------------------------------------------

    def train_step(self, X: np.ndarray, y: np.ndarray, optimizer) -> float:
        self.forward(X, train=True)
        loss = self.loss(X, y)
        self._backward(y, optimizer)
        return loss

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        blob = {
            "input_dim": self.input_dim,
            "output": self.output,
            "dropout": self.dropout,
            "n_layers": len(self._layers),
        }
        for i, layer in enumerate(self._layers):
            blob[f"W{i}"] = layer.W
            blob[f"b{i}"] = layer.b
            blob[f"act{i}"] = layer.activation
        np.savez(path, **blob)

    @classmethod
    def load(cls, path: str) -> "MLP":
        z = np.load(path, allow_pickle=False)
        input_dim = int(z["input_dim"])
        output = str(z["output"])
        n_layers = int(z["n_layers"])
        specs = [LayerSpec(units=int(z[f"W{i}"].shape[1]), activation=str(z[f"act{i}"])) for i in range(n_layers)]
        mlp = cls(input_dim=input_dim, layers=specs, output=output,
                  dropout=float(z["dropout"]) if "dropout" in z else 0.0)
        for i, layer in enumerate(mlp._layers):
            layer.W = z[f"W{i}"]
            layer.b = z[f"b{i}"]
        return mlp

    # ------------------------------------------------------------------
    # Inheritance (the spec's tt/yy/ty/yt parent->child weight copy).
    # In modern terms: warm-start / transfer learning between MLPs.
    # ------------------------------------------------------------------

    def inherit_from(
        self,
        parent: "MLP",
        *,
        mode: str = "yy",
        fresh_head: bool = True,
        verbose: bool = False,
    ) -> dict:
        """Copy parameters from `parent` into self using the spec's mode.

        Modes (from `Neural Network.txt` line 201):
          yy  copy both weights and biases   (full warm-start)
          tt  copy biases only               (the spec's "thresholds")
          yt  copy weights only              (re-randomise biases)
          ty  copy biases + randomise weights (rarely useful, kept for parity)

        `fresh_head` (default True) leaves the OUTPUT layer at its
        random initialisation. This is the standard transfer-learning
        idiom: a parent's classification head is tuned for the parent's
        label set, so reusing it on a new task is usually harmful.

        Layers whose shapes mismatch the parent are SKIPPED rather than
        raising, so warm-start works between models that share only a
        trunk.
        """
        if mode not in {"yy", "tt", "yt", "ty"}:
            raise ValueError(f"mode must be one of yy/tt/yt/ty, got {mode!r}")
        if self.input_dim != parent.input_dim:
            raise ValueError(
                f"input_dim mismatch: self={self.input_dim} parent={parent.input_dim}"
            )

        inherited = 0
        skipped: List[int] = []
        head_kept_fresh = False
        rng = np.random.default_rng()
        last_idx = len(self._layers) - 1
        for i, (child, par) in enumerate(zip(self._layers, parent._layers)):
            if fresh_head and i == last_idx:
                head_kept_fresh = True
                continue
            if child.W.shape != par.W.shape:
                skipped.append(i)
                if verbose:
                    print(f"  [inherit] layer {i} skipped: "
                          f"child={child.W.shape} parent={par.W.shape}")
                continue
            if mode in ("yy", "yt"):
                child.W = par.W.copy()
            if mode in ("yy", "tt"):
                child.b = par.b.copy()
            if mode == "ty":
                fan_in = child.W.shape[0]
                std = np.sqrt(2.0 / fan_in)
                child.W = rng.normal(0.0, std, child.W.shape).astype(np.float32)
                child.b = par.b.copy()
            if mode == "yt":
                child.b = np.zeros_like(par.b)
            # Reset optimiser state so warm-started weights start clean.
            child.mW = np.zeros(0); child.vW = np.zeros(0)
            child.mb = np.zeros(0); child.vb = np.zeros(0)
            child.velW = np.zeros(0); child.velb = np.zeros(0)
            inherited += 1

        return {
            "mode": mode,
            "inherited": inherited,
            "skipped": skipped,
            "total_layers": len(self._layers),
            "head_kept_fresh": head_kept_fresh,
        }

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def n_parameters(self) -> int:
        return sum(l.W.size + l.b.size for l in self._layers)

    def architecture(self) -> List[int]:
        return [self.input_dim] + [l.W.shape[1] for l in self._layers]

    def __repr__(self) -> str:
        arch = "->".join(map(str, self.architecture()))
        return f"MLP({arch}, out={self.output}, params={self.n_parameters})"
