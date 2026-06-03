"""Baseline comparison for VitalRoute on imbalanced classification.

Four conditions, same architecture and data split:
  1. uniform       — plain training, no sampling
  2. inv_freq      — inverse class-frequency weighting in loss
  3. vitalroute    — adaptive_controller with composite vitality sampler
  4. stasis_only   — VitalitySampler in legacy stasis-only mode

Metrics reported:
  - overall val accuracy
  - minority-class accuracy (classes with < min_frac of majority)
  - per-class accuracy table

Command:
    python examples/benchmark_baselines.py
"""

from __future__ import annotations

import textwrap
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from vitalroute import adaptive_controller
from vitalroute.backbone import MLP, LayerSpec, Adam
from vitalroute.imbalance import VitalitySampler


# ── dataset ──────────────────────────────────────────────────────────────────

def make_imbalanced(seed: int = 0, minority_n: int = 20):
    """Digits with majority classes 0-4 (145 each) and minority 5-9 (20 each)."""
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    rng = np.random.default_rng(seed)
    keep = []
    for c in range(10):
        idx = np.flatnonzero(ytr == c)
        k = 145 if c < 5 else minority_n
        keep.append(rng.choice(idx, min(k, idx.size), replace=False))
    keep = np.concatenate(keep)
    return Xtr[keep], ytr[keep], Xv, yv


def fresh_model(seed: int = 0) -> MLP:
    return MLP(
        input_dim=64,
        layers=[LayerSpec(128, "relu"), LayerSpec(64, "relu"), LayerSpec(10, "linear")],
        output="softmax",
        seed=seed,
    )


# ── training loops ────────────────────────────────────────────────────────────

def _epoch_batched(model, X, y, opt, batch=32):
    idx = np.random.permutation(len(y))
    for s in range(0, len(y), batch):
        b = idx[s:s + batch]
        model.train_step(X[b], y[b], opt)


def run_uniform(Xtr, ytr, Xv, yv, *, epochs=30, lr=1e-3, seed=0):
    model = fresh_model(seed)
    opt = Adam(lr=lr)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        _epoch_batched(model, Xtr, ytr, opt)
    return model


def run_inv_freq(Xtr, ytr, Xv, yv, *, epochs=30, lr=1e-3, seed=0):
    """Cross-entropy with per-sample inverse-frequency weighting."""
    counts = np.bincount(ytr, minlength=10).astype(np.float32)
    class_w = 1.0 / np.maximum(counts, 1)
    class_w /= class_w.sum()
    sample_w = class_w[ytr]
    sample_w /= sample_w.sum()

    model = fresh_model(seed)
    opt = Adam(lr=lr)
    rng = np.random.default_rng(seed)
    n = len(ytr)
    for ep in range(epochs):
        idx = rng.choice(n, size=n, replace=True, p=sample_w)
        _epoch_batched(model, Xtr[idx], ytr[idx], opt)
    return model


def run_vitalroute(Xtr, ytr, Xv, yv, *, epochs=30, lr=1e-3, seed=0,
                   stress_mode="composite"):
    model = fresh_model(seed)
    ctrl = adaptive_controller(ytr, 10, verbose=False,
                                sampler_stress_mode=stress_mode)
    opt = ctrl.make_optimizer("adam", lr=lr)
    sampler, _ = ctrl.bootstrap(model, Xtr, ytr, num_classes=10, seed=seed)
    rng = np.random.default_rng(seed)
    n = len(ytr)
    for ep in range(epochs):
        ctrl.on_epoch_start(model, Xtr, opt, ep)
        if sampler is not None:
            idx = sampler.sample_indices(ep, model, Xtr, ytr, n)
            Xep, yep = Xtr[idx], ytr[idx]
        else:
            Xep, yep = Xtr, ytr
        _epoch_batched(model, Xep, yep, opt)
        ctrl.after_epoch(model, Xtr, rng)
    return model


# ── evaluation ────────────────────────────────────────────────────────────────

@dataclass
class Result:
    name: str
    overall: float
    minority_acc: float
    per_class: np.ndarray


def evaluate(model, Xv, yv, minority_classes=(5, 6, 7, 8, 9)) -> Dict[str, float]:
    preds = model.predict(Xv)
    overall = float((preds == yv).mean())
    per_class = np.zeros(10)
    for c in range(10):
        mask = yv == c
        per_class[c] = float((preds[mask] == c).mean()) if mask.sum() > 0 else 0.0
    minority_acc = float(per_class[list(minority_classes)].mean())
    return overall, minority_acc, per_class


# ── main ──────────────────────────────────────────────────────────────────────

def run_trial(seed: int, minority_n: int = 20, epochs: int = 30):
    Xtr, ytr, Xv, yv = make_imbalanced(seed=seed, minority_n=minority_n)

    results: List[Result] = []
    for name, fn, kwargs in [
        ("uniform",      run_uniform,    {}),
        ("inv_freq",     run_inv_freq,   {}),
        ("vitalroute",   run_vitalroute, {"stress_mode": "composite"}),
        ("stasis_only",  run_vitalroute, {"stress_mode": "stasis"}),
    ]:
        model = fn(Xtr, ytr, Xv, yv, epochs=epochs, seed=seed, **kwargs)
        overall, minority_acc, per_class = evaluate(model, Xv, yv)
        results.append(Result(name, overall, minority_acc, per_class))

    return results


def print_table(all_results: List[List[Result]]):
    n_trials = len(all_results)
    names = [r.name for r in all_results[0]]

    print(f"\n{'=' * 60}")
    print(f"  Benchmark — {n_trials} trial(s), imbalanced digits (5:1 ratio)")
    print(f"  Majority classes 0-4 (~145 each)  |  Minority 5-9 (~20 each)")
    print(f"{'=' * 60}")
    print(f"{'Method':<16}  {'Overall':>8}  {'Minority':>8}")
    print(f"{'-' * 36}")

    for i, name in enumerate(names):
        overall_vals = [all_results[t][i].overall for t in range(n_trials)]
        minor_vals   = [all_results[t][i].minority_acc for t in range(n_trials)]
        o_mean = np.mean(overall_vals)
        m_mean = np.mean(minor_vals)
        o_std  = np.std(overall_vals)
        m_std  = np.std(minor_vals)
        print(f"{name:<16}  {o_mean:>7.1%}±{o_std:.1%}  {m_mean:>7.1%}±{m_std:.1%}")

    print(f"{'=' * 60}")

    print("\nPer-class accuracy (mean over trials):")
    header = f"{'Method':<16}" + "".join(f"  c{c}" for c in range(10))
    print(header)
    print("-" * len(header))
    for i, name in enumerate(names):
        per_class_mean = np.mean(
            [all_results[t][i].per_class for t in range(n_trials)], axis=0
        )
        row = f"{name:<16}" + "".join(f"  {v:>4.0%}" for v in per_class_mean)
        print(row)
    print()


def main(n_trials: int = 3, epochs: int = 30, minority_n: int = 20):
    print(f"Running {n_trials} trial(s), {epochs} epochs each ...")
    all_results = []
    for seed in range(n_trials):
        print(f"  trial {seed + 1}/{n_trials}", end="\r")
        all_results.append(run_trial(seed=seed, minority_n=minority_n, epochs=epochs))
    print_table(all_results)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--trials",   type=int, default=3,  help="number of random seeds")
    p.add_argument("--epochs",   type=int, default=30, help="training epochs per trial")
    p.add_argument("--minority", type=int, default=20, help="samples per minority class")
    args = p.parse_args()
    main(n_trials=args.trials, epochs=args.epochs, minority_n=args.minority)
