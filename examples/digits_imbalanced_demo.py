"""Minimal demo: vitality sampler on sklearn digits (imbalanced split)."""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from vitalroute import adaptive_controller, profile_task, route_plan
from vitalroute.backbone import MLP, LayerSpec


def main() -> None:
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    # Imbalance: keep only 20 samples for classes 5-9
    rng = np.random.default_rng(0)
    keep = []
    for c in range(10):
        idx = np.flatnonzero(ytr == c)
        k = 145 if c < 5 else 20
        keep.append(rng.choice(idx, min(k, idx.size), replace=False))
    keep = np.concatenate(keep)
    Xtr, ytr = Xtr[keep], ytr[keep]

    prof = profile_task(ytr, 10)
    plan = route_plan(prof, parent_pool_available=False)
    print(f"route: {plan.label}  (n={prof.n_samples}, imb={prof.imbalance_ratio:.3f})")

    model = MLP(
        input_dim=64,
        layers=[LayerSpec(64, "relu"), LayerSpec(32, "relu"), LayerSpec(10, "linear")],
        output="softmax",
        seed=0,
    )
    ctrl = adaptive_controller(ytr, 10, verbose=True)
    opt = ctrl.make_optimizer("adam", lr=1e-3)
    sampler, _ = ctrl.bootstrap(model, Xtr, ytr, num_classes=10)

    for epoch in range(15):
        ctrl.on_epoch_start(model, Xtr, opt, epoch)
        n = len(ytr)
        if sampler is not None:
            idx = sampler.sample_indices(epoch, model, Xtr, ytr, n)
            Xep, yep = Xtr[idx], ytr[idx]
        else:
            Xep, yep = Xtr, ytr
        for start in range(0, n, 32):
            model.train_step(Xep[start:start + 32], yep[start:start + 32], opt)
        acc = float((model.predict(Xv) == yv).mean())
        ctrl.after_epoch(model, Xtr, np.random.default_rng(epoch), verbose=(epoch % 5 == 0))
        if epoch % 5 == 0 or epoch == 14:
            print(f"epoch {epoch+1}  val_acc={acc:.4f}")

    print("done.")


if __name__ == "__main__":
    main()
