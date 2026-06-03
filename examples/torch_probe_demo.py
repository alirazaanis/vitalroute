"""Quick smoke-test: VitalityProbe on a standard PyTorch MLP.

Command:
    python examples/torch_probe_demo.py
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.optim import Adam
except ImportError:
    print("PyTorch not installed — skipping torch_probe_demo.")
    raise SystemExit(0)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from vitalroute.torch_probe import VitalityProbe


def make_mlp(input_dim: int, num_classes: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 128), nn.ReLU(),
        nn.Linear(128, 64),        nn.ReLU(),
        nn.Linear(64, num_classes),
    )


def main():
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

    model = make_mlp(64, 10)
    opt   = Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    probe = VitalityProbe(model)
    print("Tracked layers:")
    for name in probe.layer_names():
        print(f"  {name}")

    Xt = torch.from_numpy(Xtr)
    yt = torch.from_numpy(ytr).long()

    for epoch in range(10):
        model.train()
        perm = torch.randperm(len(yt))
        for s in range(0, len(yt), 32):
            b = perm[s:s + 32]
            opt.zero_grad()
            loss_fn(model(Xt[b]), yt[b]).backward()
            opt.step()

        # Observe activations once per epoch
        probe.observe(Xt)

        if epoch % 3 == 0 or epoch == 9:
            model.eval()
            with torch.no_grad():
                preds = model(torch.from_numpy(Xv)).argmax(dim=1).numpy()
            acc = (preds == yv).mean()
            print(f"\nepoch {epoch + 1:2d}  val_acc={acc:.4f}  mean_stasis={probe.mean_stasis():.4f}")
            print(probe.summary())

    probe.detach()
    print("\nDone.")


if __name__ == "__main__":
    main()
