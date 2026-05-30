"""PyTorch-native adaptive training controller.

``TorchTrainingController`` is the PyTorch counterpart of
``TrainingController``. It reads your dataset shape, chooses tactics
(vitality sampler, hard-sample sampler, LR dampening, unit monitoring),
and hands back a ``DataLoader``-compatible sampler and a thin epoch hook.

Your training loop stays untouched — you only add three calls:

    ctrl    = torch_adaptive_controller(y_train, num_classes)
    sampler = ctrl.setup(model, X_train, y_train)

    for epoch in range(epochs):
        ctrl.on_epoch_start(model, X_train, optimizer, epoch)
        loader = DataLoader(dataset, sampler=sampler, batch_size=64)
        for X_batch, y_batch in loader:
            ...  # your normal loss + backward + step
        ctrl.after_epoch(model, X_train, y_train)

``ctrl.setup()`` returns ``None`` when no sampler is needed (balanced
dataset); in that case use a standard ``DataLoader`` without a sampler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_controller. "
        "Install it with: pip install torch"
    ) from exc

from .router import profile_task, route_plan
from .torch_probe import VitalityProbe
from .torch_samplers import TorchHardSampleSampler, TorchVitalitySampler


EpochSampler = Union[TorchVitalitySampler, TorchHardSampleSampler]


@dataclass
class TorchTrainingController:
    """Adaptive training controller for PyTorch models.

    Instantiate via ``torch_adaptive_controller()`` rather than directly.
    """

    routing_label: str = "monitor"
    use_imbalance_sampler: bool = False
    use_hard_sample_sampler: bool = False
    use_lr_scale: bool = False
    lr_scale_alpha: float = 4.0
    lr_scale_min: float = 0.1
    lr_scale_refresh_every: int = 1

    sampler_strength: float = 0.7
    sampler_beta: float = 4.0
    sampler_refresh_every: int = 2

    hard_strength: float = 0.6
    hard_beta: float = 3.0
    hard_refresh_every: int = 1

    monitor: bool = True
    verbose: bool = False

    _probe: Optional[VitalityProbe] = field(default=None, init=False)
    _sampler: Optional[EpochSampler] = field(default=None, init=False)
    _y_train: Optional[np.ndarray] = field(default=None, init=False)
    _epoch: int = field(default=0, init=False)
    _lr_scales: dict = field(default_factory=dict, init=False)

    # ── setup ──────────────────────────────────────────────────────────────

    def setup(
        self,
        model: nn.Module,
        X_probe: "torch.Tensor | np.ndarray",
        y_probe: "torch.Tensor | np.ndarray",
        *,
        y_full: Optional["torch.Tensor | np.ndarray"] = None,
        num_classes: int,
        seed: int = 0,
    ) -> Optional[EpochSampler]:
        """Attach probe, build sampler. Call once before the training loop.

        Parameters
        ----------
        X_probe / y_probe:
            A representative (stratified) subset of the training data used
            to run the vitality probe. Should have samples from every class.
            Typically 50–100 per class is sufficient.
        y_full:
            Full training labels (one per training example). Used to build
            the sampler's class pools. If ``None``, ``y_probe`` is used —
            only correct when ``X_probe`` covers the whole training set.

        Returns the sampler (or ``None`` if routing chose no sampler).
        """
        self._probe = VitalityProbe(model)
        self._probe.observe(X_probe)

        if isinstance(y_probe, torch.Tensor):
            probe_labels = y_probe.cpu().numpy().astype(np.int64)
        else:
            probe_labels = np.asarray(y_probe, dtype=np.int64)
        self._y_train = probe_labels   # stored for probe refresh in on_epoch_start

        if y_full is not None:
            if isinstance(y_full, torch.Tensor):
                sampler_labels = y_full.cpu().numpy().astype(np.int64)
            else:
                sampler_labels = np.asarray(y_full, dtype=np.int64)
        else:
            sampler_labels = probe_labels

        if self.verbose:
            print(f"  [vitalroute] route={self.routing_label}  "
                  f"probe on {len(self._probe.layer_names())} layers  "
                  f"(probe_n={len(probe_labels)}  full_n={len(sampler_labels)})")

        if self.use_imbalance_sampler and num_classes > 1:
            self._sampler = TorchVitalitySampler(
                sampler_labels, num_classes, self._probe,
                strength=self.sampler_strength,
                beta=self.sampler_beta,
                seed=seed,
            )
            if self.verbose:
                print("  [vitalroute] vitality class sampler on")
        elif self.use_hard_sample_sampler:
            self._sampler = TorchHardSampleSampler(
                len(sampler_labels), self._probe,
                strength=self.hard_strength,
                beta=self.hard_beta,
                seed=seed,
            )
            if self.verbose:
                print("  [vitalroute] hard-sample sampler on")

        if self.use_lr_scale and self.verbose:
            print(f"  [vitalroute] lr_scale on (alpha={self.lr_scale_alpha})")

        return self._sampler

    # ── epoch hooks ────────────────────────────────────────────────────────

    def on_epoch_start(
        self,
        model: nn.Module,
        X_train: "torch.Tensor | np.ndarray",
        optimizer: "torch.optim.Optimizer",
        epoch: int,
    ) -> None:
        """Refresh probe, update LR scales and sampler weights.

        Call at the top of each epoch before building the DataLoader.
        """
        self._epoch = epoch
        if self._probe is None:
            return

        if epoch % max(1, self.lr_scale_refresh_every) == 0:
            self._probe.observe(X_train)

        if self.use_lr_scale:
            self._apply_lr_scales(model, optimizer)

        if self._sampler is not None and self._y_train is not None:
            refresh_every = (
                self.sampler_refresh_every
                if self.use_imbalance_sampler
                else self.hard_refresh_every
            )
            if epoch % max(1, refresh_every) == 0:
                if isinstance(X_train, np.ndarray):
                    X_t = torch.from_numpy(X_train)
                else:
                    X_t = X_train
                y_t = torch.from_numpy(self._y_train)
                self._sampler.refresh(X_t, y_t, model)

    def _apply_lr_scales(
        self, model: nn.Module, optimizer: "torch.optim.Optimizer"
    ) -> None:
        """Scale per-layer LR by 1 / (1 + alpha * stasis_rate)."""
        if self._probe is None:
            return
        rates = self._probe.stasis_rates()
        names = self._probe.layer_names()
        scale_map = {}
        for name, rate in zip(names, rates):
            scale = max(self.lr_scale_min, 1.0 / (1.0 + self.lr_scale_alpha * float(rate)))
            scale_map[name] = scale

        mean_scale = float(np.mean(list(scale_map.values()))) if scale_map else 1.0
        for pg in optimizer.param_groups:
            if "name" in pg and pg["name"] in scale_map:
                if "base_lr" not in pg:
                    pg["base_lr"] = pg["lr"]
                pg["lr"] = pg["base_lr"] * scale_map[pg["name"]]
        self._lr_scales = scale_map
        if self.verbose and scale_map:
            vec = [f"{v:.2f}" for v in scale_map.values()]
            print(f"    [vitalroute] lr_scale: [{', '.join(vec)}]  mean={mean_scale:.2f}")

    def after_epoch(
        self,
        model: nn.Module,
        X_train: "torch.Tensor | np.ndarray",
        y_train: "torch.Tensor | np.ndarray",
    ) -> dict:
        """Log vitality after the epoch. Returns stasis and composite stats."""
        if self._probe is None:
            return {}
        self._probe.observe(X_train)
        stasis = self._probe.mean_stasis()
        composite = self._probe.composite_stress()
        info = {"mean_stasis": stasis, "composite_mean": float(composite.mean())}
        if self.verbose:
            print(f"    [vitalroute] {self._probe.summary()}")
        return info

    def detach(self) -> None:
        """Remove forward hooks. Call after training is complete."""
        if self._probe is not None:
            self._probe.detach()
            self._probe = None

    @property
    def probe(self) -> Optional[VitalityProbe]:
        return self._probe


# ── factory ──────────────────────────────────────────────────────────────────

def torch_adaptive_controller(
    y_train: "torch.Tensor | np.ndarray",
    num_classes: int,
    *,
    verbose: bool = False,
    sampler_strength: float = 0.7,
    sampler_beta: float = 4.0,
    sampler_refresh_every: int = 2,
    lr_scale_alpha: float = 4.0,
    imbalance_threshold: float = 0.25,
    min_class_for_sampler: int = 15,
    max_samples_for_transfer: int = 200,
    min_samples_for_lr_scale: int = 80,
    min_samples_for_hard_sampler: int = 40,
) -> TorchTrainingController:
    """Build a ``TorchTrainingController`` with tactics chosen from class counts.

    Mirrors ``adaptive_controller()`` from the NumPy API.
    Transfer pick is skipped (PyTorch models handle that separately).
    """
    if isinstance(y_train, torch.Tensor):
        y_np = y_train.cpu().numpy().astype(np.int64)
    else:
        y_np = np.asarray(y_train, dtype=np.int64)

    profile = profile_task(y_np, num_classes)
    plan = route_plan(
        profile,
        parent_pool_available=False,
        imbalance_threshold=imbalance_threshold,
        min_class_for_sampler=min_class_for_sampler,
        max_samples_for_transfer=max_samples_for_transfer,
        min_samples_for_lr_scale=min_samples_for_lr_scale,
        min_samples_for_hard_sampler=min_samples_for_hard_sampler,
    )

    if verbose:
        print(
            f"  [vitalroute] n={profile.n_samples}  "
            f"classes={profile.min_per_class}-{profile.max_per_class}  "
            f"imb={profile.imbalance_ratio:.3f}  -> {plan.label}"
        )

    return TorchTrainingController(
        routing_label=plan.label,
        use_imbalance_sampler=plan.use_imbalance_sampler,
        use_hard_sample_sampler=plan.use_hard_sample_sampler,
        use_lr_scale=plan.use_lr_scale,
        lr_scale_alpha=lr_scale_alpha,
        sampler_strength=sampler_strength,
        sampler_beta=sampler_beta,
        sampler_refresh_every=sampler_refresh_every,
        monitor=True,
        verbose=verbose,
    )
