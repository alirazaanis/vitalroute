"""PyTorch-native adaptive training controller.

``TorchTrainingController`` is the PyTorch counterpart of
``TrainingController``. It reads the dataset shape, chooses tactics
(vitality sampler, hard-sample sampler, LR dampening, unit monitoring),
and hands back a ``DataLoader``-compatible sampler and a thin epoch hook.

CNN support includes stratified probe batches (``torch_data``), ``CNNVitalityProbe``,
PyTorch transfer pick (``torch_transfer``), and per-layer LR param groups
(``torch_lr_scale`` / ``make_vitality_optimizer``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple, Union

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
from .torch_lr_scale import apply_lr_scales, make_vitality_optimizer
from .torch_probes import Architecture, ProbeZone, make_probe
from .torch_samplers import TorchHardSampleSampler, TorchVitalitySampler
from .torch_transfer import pick_transfer_parent_torch, warm_start_from_parent

EpochSampler = Union[TorchVitalitySampler, TorchHardSampleSampler]


@dataclass
class TorchTrainingController:
    """Adaptive training controller for PyTorch models."""

    routing_label: str = "monitor"
    use_imbalance_sampler: bool = False
    use_hard_sample_sampler: bool = False
    use_transfer_pick: bool = False
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

    architecture: Architecture = "auto"
    probe_zone: ProbeZone = "all"
    parent_pool: Optional[List[Tuple[str, nn.Module]]] = None

    monitor: bool = True
    verbose: bool = False

    picked_parent_name: Optional[str] = field(default=None, init=False)
    picked_parent_stasis: Optional[float] = field(default=None, init=False)

    _probe: Optional[object] = field(default=None, init=False)
    _sampler: Optional[EpochSampler] = field(default=None, init=False)
    _y_probe: Optional[np.ndarray] = field(default=None, init=False)
    _epoch: int = field(default=0, init=False)
    _lr_scales: dict = field(default_factory=dict, init=False)

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
        """Attach probe, optional transfer warm-start, and build sampler."""
        if self.use_transfer_pick and self.parent_pool:
            name, parent, scores = pick_transfer_parent_torch(
                self.parent_pool, X_probe, probe_zone=self.probe_zone
            )
            info = warm_start_from_parent(model, parent, fresh_head=True)
            self.picked_parent_name = name
            self.picked_parent_stasis = scores[0].stasis
            if self.verbose:
                print(
                    f"  [vitalroute] transfer pick: {name}  "
                    f"stasis={scores[0].stasis:.4f}  loaded={info['n_loaded']}"
                )

        self._probe = make_probe(
            model,
            self.architecture,
            probe_zone=self.probe_zone,
        )
        self._probe.observe(X_probe)

        if isinstance(y_probe, torch.Tensor):
            probe_labels = y_probe.cpu().numpy().astype(np.int64)
        else:
            probe_labels = np.asarray(y_probe, dtype=np.int64)
        self._y_probe = probe_labels

        if y_full is not None:
            if isinstance(y_full, torch.Tensor):
                sampler_labels = y_full.cpu().numpy().astype(np.int64)
            else:
                sampler_labels = np.asarray(y_full, dtype=np.int64)
        else:
            sampler_labels = probe_labels

        if self.verbose:
            print(
                f"  [vitalroute] route={self.routing_label}  "
                f"arch={self.architecture} zone={self.probe_zone}  "
                f"probe on {len(self._probe.layer_names())} layers  "
                f"(probe_n={len(probe_labels)}  full_n={len(sampler_labels)})"
            )

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

    def make_optimizer(
        self,
        model: nn.Module,
        optimizer_cls: type = torch.optim.Adam,
        lr: float = 1e-3,
        **kwargs,
    ) -> torch.optim.Optimizer:
        """Plain optimizer, or vitality-scaled param groups when lr_scale is on."""
        if self.use_lr_scale and self._probe is not None:
            return make_vitality_optimizer(model, self._probe, optimizer_cls, lr, **kwargs)
        return optimizer_cls(model.parameters(), lr=lr, **kwargs)

    def on_epoch_start(
        self,
        model: nn.Module,
        X_train: "torch.Tensor | np.ndarray",
        optimizer: "torch.optim.Optimizer",
        epoch: int,
    ) -> None:
        self._epoch = epoch
        if self._probe is None:
            return

        if epoch % max(1, self.lr_scale_refresh_every) == 0:
            self._probe.observe(X_train)

        if self.use_lr_scale:
            self._lr_scales = apply_lr_scales(
                optimizer,
                self._probe,
                alpha=self.lr_scale_alpha,
                lr_min=self.lr_scale_min,
            )
            if self.verbose and self._lr_scales:
                vec = [f"{v:.2f}" for v in self._lr_scales.values()]
                print(f"    [vitalroute] lr_scale: [{', '.join(vec)}]")

        if self._sampler is not None and self._y_probe is not None:
            refresh_every = (
                self.sampler_refresh_every
                if self.use_imbalance_sampler
                else self.hard_refresh_every
            )
            if epoch % max(1, refresh_every) == 0:
                X_t = torch.from_numpy(X_train) if isinstance(X_train, np.ndarray) else X_train
                y_t = torch.from_numpy(self._y_probe)
                self._sampler.refresh(X_t, y_t, model)

    def after_epoch(
        self,
        model: nn.Module,
        X_train: "torch.Tensor | np.ndarray",
        y_train: "torch.Tensor | np.ndarray",
    ) -> dict:
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
        if self._probe is not None:
            self._probe.detach()
            self._probe = None

    @property
    def probe(self):
        return self._probe


def torch_adaptive_controller(
    y_train: "torch.Tensor | np.ndarray",
    num_classes: int,
    *,
    parent_pool: Optional[List[Tuple[str, nn.Module]]] = None,
    architecture: Architecture = "auto",
    probe_zone: ProbeZone = "all",
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
    """Build a ``TorchTrainingController`` with tactics chosen from class counts."""
    if isinstance(y_train, torch.Tensor):
        y_np = y_train.cpu().numpy().astype(np.int64)
    else:
        y_np = np.asarray(y_train, dtype=np.int64)

    profile = profile_task(y_np, num_classes)
    plan = route_plan(
        profile,
        parent_pool_available=bool(parent_pool),
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
        use_transfer_pick=plan.use_transfer_pick,
        use_lr_scale=plan.use_lr_scale,
        lr_scale_alpha=lr_scale_alpha,
        sampler_strength=sampler_strength,
        sampler_beta=sampler_beta,
        sampler_refresh_every=sampler_refresh_every,
        architecture=architecture,
        probe_zone=probe_zone,
        parent_pool=parent_pool,
        monitor=True,
        verbose=verbose,
    )
