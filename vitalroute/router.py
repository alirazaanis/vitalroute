"""Task-aware training controller: route tactics from dataset shape only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import numpy as np

from .backbone.mlp import MLP
from .hard_samples import HardSampleSampler
from .imbalance import VitalitySampler
from .lr_scale import make_vitality_scaled_optimizer, refresh_lr_scales
from .transfer import pick_transfer_parent
from .vitality import diagnose_any, layer_stasis_rates_any, resurrect_dead

EpochSampler = Union[VitalitySampler, HardSampleSampler]


def _has_separate_head(model: object) -> bool:
    return hasattr(model, "head") and hasattr(model, "flatten_features")


@dataclass
class TaskProfile:
    n_samples: int
    n_classes: int
    min_per_class: int
    max_per_class: int
    imbalance_ratio: float


@dataclass
class RoutePlan:
    use_imbalance_sampler: bool
    use_transfer_pick: bool
    use_lr_scale: bool
    use_hard_sample_sampler: bool
    label: str

    @property
    def enable_c2(self) -> bool:
        return self.use_imbalance_sampler

    @property
    def enable_b3(self) -> bool:
        return self.use_transfer_pick

    @property
    def enable_b1(self) -> bool:
        return self.use_lr_scale


def profile_task(y_train: np.ndarray, num_classes: int) -> TaskProfile:
    y_train = np.asarray(y_train)
    counts = np.bincount(y_train.astype(np.int64), minlength=num_classes)
    min_c, max_c = int(counts.min()), int(counts.max())
    return TaskProfile(
        n_samples=int(y_train.size),
        n_classes=num_classes,
        min_per_class=min_c,
        max_per_class=max_c,
        imbalance_ratio=(min_c / max_c) if max_c > 0 else 1.0,
    )


def route_plan(
    profile: TaskProfile,
    *,
    parent_pool_available: bool,
    imbalance_threshold: float = 0.25,
    min_class_for_sampler: int = 15,
    max_samples_for_transfer: int = 200,
    max_min_class_for_transfer: int = 12,
    min_samples_for_lr_scale: int = 80,
    min_samples_for_hard_sampler: int = 40,
) -> RoutePlan:
    use_sampler = (
        profile.imbalance_ratio < imbalance_threshold
        and profile.min_per_class >= min_class_for_sampler
        and profile.n_classes > 1
    )
    use_transfer = bool(parent_pool_available) and (
        profile.n_samples <= max_samples_for_transfer
        or profile.min_per_class <= max_min_class_for_transfer
    )
    if use_transfer and profile.imbalance_ratio >= 0.5 and profile.n_samples <= max_samples_for_transfer:
        use_sampler = False

    use_lr_scale = (
        not use_sampler
        and profile.n_samples >= min_samples_for_lr_scale
        and profile.n_classes > 1
    )
    use_hard = (
        not use_sampler
        and profile.n_samples >= min_samples_for_hard_sampler
        and profile.n_classes > 1
    )

    parts: List[str] = []
    if use_transfer:
        parts.append("transfer")
    if use_sampler:
        parts.append("imbalance")
    elif use_hard:
        parts.append("hard")
    if use_lr_scale:
        parts.append("lr_scale")
    if not parts:
        parts.append("monitor")
    return RoutePlan(
        use_imbalance_sampler=use_sampler,
        use_transfer_pick=use_transfer,
        use_lr_scale=use_lr_scale,
        use_hard_sample_sampler=use_hard,
        label="+".join(parts),
    )


@dataclass
class TrainingController:
    """Orchestrates vitality sampling, transfer warm-start, LR scale, and monitoring."""

    use_imbalance_sampler: bool = True
    sampler_strength: float = 0.7
    sampler_beta: float = 4.0
    sampler_refresh_every: int = 2
    sampler_stress_mode: str = "composite"

    use_hard_sample_sampler: bool = False
    hard_sample_strength: float = 0.6
    hard_sample_beta: float = 3.0
    hard_sample_refresh_every: int = 1

    use_lr_scale: bool = False
    lr_scale_alpha: float = 4.0
    lr_scale_min: float = 0.1
    lr_scale_refresh_every: int = 1

    parent_pool: Optional[List[Tuple[str, object]]] = None

    monitor: bool = True
    reset_on_high_stasis: bool = True
    stasis_reset_threshold: float = 0.12
    skip_reset_for_head_models: bool = True

    picked_parent_name: Optional[str] = field(default=None, init=False)
    picked_parent_stasis: Optional[float] = field(default=None, init=False)
    routing_label: Optional[str] = field(default=None, init=False)

    def mean_stasis(self, model: object, X: np.ndarray) -> float:
        rates = layer_stasis_rates_any(model, X)
        if len(rates) > 1:
            return float(rates[:-1].mean())
        return float(rates.mean())

    def make_optimizer(
        self,
        name: str = "adam",
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        momentum: float = 0.9,
    ) -> object:
        """Return a vitality-scaled optimizer when ``use_lr_scale``; else plain Adam/SGD."""
        if self.use_lr_scale:
            return make_vitality_scaled_optimizer(
                name,
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                alpha=self.lr_scale_alpha,
                min_scale=self.lr_scale_min,
            )
        from .backbone.mlp import Adam, SGD
        if name == "adam":
            return Adam(lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return SGD(lr=lr, momentum=momentum, weight_decay=weight_decay)
        raise ValueError(f"unknown optimizer: {name!r}")

    def on_epoch_start(
        self,
        model: object,
        X_train: np.ndarray,
        optimizer: object,
        epoch: int,
        *,
        verbose: bool = False,
    ) -> Optional[dict]:
        """Refresh per-layer LR scales at the start of an epoch (B1)."""
        scales = refresh_lr_scales(
            optimizer, model, X_train,
            epoch=epoch, refresh_every=self.lr_scale_refresh_every,
        )
        if verbose and scales:
            vec = [f"{s:.2f}" for s in scales.values()]
            print(f"    [vitalroute] lr_scale per-layer: [{', '.join(vec)}]")
        return scales

    def bootstrap(
        self,
        model: object,
        X_train: np.ndarray,
        y_train: np.ndarray,
        *,
        num_classes: int,
        seed: Optional[int] = 0,
        verbose: bool = False,
    ) -> Tuple[Optional[EpochSampler], Optional[object]]:
        parent = None
        if self.parent_pool:
            name, parent, scores = pick_transfer_parent(self.parent_pool, X_train)
            self.picked_parent_name = name
            self.picked_parent_stasis = scores[0].stasis
            if verbose:
                print(f"  [vitalroute] transfer pick: {name}  stasis={scores[0].stasis:.4f}")
            info = model.inherit_from(parent, mode="yy", fresh_head=True, verbose=False)
            if verbose:
                if _has_separate_head(model):
                    print(f"  [vitalroute] warm-start trunk + head layers "
                          f"{info.get('inherited', 0)}/{info.get('total_layers', 0)}")
                else:
                    print(f"  [vitalroute] warm-start {info['inherited']}/{info['total_layers']} layers")

        sampler: Optional[EpochSampler] = None
        if self.use_imbalance_sampler and num_classes > 1:
            sampler = VitalitySampler(
                num_classes=num_classes,
                strength=self.sampler_strength,
                beta=self.sampler_beta,
                refresh_every=self.sampler_refresh_every,
                stress_mode=self.sampler_stress_mode,  # type: ignore[arg-type]
                seed=seed,
            )
            if verbose:
                print(f"  [vitalroute] vitality sampler ({self.sampler_stress_mode})")
        elif self.use_hard_sample_sampler:
            sampler = HardSampleSampler(
                strength=self.hard_sample_strength,
                beta=self.hard_sample_beta,
                refresh_every=self.hard_sample_refresh_every,
                seed=seed,
            )
            if verbose:
                print(f"  [vitalroute] hard-sample sampler on")
        if self.use_lr_scale and verbose:
            print(f"  [vitalroute] lr_scale on (alpha={self.lr_scale_alpha})")
        if self.monitor and verbose:
            print(f"  [vitalroute] monitor on (reset if stasis>={self.stasis_reset_threshold:.2f})")
        return sampler, parent

    def after_epoch(
        self,
        model: object,
        X_train: np.ndarray,
        rng: np.random.Generator,
        *,
        verbose: bool = False,
        log: bool = True,
    ) -> Tuple[object, int]:
        report = diagnose_any(model, X_train)
        reset_count = 0
        skip_reset = self.skip_reset_for_head_models and _has_separate_head(model)
        if self.reset_on_high_stasis and not skip_reset:
            stasis = self.mean_stasis(model, X_train)
            if stasis >= self.stasis_reset_threshold:
                reset_count = resurrect_dead(model, X_train, rng=rng)
                report.resurrected = reset_count
                if verbose and log:
                    print(f"    [vitalroute] unit reset: stasis={stasis:.3f}  +{reset_count}")
        elif self.reset_on_high_stasis and skip_reset and verbose and log:
            stasis = self.mean_stasis(model, X_train)
            if stasis >= self.stasis_reset_threshold:
                print(f"    [vitalroute] monitor only (reset skipped, stasis={stasis:.3f})")
        if verbose and log and self.monitor:
            msg = f"    [vitalroute] {report.summary()}"
            if reset_count:
                msg += f"  +{reset_count} reset"
            print(msg)
        return report, reset_count


def adaptive_controller(
    y_train: np.ndarray,
    num_classes: int,
    parent_pool: Optional[List[Tuple[str, object]]] = None,
    *,
    verbose: bool = False,
    sampler_strength: float = 0.7,
    sampler_beta: float = 4.0,
    sampler_refresh_every: int = 2,
    sampler_stress_mode: str = "composite",
    stasis_reset_threshold: float = 0.12,
    lr_scale_alpha: float = 4.0,
) -> TrainingController:
    """Build a controller with tactics chosen from class counts and dataset size."""
    profile = profile_task(y_train, num_classes)
    plan = route_plan(profile, parent_pool_available=parent_pool is not None)
    if verbose:
        print(
            f"  [vitalroute] n={profile.n_samples}  "
            f"classes={profile.min_per_class}-{profile.max_per_class}  "
            f"imb={profile.imbalance_ratio:.3f}  -> {plan.label}"
        )
    ctrl = TrainingController(
        use_imbalance_sampler=plan.use_imbalance_sampler,
        sampler_strength=sampler_strength,
        sampler_beta=sampler_beta,
        sampler_refresh_every=sampler_refresh_every,
        sampler_stress_mode=sampler_stress_mode,
        use_hard_sample_sampler=plan.use_hard_sample_sampler,
        use_lr_scale=plan.use_lr_scale,
        lr_scale_alpha=lr_scale_alpha,
        parent_pool=parent_pool if plan.use_transfer_pick else None,
        monitor=True,
        reset_on_high_stasis=True,
        stasis_reset_threshold=stasis_reset_threshold,
    )
    ctrl.routing_label = plan.label
    return ctrl


# Aliases for integrators migrating from older names
ControlLoopConfig = TrainingController
adaptive_control_config = adaptive_controller
route_tactics = route_plan
AdaptiveRouting = RoutePlan
