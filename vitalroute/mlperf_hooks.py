"""MLPerf Training integration hooks for VitalRoute.

Epoch callbacks aligned with MLPerf-style training loops (ResNet / image
classification). The MLPerf repository is not required.

Usage in an MLPerf-like loop::

    from vitalroute.mlperf_hooks import VitalRouteMLPerfCallback

    callback = VitalRouteMLPerfCallback(
        y_train=train_labels,
        num_classes=1000,
        architecture="cnn",
        probe_zone="head",
    )
    callback.before_train(model, train_dataset, device="cuda")
    optimizer = callback.make_optimizer(model, torch.optim.SGD, lr=0.1, momentum=0.9)

    for epoch in range(epochs):
        callback.on_epoch_begin(epoch, model, optimizer)
        train_one_epoch(model, callback.get_train_loader(train_dataset), optimizer)
        callback.on_epoch_end(epoch, model)
    callback.after_train()
"""

from __future__ import annotations

from typing import Callable, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset, Sampler
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.mlperf_hooks. "
        "Install it with: pip install torch"
    ) from exc

from .torch_controller import TorchTrainingController, torch_adaptive_controller
from .torch_data import stratified_probe_batch


class VitalRouteMLPerfCallback:
    """Epoch callback that wraps ``TorchTrainingController`` for MLPerf-style loops."""

    def __init__(
        self,
        y_train: Union[np.ndarray, "torch.Tensor"],
        num_classes: int,
        *,
        parent_pool=None,
        architecture: str = "auto",
        probe_zone: str = "head",
        per_class_probe: int = 50,
        batch_size: int = 128,
        num_workers: int = 0,
        verbose: bool = True,
        seed: int = 0,
        **controller_kwargs,
    ):
        self._y_train = y_train
        self._num_classes = num_classes
        self._per_class_probe = per_class_probe
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._seed = seed
        self._device: Optional[torch.device] = None
        self._ctrl: TorchTrainingController = torch_adaptive_controller(
            y_train,
            num_classes,
            parent_pool=parent_pool,
            architecture=architecture,  # type: ignore[arg-type]
            probe_zone=probe_zone,  # type: ignore[arg-type]
            verbose=verbose,
            **controller_kwargs,
        )
        self._sampler: Optional[Sampler] = None
        self._X_probe: Optional[torch.Tensor] = None
        self._y_probe: Optional[torch.Tensor] = None
        self._metrics: dict = {}

    @property
    def controller(self) -> TorchTrainingController:
        return self._ctrl

    @property
    def routing_label(self) -> str:
        return self._ctrl.routing_label

    @property
    def last_metrics(self) -> dict:
        return dict(self._metrics)

    def before_train(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        *,
        device: Union[str, torch.device] = "cpu",
        labels: Optional[np.ndarray] = None,
    ) -> None:
        """Attach probe and build sampler before epoch 0."""
        self._device = torch.device(device)
        y = labels if labels is not None else self._y_train
        if isinstance(y, torch.Tensor):
            y_np = y.cpu().numpy()
        else:
            y_np = np.asarray(y)

        self._X_probe, self._y_probe, _ = stratified_probe_batch(
            train_dataset,
            y_np,
            per_class=self._per_class_probe,
            num_classes=self._num_classes,
            seed=self._seed,
            device=self._device,
        )
        self._sampler = self._ctrl.setup(
            model,
            self._X_probe,
            self._y_probe,
            y_full=y_np,
            num_classes=self._num_classes,
            seed=self._seed,
        )

    def make_optimizer(
        self,
        model: nn.Module,
        optimizer_cls: type = torch.optim.SGD,
        lr: float = 0.1,
        **kwargs,
    ) -> torch.optim.Optimizer:
        return self._ctrl.make_optimizer(model, optimizer_cls, lr, **kwargs)

    def on_epoch_begin(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self._X_probe is None:
            raise RuntimeError("Call before_train() first")
        self._ctrl.on_epoch_start(model, self._X_probe, optimizer, epoch)
        if self._sampler is not None and hasattr(self._sampler, "refresh"):
            if epoch % max(1, self._ctrl.sampler_refresh_every) == 0:
                self._sampler.refresh(self._X_probe, self._y_probe, model)

    def on_epoch_end(
        self,
        epoch: int,
        model: nn.Module,
    ) -> dict:
        if self._X_probe is None or self._y_probe is None:
            return {}
        self._metrics = self._ctrl.after_epoch(model, self._X_probe, self._y_probe)
        self._metrics["epoch"] = epoch
        self._metrics["route"] = self._ctrl.routing_label
        return self._metrics

    def get_train_sampler(self) -> Optional[Sampler]:
        return self._sampler

    def get_train_loader(
        self,
        train_dataset: Dataset,
        *,
        batch_size: Optional[int] = None,
        collate_fn: Optional[Callable] = None,
    ) -> DataLoader:
        bs = batch_size if batch_size is not None else self._batch_size
        kwargs = dict(
            batch_size=bs,
            num_workers=self._num_workers,
            pin_memory=self._device is not None and self._device.type == "cuda",
        )
        if collate_fn is not None:
            kwargs["collate_fn"] = collate_fn
        if self._sampler is not None:
            return DataLoader(train_dataset, sampler=self._sampler, **kwargs)
        return DataLoader(train_dataset, shuffle=True, **kwargs)

    def after_train(self) -> None:
        self._ctrl.detach()

    def log_mlperf_tags(self) -> dict:
        """Key-value tags for MLPerf result logging / compliance."""
        return {
            "vitalroute_enabled": True,
            "vitalroute_route": self._ctrl.routing_label,
            "vitalroute_architecture": self._ctrl.architecture,
            "vitalroute_probe_zone": self._ctrl.probe_zone,
            "vitalroute_transfer_parent": self._ctrl.picked_parent_name or "",
        }
