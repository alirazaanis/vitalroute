"""Probe registry: selects ``VitalityProbe`` or ``CNNVitalityProbe`` by architecture."""

from __future__ import annotations

from typing import Literal, Optional, Union

try:
    import torch.nn as nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "PyTorch is required for vitalroute.torch_probes. "
        "Install it with: pip install torch"
    ) from exc

from .torch_probe import VitalityProbe
from .torch_probe_cnn import CNNVitalityProbe, ProbeZone

Architecture = Literal["auto", "mlp", "cnn"]

__all__ = ["Architecture", "ProbeZone", "detect_architecture", "make_probe", "CNNVitalityProbe", "VitalityProbe"]


def detect_architecture(model: nn.Module) -> Architecture:
    """Returns ``cnn`` if the model contains ``Conv2d`` layers, else ``mlp``."""
    has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())
    return "cnn" if has_conv else "mlp"


def make_probe(
    model: nn.Module,
    architecture: Architecture = "auto",
    *,
    probe_zone: ProbeZone = "all",
    layer_names: Optional[list] = None,
    dead_threshold: float = 0.01,
    sample_size: int = 1024,
) -> Union[VitalityProbe, CNNVitalityProbe]:
    """Returns ``CNNVitalityProbe`` for ``cnn``, else ``VitalityProbe``."""
    arch = detect_architecture(model) if architecture == "auto" else architecture
    if arch == "cnn":
        return CNNVitalityProbe(
            model,
            layer_names,
            probe_zone=probe_zone,
            dead_threshold=dead_threshold,
            sample_size=sample_size,
        )
    return VitalityProbe(
        model,
        layer_names,
        dead_threshold=dead_threshold,
        sample_size=sample_size,
    )
