"""Optional reference MLP backbone for demos and vitality probes."""

from .mlp import MLP, LayerSpec, Adam, SGD

__all__ = ["MLP", "LayerSpec", "Adam", "SGD"]
