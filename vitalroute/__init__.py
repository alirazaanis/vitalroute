"""VitalRoute — task-aware training controller on top of your optimizer."""

from .router import (
    TrainingController,
    TaskProfile,
    RoutePlan,
    profile_task,
    route_plan,
    adaptive_controller,
    ControlLoopConfig,
    adaptive_control_config,
    route_tactics,
    AdaptiveRouting,
    EpochSampler,
)
from .imbalance import VitalitySampler, StressMode
from .hard_samples import HardSampleSampler
from .lr_scale import (
    VitalityScaledAdam,
    VitalityScaledSGD,
    make_vitality_scaled_optimizer,
    refresh_lr_scales,
)
from .transfer import pick_transfer_parent, score_transfer_candidates, TransferScore
from .vitality import (
    diagnose_any,
    resurrect_dead,
    HealthReport,
    per_class_health,
    per_class_stress,
    per_sample_difficulty,
    per_sample_stress,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # NumPy / backbone API
    "TrainingController",
    "TaskProfile",
    "RoutePlan",
    "profile_task",
    "route_plan",
    "adaptive_controller",
    "EpochSampler",
    "VitalitySampler",
    "StressMode",
    "HardSampleSampler",
    "VitalityScaledAdam",
    "VitalityScaledSGD",
    "make_vitality_scaled_optimizer",
    "refresh_lr_scales",
    "pick_transfer_parent",
    "score_transfer_candidates",
    "TransferScore",
    "diagnose_any",
    "resurrect_dead",
    "HealthReport",
    "per_class_health",
    "per_class_stress",
    "per_sample_difficulty",
    "per_sample_stress",
    "ControlLoopConfig",
    "adaptive_control_config",
    "route_tactics",
    "AdaptiveRouting",
    # PyTorch API (imported lazily to avoid hard torch dependency)
    # from vitalroute.torch_probe      import VitalityProbe
    # from vitalroute.torch_samplers   import TorchVitalitySampler, TorchHardSampleSampler
    # from vitalroute.torch_controller import TorchTrainingController, torch_adaptive_controller
]
