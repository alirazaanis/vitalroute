# Changelog

All notable changes to VitalRoute are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).  
VitalRoute uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] — 2026-06-03 — CNN, MLPerf, and public contributions

### Added

**CNN modules**
- `torch_data.py` — `stratified_probe_indices`, `stratified_probe_batch`
- `torch_probe_cnn.py` — `CNNVitalityProbe` with BatchNorm-aware conv probing and `probe_zone=head|trunk|all`
- `torch_probes.py` — `make_probe`, `detect_architecture`
- `torch_transfer.py` — `pick_transfer_parent_torch`, `warm_start_from_parent`
- `torch_lr_scale.py` — `build_layer_param_groups`, `make_vitality_optimizer`, `apply_lr_scales`
- `TorchTrainingController` — `architecture`, `probe_zone`, `parent_pool`, `make_optimizer()`

**MLPerf integration**
- `mlperf_hooks.py` — `VitalRouteMLPerfCallback` with `log_mlperf_tags()`

**Examples**
- `examples/cifar10_lt_benchmark.py` — full LT + transfer-pick demo
- `examples/imagenet_lt_benchmark.py` — CIFAR-100-LT local / ImageNet-LT
- `examples/mlperf_resnet_integration.py` — MLPerf callback demo
- `examples/run_cnn_benchmarks.py` — unified benchmark runner
- `examples/benchmark_cnn_common.py` — shared CNN benchmark utilities

**Tests**
- `tests/test_torch_cnn.py` — 8 tests for CNN probe, transfer, LR scale, MLPerf callback

### Fixed
- `cifar10_resnet_benchmark.py` — stratified probe batch and `y_full=targets_np` (was incorrectly `y_full=y_seed`)

### Documentation
- `README.md` — CNN benchmark results, mechanism table, package layout
- `INTEGRATION.md` — CNN, MLPerf, and benchmark result sections

### Repository
- `.github/workflows/ci.yml` — pytest on Python 3.10–3.12 for push and PR
- `.github/workflows/publish-pypi.yml` — publish to PyPI on GitHub release
- `.github/workflows/codeql.yml` — Python CodeQL analysis
- `.github/dependabot.yml` — weekly pip and GitHub Actions updates
- `.github/PULL_REQUEST_TEMPLATE.md` — PR checklist
- `.github/ISSUE_TEMPLATE/` — bug report and feature request forms
- `CODE_OF_CONDUCT.md`, `SECURITY.md` — community and security policies
- Expanded `CONTRIBUTING.md` for public contributions

---

## [0.1.1] — 2026-05-30 — Metadata fix

### Fixed
- Corrected repository URLs in `pyproject.toml` and `CITATION.cff` to `github.com/alirazaanis/vitalroute`

---

## [0.1.0] — 2026-05-30 — Initial public release

### Added

**Core vitality engine (`vitalroute/vitality.py`)**
- Four per-unit stress signals: stasis, weak weights, weak input, saturation
- Composite stress aggregation with configurable per-signal weights
- `per_class_stress` and `per_sample_stress` for class- and instance-level difficulty
- `resurrect_dead` — re-initialises dead units and clears optimizer momentum

**Adaptive router (`vitalroute/router.py`)**
- `profile_task` — derives imbalance ratio and sample counts from label arrays
- `route_plan` — maps task profile to a tactic set (sampler / transfer / LR scale / monitor)
- `TrainingController` and `adaptive_controller` factory for the NumPy backbone
- Configurable routing thresholds via `ControlLoopConfig`

**Samplers (`vitalroute/imbalance.py`, `vitalroute/hard_samples.py`)**
- `VitalitySampler` — composite-stress-weighted class oversampler (NumPy)
- `HardSampleSampler` — per-sample stress × confidence curriculum sampler (NumPy)

**LR scaling (`vitalroute/lr_scale.py`)**
- `VitalityScaledAdam` and `VitalityScaledSGD` — optimizers whose per-layer LR is dampened by stasis rate
- `refresh_lr_scales` — standalone utility for arbitrary optimizers

**Label-free transfer selection (`vitalroute/transfer.py`)**
- `score_transfer_candidates` — ranks pretrained parents by lowest stasis on new inputs (no labels)
- `pick_transfer_parent` — returns the best parent and warm-starts weights

**PyTorch integration (no changes to model or optimizer required)**
- `VitalityProbe` (`vitalroute/torch_probe.py`) — attaches four stress signals to any `nn.Module` via forward hooks; auto-pairs `Linear`/`Conv2d` with subsequent activation layers
- `TorchVitalitySampler` and `TorchHardSampleSampler` (`vitalroute/torch_samplers.py`) — drop-in `torch.utils.data.Sampler` replacements driven by probe stress scores
- `TorchTrainingController` and `torch_adaptive_controller` (`vitalroute/torch_controller.py`) — full adaptive controller mirroring NumPy API; `setup()` takes separate probe data and full training labels

**Examples**
- `examples/digits_imbalanced_demo.py` — NumPy backbone quick demo
- `examples/benchmark_baselines.py` — NumPy vs uniform / inv-freq baseline comparison
- `examples/torch_probe_demo.py` — standalone `VitalityProbe` usage
- `examples/torch_benchmark_fmnist.py` — PyTorch Fashion-MNIST long-tail benchmark
- `examples/cifar10_resnet_benchmark.py` — ResNet18 / CIFAR-10 (GPU recommended)

**Documentation**
- `README.md` — overview, quick start, router rules, evidence summary, PyTorch integration guide
- `INTEGRATION.md` — step-by-step integration for NumPy backbone and PyTorch training loops
- `PAPER.md` — research paper style writeup with formal definitions and benchmark results
- `CITATION.cff` — machine-readable citation metadata

**Tests (`tests/`)**
- 14 unit tests covering `VitalityProbe`, `TorchVitalitySampler`, `TorchHardSampleSampler`, `TorchTrainingController`

### Package metadata
- Python ≥ 3.10, NumPy ≥ 1.24 (core); PyTorch ≥ 2.0 optional extra (`pip install vitalroute[torch]`)
- MIT license
