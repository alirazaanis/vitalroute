# Contributing to VitalRoute

VitalRoute is an open-source project (MIT license). Contributions include bug
fixes, tests, documentation, benchmarks, and new training tactics.

## Before you start

1. Search [existing issues](https://github.com/alirazaanis/vitalroute/issues) for duplicates.
2. For large features, open an issue first to align on design.
3. Read [INTEGRATION.md](INTEGRATION.md) for training-loop integration patterns.

## Development setup

```bash
git clone https://github.com/alirazaanis/vitalroute.git
cd vitalroute
pip install -e ".[dev]"   # numpy, torch, torchvision, scikit-learn, pytest
pytest tests/             # must pass before opening a PR
```

Optional CNN benchmarks (download datasets locally; not committed):

```bash
python examples/run_cnn_benchmarks.py --quick
```

## What to contribute

| Area | Examples |
|---|---|
| Bug fixes | Stress calculation, sampler edge cases, hook cleanup |
| Tests | NumPy backbone, PyTorch probe, CNN, edge cases |
| Documentation | README, INTEGRATION, PAPER clarifications |
| Benchmarks | CIFAR-LT, ImageNet-LT, tabular imbalanced data |
| PyTorch | Probes, controllers, LR scale, transfer pick |
| CNN / vision | BatchNorm stasis, probe zones, ResNet benchmarks |
| MLPerf | Callback hooks, reference-loop integration |
| New tactics | Mixup-driven stress, transformer / LLM probes (see issues) |

## Pull request process

1. Fork the repository and create a branch from `master`.
2. Make a focused change (one logical topic per PR).
3. Add or update tests for changed public behavior.
4. Update docs when APIs, examples, or behavior change.
5. Confirm `pytest tests/` passes.
6. Open a PR using the template. Link related issues (`Fixes #123`).

### PR requirements

| Requirement | Detail |
|---|---|
| Tests | All tests pass; new public APIs have coverage |
| Scope | One logical change; split large work across PRs |
| Docs | Update README / INTEGRATION / CHANGELOG as needed |
| Version | Maintainers bump version; do not change `pyproject.toml` version in PRs |
| Secrets | No API keys, credentials, or personal dataset paths |
| Data | Do not commit `.data/`, checkpoints, or downloaded datasets |

### Review

Maintainers review for correctness, test coverage, API consistency, and
documentation. Feedback may request changes before merge.

## Code guidelines

### Style

- Match existing code in the file you edit.
- Type hints on public functions are appreciated.
- Core package depends on NumPy only; PyTorch code lives in `torch_*.py` and `mlperf_hooks.py`.
- Comments explain non-obvious logic; avoid restating the code.

### Architecture rules

- VitalRoute does not own the backward pass; it provides epoch hooks and samplers.
- Probe data (`X_probe`, `y_probe`) must be stratified across classes.
- Sampler labels (`y_full`) must cover the full training set.
- CNN models use `architecture="cnn"` and `CNNVitalityProbe` via `make_probe()`.

### New tactics

A new tactic (new module under `vitalroute/`) requires:

1. Export from `vitalroute/__init__.py` (NumPy) or documented import path (PyTorch).
2. At least one unit test in `tests/`.
3. A minimal example in `INTEGRATION.md`.
4. A one-line entry in the package layout table in `README.md`.
5. An entry in `CHANGELOG.md` under `[Unreleased]`.

## Reporting bugs

Include:

- Python version and OS
- Output of `pip show vitalroute torch numpy`
- Minimal reproducible example
- Full traceback

Use the [bug report issue template](https://github.com/alirazaanis/vitalroute/issues/new?template=bug_report.yml).

For security vulnerabilities, see [SECURITY.md](SECURITY.md). Do not file public issues for security bugs.

## Feature requests

Use the [feature request template](https://github.com/alirazaanis/vitalroute/issues/new?template=feature_request.yml).
State the problem, proposed behavior, and affected area (router, probe, CNN, etc.).

## Code of conduct

This project follows the [Code of Conduct](CODE_OF_CONDUCT.md). Participants are
expected to uphold it in issues, pull requests, and discussions.

## License

By contributing, you agree that your contributions are licensed under the
[MIT License](LICENSE).
