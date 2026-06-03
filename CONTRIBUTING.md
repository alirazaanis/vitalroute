# Contributing to VitalRoute

Contributions welcome — bug fixes, benchmarks, documentation, and new tactics.

## Quick start

```bash
git clone https://github.com/alirazaanis/vitalroute.git
cd vitalroute
pip install -e ".[dev]"   # installs numpy, torch, torchvision, scikit-learn
pytest tests/             # all tests must pass
```

## What to contribute

| Area | Examples |
|---|---|
| Bug fixes | Incorrect stress calculation, sampler edge cases, hook cleanup |
| New benchmarks | CIFAR-LT, ImageNet-LT, tabular imbalanced datasets |
| New tactics | Mixup-driven stress, per-head stress for transformers |
| PyTorch improvements | BatchNorm-aware stasis, Conv2d composite stress |
| Documentation | Clarifications in README, INTEGRATION, PAPER |
| Tests | Coverage for NumPy backbone, edge cases |

## Guidelines

### Code style
- Match existing code style (no external formatter required)
- Type hints on public functions are appreciated; type stubs are not required
- Core package dependencies are limited to NumPy; PyTorch extensions belong in `torch_*.py` files only

### Tests
- Changed or new public functions require a test in `tests/`
- All tests must pass (`pytest tests/`) before submission
- PyTorch tests use small synthetic tensors; total runtime under 10 seconds

### Commits and PRs
- One logical change per PR
- PR title: one-line summary of the change
- Related issues referenced in the PR description
- Version bumps are handled by maintainers, not in PRs

### New tactics
New tactics (new file in `vitalroute/`) require:
1. Export from `vitalroute/__init__.py` in the relevant section
2. At least one unit test
3. Documentation in `INTEGRATION.md` with a minimal usage example
4. A one-line entry in the package layout table in `README.md`

## Reporting bugs

Bug reports should include:
- Python version and OS
- `pip show vitalroute torch numpy` output
- Minimal reproducible example
- Full traceback

## License

Contributions are licensed under the MIT License.
