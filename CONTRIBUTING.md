# Contributing to VitalRoute

Thank you for considering a contribution. VitalRoute is a small research library and any improvement — bug fix, new benchmark, documentation edit, or new tactic — is welcome.

## Quick start

```bash
git clone https://github.com/alirazaanis/vitalroute.git
cd vitalroute
pip install -e ".[dev]"   # installs numpy, torch, torchvision, scikit-learn
pytest tests/             # all tests should pass
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
- Follow existing code style (no external formatter required, keep it readable)
- No type stubs required, but type hints on public functions are appreciated
- Do not add dependencies beyond NumPy to the core package; PyTorch extensions go in `torch_*.py` files only

### Tests
- Add or update a test in `tests/` for any changed or new public function
- Run `pytest tests/` before submitting; all tests must pass
- For PyTorch tests, use small synthetic tensors — keep test runtime under 10 seconds total

### Commits and PRs
- One logical change per PR
- Write a one-line summary of what the PR does in the PR title
- Reference any related issue in the PR description
- Do not bump version numbers in a PR — maintainers handle releases

### New tactics
If you are adding a new tactic (new file in `vitalroute/`):
1. Add it to `vitalroute/__init__.py` in the relevant section
2. Add at least one unit test
3. Document it in `INTEGRATION.md` with a minimal usage example
4. Add a one-line description to the package layout table in `README.md`

## Reporting bugs

Open a GitHub issue with:
- Python version and OS
- `pip show vitalroute torch numpy` output
- Minimal reproducible example
- Full traceback

## License

By contributing, you agree your contributions are licensed under the MIT License.
