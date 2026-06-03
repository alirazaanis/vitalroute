# Integrating VitalRoute with a training loop

Install from PyPI: `pip install vitalroute` ([package page](https://pypi.org/project/vitalroute/)).
PyTorch support: `pip install "vitalroute[torch]"`.

VitalRoute does **not** own the backward pass. It provides epoch hooks invoked
at fixed points in the loop. Two integration paths:

- **NumPy backbone** — `TrainingController` / `adaptive_controller`
- **PyTorch (any model)** — `TorchTrainingController` / `torch_adaptive_controller`

---

## NumPy backbone

### What the controller provides

- `TrainingController.bootstrap()` — once before epoch 0
- `TrainingController.on_epoch_start()` — refresh per-layer LR scales (when `lr_scale` route is on)
- `TrainingController.after_epoch()` — after each epoch
- `TrainingController.make_optimizer()` — plain or vitality-scaled Adam/SGD
- Optional `VitalitySampler` or `HardSampleSampler` — `sample_indices()` each epoch

### Required model interface

The classifier should support:

| Method / attribute | Purpose |
|---|---|
| `._layers` | List of layers with `.W`, `.b`, `.activation` |
| `.inherit_from(parent, mode="yy", fresh_head=True)` | Transfer warm-start |
| `.forward(X, train=True/False)` | Forward pass |
| `.train_step(X, y, optimizer)` or equivalent | Training step |

Optional for CNN-style models:

| `.head` | MLP head after conv trunk |
| `.flatten_features(X)` | Map raw input to head input |

### Minimal loop

```python
ctrl = adaptive_controller(y_train, num_classes, parent_pool=my_parents)
opt = ctrl.make_optimizer("adam", lr=2e-3)
sampler, parent = ctrl.bootstrap(model, X_train, y_train, num_classes=C)

for epoch in range(E):
    ctrl.on_epoch_start(model, X_train, opt, epoch)
    if sampler:
        idx = sampler.sample_indices(epoch, model, X_train, y_train, n)
        train_on_shuffled_batches(X_train[idx], y_train[idx])
    else:
        train_on_shuffled_batches(X_train, y_train)
    ctrl.after_epoch(model, X_train, rng)
```

### Composite stress APIs

```python
from vitalroute import per_class_stress, per_sample_stress

class_scores  = per_class_stress(model, X_train, y_train, num_classes=10)
sample_scores = per_sample_stress(model, X_train, y_train)
```

Legacy stasis-only class weighting: `VitalitySampler(..., stress_mode="stasis")`.

### Parent pool format

```python
parent_pool = [
    ("parent_a", model_a),  # pretrained on related task
    ("parent_b", model_b),
]
```

Scored via `score_transfer_candidates(parent_pool, X_new)` without training.

### When reset is disabled

With `skip_reset_for_head_models=True` (default), reset is skipped when stasis is computed on
flattened conv features — mass reset of the MLP head often hurts. The controller
still logs vitality.

---

## PyTorch (any nn.Module)

### What the controller provides

- `TorchTrainingController.setup()` — attach probe + build sampler, once before training
- `TorchTrainingController.on_epoch_start()` — refresh probe + sampler weights
- `TorchTrainingController.after_epoch()` — log vitality
- `TorchTrainingController.detach()` — remove hooks after training

### Minimal loop

```python
from vitalroute.torch_controller import torch_adaptive_controller
from torch.utils.data import DataLoader

ctrl = torch_adaptive_controller(y_train, num_classes=10, verbose=True)

# X_probe / y_probe: stratified sample (~50/class) for the probe
# y_full: full training labels so the sampler covers all examples
sampler = ctrl.setup(model, X_probe, y_probe,
                     y_full=y_train_full, num_classes=10)

loader = DataLoader(dataset, sampler=sampler, batch_size=64)

for epoch in range(epochs):
    ctrl.on_epoch_start(model, X_probe, optimizer, epoch)
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        loss_fn(model(X_batch), y_batch).backward()
        optimizer.step()
    ctrl.after_epoch(model, X_probe, y_probe)

ctrl.detach()
```

### Probe only (no controller)

```python
from vitalroute.torch_probe import VitalityProbe

probe = VitalityProbe(model)   # hooks Linear→ReLU pairs automatically

probe.observe(X_batch)         # one forward pass, no gradients
print(probe.summary())         # stasis + composite stress per layer
print(probe.mean_stasis())     # scalar health indicator

probe.detach()                 # remove hooks
```

### Key design note: probe data vs sampler data

`setup()` takes two separate data inputs:

| Parameter | Purpose | Typical size |
|---|---|---|
| `X_probe, y_probe` | Vitality probe input — must be **stratified** (all classes present) | ~50 per class |
| `y_full` | Build sampler class pools — must cover **all training examples** | full dataset |

Passing an unrepresentative `y_probe` (e.g. only one class) causes the
sampler to over-weight that class and break learning. The probe batch must be
stratified.

Use `stratified_probe_batch()` to build probe tensors safely:

```python
from vitalroute.torch_data import stratified_probe_batch

X_probe, y_probe, idx = stratified_probe_batch(
    train_dataset, y_train, per_class=50, num_classes=10, device="cuda"
)
```

---

## CNN (ResNet / ConvNet)

### CNNVitalityProbe

`CNNVitalityProbe` extends vitality probing to BatchNorm–conv stacks. Three probe zones are defined:

| Zone | Tracks |
|---|---|
| `"head"` | Linear classifier only (default for ResNet) |
| `"trunk"` | Conv → BN → ReLU blocks |
| `"all"` | Both trunk and head |

```python
from vitalroute.torch_probes import make_probe

probe = make_probe(model, architecture="cnn", probe_zone="all")
probe.observe(X_batch)
print(probe.summary())
probe.detach()
```

### PyTorch transfer pick

```python
from vitalroute.torch_transfer import pick_transfer_parent_torch, warm_start_from_parent

name, parent, scores = pick_transfer_parent_torch(parent_pool, X_unlabeled, probe_zone="head")
warm_start_from_parent(model, parent, fresh_head=True)
```

Transfer pick also runs inside `setup()` when `parent_pool` is passed to `torch_adaptive_controller()`.

### Per-layer LR scale

When the router enables `lr_scale`, the optimizer is built via `ctrl.make_optimizer()`:

```python
ctrl = torch_adaptive_controller(y_train, num_classes, architecture="cnn", probe_zone="head")
ctrl.setup(model, X_probe, y_probe, y_full=y_train, num_classes=10)
optimizer = ctrl.make_optimizer(model, torch.optim.SGD, lr=0.1, momentum=0.9, weight_decay=1e-4)
```

### Benchmarks

| Script | Description |
|---|---|
| `examples/cifar10_resnet_benchmark.py` | CIFAR-10-LT vs baselines |
| `examples/cifar10_lt_benchmark.py` | Full + scarce transfer-pick demo |
| `examples/imagenet_lt_benchmark.py` | CIFAR-100-LT local / ImageNet-LT with `--data-dir` |
| `examples/run_cnn_benchmarks.py` | Run all benchmarks (`--quick` or `--full`) |

---

## MLPerf Training integration

`VitalRouteMLPerfCallback` provides epoch hooks that match the structure of MLPerf
reference training loops. The MLPerf repository is not required.

```python
from vitalroute.mlperf_hooks import VitalRouteMLPerfCallback

callback = VitalRouteMLPerfCallback(y_train, num_classes=1000, architecture="cnn", probe_zone="head")
callback.before_train(model, train_dataset, device="cuda")
optimizer = callback.make_optimizer(model, torch.optim.SGD, lr=0.1, momentum=0.9)

for epoch in range(epochs):
    callback.on_epoch_begin(epoch, model, optimizer)
    train_one_epoch(model, callback.get_train_loader(train_dataset), optimizer)
    metrics = callback.on_epoch_end(epoch, model)

tags = callback.log_mlperf_tags()   # for MLPerf result metadata
callback.after_train()
```

Example: `examples/mlperf_resnet_integration.py`.

In an MLPerf Training ResNet reference, the callback is inserted at the same points
as `before_run`, `pre_epoch`, and `post_epoch`. The training `DataLoader` uses
`callback.get_train_sampler()` when a sampler is active.

---

## Benchmark results (CNN)

Reproduction command: `python examples/run_cnn_benchmarks.py --full`.

### CIFAR-10 ResNet18 (10:1 long-tail, 2 epochs, 1 seed)

```
Method         Overall    Minority
uniform        26.6%       0.0%
inv_freq       38.7%      29.4%
focal          27.4%       0.0%
vitalroute     36.9%      33.5%
```

### Comparison summary

| Comparison | Result (smoke run) |
|---|---|
| vitalroute vs uniform | +10.3% overall; minority 33.5% vs 0% at epoch 2 |
| vitalroute vs inv_freq | +4.1% minority at epoch 2 |
| vitalroute vs focal | +9.5% overall; minority 33.5% vs 0% at epoch 2 |
| Transfer pick | Parent ranked by stasis; see `cifar10_lt_benchmark.py --mode transfer` |
| Seed variance | Lowest among tested methods on digits and Fashion-MNIST (see README) |

Multi-seed 15-epoch runs are defined in `run_cnn_benchmarks.py --full`. On those
tasks, vitalroute tracks inv_freq on overall accuracy; see Fashion-MNIST and digits
tables in README.md.
