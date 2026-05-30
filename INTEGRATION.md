# Integrating VitalRoute with your trainer

VitalRoute does **not** own your backward pass. It provides hooks you call
at specific points in your loop. Two integration paths:

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

Your classifier should support:

| Method / attribute | Purpose |
|---|---|
| `._layers` | List of layers with `.W`, `.b`, `.activation` |
| `.inherit_from(parent, mode="yy", fresh_head=True)` | Transfer warm-start |
| `.forward(X, train=True/False)` | Forward pass |
| `.train_step(X, y, optimizer)` or equivalent | Your training |

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

Use `VitalitySampler(..., stress_mode="stasis")` for legacy stasis-only class weighting.

### Parent pool format

```python
parent_pool = [
    ("parent_a", model_a),  # pretrained on related task
    ("parent_b", model_b),
]
```

Score with `score_transfer_candidates(parent_pool, X_new)` without training.

### When *not* to enable reset

Set `skip_reset_for_head_models=True` (default) when stasis is computed on
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

probe.detach()                 # clean up
```

### Key design note: probe data vs sampler data

`setup()` takes two separate data inputs:

| Parameter | Purpose | Typical size |
|---|---|---|
| `X_probe, y_probe` | Run the vitality probe — must be **stratified** (all classes present) | ~50 per class |
| `y_full` | Build sampler class pools — must cover **all training examples** | full dataset |

Passing an unrepresentative `y_probe` (e.g. only one class) will cause the
sampler to over-weight that class and break learning. Always use a stratified
probe batch.
