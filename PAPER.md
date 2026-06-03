# VitalRoute: Vitality-Driven Adaptive Training for Imbalanced and Scarce Classification Tasks

---

## Abstract

**VitalRoute** is a task-aware training controller that monitors the internal health of feed-forward neural networks during training and uses those observations to automatically select and apply remediation tactics. The core contribution is a set of four per-unit *vitality signals* — stasis, weak weights, weak input, and saturation — aggregated into a *composite stress* score that drives class-aware oversampling, label-free transfer parent selection, and per-layer learning rate dampening. An *adaptive router* reads only the shape of the training set (class counts and total size) to decide which tactics to activate, requiring no per-dataset configuration. Evaluation on long-tail digit and Fashion-MNIST classification shows consistent gains over uniform sampling, with performance matching inverse-frequency weighting at notably lower variance. A PyTorch integration layer (`VitalityProbe`) extends the system to any `nn.Module` via forward hooks without modifying the model, optimizer, or training loop.

---

## 1. Introduction

Training neural networks on imbalanced or scarce datasets remains a practical challenge. The dominant remedies — inverse-frequency class weighting, focal loss, and simple oversampling — treat the problem from the data distribution perspective and are agnostic to what is happening inside the network.

A complementary view treats class difficulty as reflected in the *activation patterns* of hidden units. A class that consistently causes neurons to fall silent (dead ReLUs), collapse their weights, or saturate is a class the network fails to represent. Oversampling based on this internal signal — rather than purely on frequency — is the central hypothesis of VitalRoute.

This report describes the design, implementation, and empirical evaluation of VitalRoute v0.2.0.

---

## 2. Vitality Signals

For each hidden layer \( l \) with activation matrix \( A^{(l)} \in \mathbb{R}^{N \times d_l} \) computed on a batch of \( N \) inputs, four scalar stress signals are defined over units \( j = 1, \ldots, d_l \):

### 2.1 Stasis (dead-unit rate)

A unit \( j \) is in stasis if it barely activates across the batch:

$$\text{stasis}_j = \mathbf{1}\!\left[\bar{f}_j < \tau_s \;\lor\; \text{Var}(A^{(l)}_{:,j}) < \tau_v \right]$$

where \( \bar{f}_j = \frac{1}{N}\sum_i \mathbf{1}[A^{(l)}_{ij} > 0] \) is the empirical firing rate for ReLU activations (adapted for tanh/sigmoid), \( \tau_s = 0.01 \), and \( \tau_v = 10^{-6} \). The layer stasis rate is \( s_l = \frac{1}{d_l}\sum_j \text{stasis}_j \).

### 2.2 Weak Weights

A unit \( j \) has weak weights if its incoming weight column has collapsed:

$$\text{weak-weights}_j = \mathbf{1}\!\left[\|W^{(l)}_{:,j}\|_2 < \tau_w\right], \quad \tau_w = 0.05$$

### 2.3 Weak Input

A unit \( j \) has a weak input signal if most of its incoming activations are smaller in magnitude than the corresponding weights — the network is poorly matched to the data:

$$\text{weak-input}_j = \mathbf{1}\!\left[\frac{1}{d_{l-1}}\sum_k \mathbf{1}[\bar{x}_k < |W^{(l)}_{kj}|] > \phi\right], \quad \phi = 0.8$$

where \( \bar{x}_k = \frac{1}{N}\sum_i |A^{(l-1)}_{ik}| \) is the mean activation magnitude of incoming unit \( k \).

### 2.4 Saturation

A unit is saturating if it outputs near a constant non-zero value:

$$\text{saturation}_j = \mathbf{1}\!\left[\text{Var}(A^{(l)}_{:,j}) < \tau_v \;\land\; |\mu_j| > 10^{-3}\right]$$

where \( \mu_j = \frac{1}{N}\sum_i A^{(l)}_{ij} \).

### 2.5 Composite Stress

The four signals are combined into a single composite stress rate per layer:

$$\sigma_l = \frac{w_s \cdot s_l + w_w \cdot w_l + w_i \cdot i_l + w_t \cdot t_l}{w_s + w_w + w_i + w_t}$$

with default weights \( (w_s, w_w, w_i, w_t) = (1.0, 0.6, 0.6, 0.5) \), giving stasis the most influence.

For a model with \( L \) hidden layers, the **per-class composite stress** is computed by running only the samples of class \( c \) through the model and averaging \( \sigma_l \) over hidden layers:

$$\Sigma_c = \frac{1}{L}\sum_{l=1}^{L} \sigma_l\!\left(X_c\right)$$

---

## 3. Training Tactics

### 3.1 Vitality Class Sampler

For imbalanced datasets, uniform class sampling is replaced with a distribution proportional to composite stress. Given class stress scores \( \Sigma = (\Sigma_1, \ldots, \Sigma_C) \), the sampling probability for class \( c \) is:

$$p_c = (1 - \alpha) \cdot \frac{1}{C} + \alpha \cdot \frac{\exp(\beta \Sigma_c)}{\sum_{c'} \exp(\beta \Sigma_{c'})}$$

where \( \alpha \in [0,1] \) controls the blend between uniform and stress-weighted sampling (default \( \alpha = 0.7 \)) and \( \beta = 4 \) is a temperature parameter. Scores are refreshed every 2 epochs.

This is activated when the imbalance ratio \( \min_c n_c \,/\, \max_c n_c < 0.25 \) and the minority class has at least 15 samples.

### 3.2 Label-Free Transfer Parent Selection

For scarce tasks (total \( n \leq 200 \) or minority \( \leq 12 \) samples per class), VitalRoute can select the best pretrained model from a pool without requiring target labels. Each candidate parent \( M_k \) is scored by its mean stasis on the new dataset:

$$\text{score}(M_k) = 1 - \bar{s}(M_k, X_{\text{new}})$$

where \( \bar{s} \) averages stasis rates over hidden layers. The parent with the highest score (lowest stasis) is selected and used to warm-start the child model via weight inheritance.

The intuition: a model whose neurons remain active on the new data is extracting useful features from it, making it a better transfer source than a model that goes silent.

### 3.3 Hard-Sample Curriculum Sampler

For scarce balanced datasets, individual examples with high per-sample stress are oversampled. The per-sample score combines layer-level stasis (averaged over hidden layers for this specific input) with the model's confidence gap:

$$\psi_i = 0.5 \cdot \bar{s}_i + 0.5 \cdot (1 - \hat{p}_{y_i})$$

where \( \hat{p}_{y_i} \) is the model's predicted probability for the true class. Sampling probabilities follow the same softmax-blend formula as the class sampler.

### 3.4 Per-Layer Learning Rate Scaling

When the sampler is not active, VitalRoute optionally dampens the learning rate of stasis-heavy layers:

$$\eta_l = \frac{\eta_0}{1 + \alpha \cdot s_l}, \quad \alpha = 4, \quad \eta_{\min} = 0.1 \cdot \eta_0$$

This slows updates in layers that are already struggling, preventing them from being pushed further into dead regions.

### 3.5 Conditional Unit Resurrection

After each epoch, if the mean stasis across hidden layers exceeds a threshold \( \tau_r = 0.12 \), dead units are re-initialized with He/Glorot weights. Crucially, the corresponding optimizer momentum buffers are also zeroed, preventing stale gradients from immediately undoing the reset.

---

## 4. Adaptive Router

Rather than exposing individual tactic flags, VitalRoute exposes a single `adaptive_controller` entry point that selects tactics from three statistics computed on the training labels:

| Statistic | Computed as |
|---|---|
| \( n \) | total training samples |
| \( r \) | \( \min_c n_c \,/\, \max_c n_c \) (imbalance ratio) |
| \( m \) | \( \min_c n_c \) (smallest class count) |

The routing rules (defaults, all overridable):

| Condition | Tactic enabled |
|---|---|
| \( r < 0.25 \) and \( m \geq 15 \) | Vitality class sampler |
| \( n \leq 200 \) or \( m \leq 12 \), and parent pool provided | Transfer pick |
| Scarce balanced data | Hard-sample sampler (no class sampler) |
| \( n \geq 80 \), sampler off | LR scale |
| Always | Monitor + conditional reset |

---

## 5. PyTorch Integration

The vitality computation was originally developed for a custom NumPy MLP. VitalRoute introduces `VitalityProbe`, which attaches the same four stress signals to any `torch.nn.Module` via `register_forward_hook`. The probe automatically pairs `Linear` / `Conv2d` modules with the activation function that immediately follows them in the module list, ensuring that stasis is measured on post-activation values (where dead neurons actually manifest).

The PyTorch `TorchTrainingController` mirrors the NumPy `TrainingController` API. A key design decision is the separation of *probe data* (a small stratified batch used to compute vitality signals, ~50 samples per class) from *sampler labels* (the full training label vector used to build class pools). Conflating these leads to biased stress estimates and degenerate sampling.

---

## 6. Experiments

### 6.1 Setup

All experiments use the public scikit-learn digits dataset (NumPy backbone) and Fashion-MNIST (PyTorch). Imbalance is introduced by downsampling majority classes while keeping minority classes intact.

**Baselines:**
- **Uniform** — standard training, no modification
- **Inv-freq** — `WeightedRandomSampler` with inverse class frequency weights
- **Focal** — focal loss with \( \gamma = 2 \), uniform sampling
- **Stasis-only** — VitalRoute vitality sampler using only the stasis signal (no composite)

### 6.2 Results

**NumPy backbone — digits, 5:1 imbalance, 3 seeds, 30 epochs:**

| Method | Overall | Minority acc |
|---|---|---|
| Uniform | 93.7% ± 1.1% | 87.9% ± 2.3% |
| Inv-freq | 94.4% ± 0.8% | 90.1% ± 1.0% |
| Stasis-only | 95.0% ± 0.7% | 90.7% ± 1.5% |
| **VitalRoute** | **95.1% ± 0.3%** | **90.8% ± 1.0%** |

VitalRoute achieves the best overall accuracy and the lowest variance across seeds, suggesting that composite stress provides a more stable training signal than frequency or stasis alone.

**PyTorch / Fashion-MNIST MLP — 10:1 imbalance, 3 seeds, 20 epochs:**

| Method | Overall | Minority acc |
|---|---|---|
| Uniform | 80.1% ± 0.4% | 72.8% ± 1.2% |
| Focal loss | 80.0% ± 0.3% | 72.6% ± 0.2% |
| Inv-freq | 81.7% ± 0.5% | 77.6% ± 0.9% |
| **VitalRoute** | **81.7% ± 0.2%** | 76.5% ± 0.6% |

VitalRoute matches inv-freq on overall accuracy (81.7%) and shows the lowest variance across seeds. Minority accuracy is 76.5% vs inv-freq's 77.6% — 1.1 points below, not a match; the trade-off is lower variance. Focal loss does not improve over uniform on this benchmark, consistent with prior findings that focal loss benefits are dataset-dependent.

### 6.3 Discussion

On clean long-tail benchmarks, VitalRoute and inverse-frequency weighting converge to similar accuracy. This is expected: rare classes are also the classes where neuron death is most prevalent, so both signals point at the same targets. VitalRoute's advantage would be more pronounced in settings where class difficulty is not purely a function of frequency — confusable classes with sufficient samples, or tasks where class difficulty shifts mid-training. The label-free transfer parent selection has no analogue in frequency-based methods and represents the contribution with the least overlap with existing approaches.

---

## 7. Related Work

### Adaptive class resampling

The closest published work to VitalRoute's class sampler is **ART** [[1]](#ref-1), which periodically refreshes class sampling weights using class-wise macro F1 scores computed on a held-out set. The distinction is in the signal: ART reads an *external* performance metric; VitalRoute reads the *internal* activation state of hidden units. On clean long-tail datasets both converge to similar accuracy, but they respond differently when class difficulty is not a direct function of frequency.

Earlier work on instance-level difficulty includes **Online Hard Example Mining (OHEM)** [[2]](#ref-2), which selects high-loss examples per mini-batch, and the **Self-Paced Learning** framework [[3]](#ref-3). VitalRoute's hard-sample sampler differs by computing difficulty from activation stress rather than loss values, making it applicable even before the model has trained enough to produce informative losses.

### Dead neuron research

Dead ReLU units (neurons that output zero for all inputs) are a well-documented failure mode [[4]](#ref-4). Prior work has used dead-neuron counts to guide **structured pruning** during training [[5]](#ref-5): units that consistently die are candidates for removal. VitalRoute repurposes the same signal in the opposite direction — dead neurons identify *which classes and samples the model is failing on* — driving oversampling rather than removal.

A concurrent approach [[6]](#ref-6) dynamically grows and prunes neurons based on gradient magnitude for minority classes. This modifies model architecture; VitalRoute operates purely through data routing and optimizer state.

### Per-layer learning rate scaling

Several methods assign distinct learning rates per layer. **LARS** [[7]](#ref-7) and **LAMB** [[8]](#ref-8) scale by weight-to-gradient norm ratio, primarily for large-batch distributed training. **LENA** [[9]](#ref-9) scales by gradient variance per layer. **LLR** [[10]](#ref-10) uses heavy-tail spectral analysis of weight matrices. **AdaLip** [[11]](#ref-11) approximates the Lipschitz constant of gradients per layer. VitalRoute's rule `lr_l = base_lr / (1 + α · stasis_l)` uses the dead-unit fraction as the diagnostic signal — a complementary approach motivated by the observation that layers with high stasis are already under-stimulated and should not be pushed harder.

### Label-free transfer model selection

**TURTLE** [[12]](#ref-12) selects pretrained models without target labels by optimising a representation-level generalization objective. **DISCO** [[13]](#ref-13) scores transferability via the distribution of singular values in extracted features. **CODA** [[14]](#ref-14) uses a consensus-driven probabilistic model with active label queries (~25 labels). VitalRoute's criterion — lowest stasis on new unlabeled data — is simpler than these and uses a very different rationale: a model whose neurons remain alive on new inputs is extracting useful structure from them, making it a better transfer source.

### Focal Loss

**Focal Loss** [[15]](#ref-15) re-weights the cross-entropy loss per sample by a factor `(1 - p_t)^γ`, focusing training on hard examples. It is a loss-level intervention; VitalRoute operates at the data-sampling level. On Fashion-MNIST (the included PyTorch benchmark) focal loss did not improve over uniform sampling, consistent with reports that its benefits are dataset- and architecture-dependent.

---

## 8. Limitations

- **Routing thresholds are heuristic.** The values ( \( r < 0.25 \), \( n \leq 200 \), etc.) were chosen empirically on a small set of tasks and may not generalise.
- **Probe overhead.** Computing per-class stress requires one forward pass per class per refresh epoch. For large models and many classes this cost is non-trivial.
- **ResNet / deep CNN stasis.** The stasis probe works best on linear layers with explicit ReLU activations. In architectures with BatchNorm, the normalisation masks stasis in conv layers; the probe reduces to reading stress only on linear head layers.
- **No theoretical convergence guarantee.** The routing and sampling rules are empirically motivated; formal convergence analysis is left to future work.

---

## 9. Conclusion

VitalRoute is a small, composable library that brings network health monitoring into the training loop. The four vitality signals — stasis, weak weights, weak input, saturation — provide a richer picture of per-class difficulty than frequency alone. The adaptive router selects tactics automatically from dataset shape, removing the need for per-dataset configuration. On public benchmarks, VitalRoute matches inverse-frequency weighting on overall accuracy with lower variance; minority accuracy on Fashion-MNIST is slightly below inv-freq (76.5% vs 77.6%) while digits minority exceeds it (+0.7%), and extends cleanly to any PyTorch model via forward hooks.

---

## Reproducibility

All benchmarks are reproducible from the included scripts:

```bash
# NumPy backbone
python examples/benchmark_baselines.py --trials 3 --epochs 30

# PyTorch / Fashion-MNIST
python examples/torch_benchmark_fmnist.py --trials 3 --epochs 20

# ResNet18 / CIFAR-10 (GPU recommended)
python examples/cifar10_resnet_benchmark.py --epochs 30
```

Random seeds are fixed per trial (seed = trial index). No external data beyond scikit-learn digits and standard torchvision datasets is required.

---

## References

<a id="ref-1"></a>[1] Singh, A. et al. **ART: Adaptive Resampling-based Training for Imbalanced Classification.** arXiv:2509.00955, 2025. <https://arxiv.org/abs/2509.00955>

<a id="ref-2"></a>[2] Shrivastava, A., Gupta, A., and Girshick, R. **Training Region-based Object Detectors with Online Hard Example Mining.** CVPR, 2016. <https://arxiv.org/abs/1604.03540>

<a id="ref-3"></a>[3] Kumar, M. P. et al. **Self-Paced Learning for Latent Variable Models.** NeurIPS, 2010. <https://papers.nips.cc/paper_files/paper/2010/hash/e57c6b956a6521b28495f2886ca0977a-Abstract.html>

<a id="ref-4"></a>[4] Overview of dead neurons in deep learning. <https://medium.com/@abhishekjainindore24/dead-neurons-in-deep-learning-their-effects-and-remedies-to-solve-it-e63da4dd9212>

<a id="ref-5"></a>[5] **When to Prune? A Policy towards Early Structural Pruning.** OpenReview. <https://openreview.net/pdf?id=2wFXD2upSQ>

<a id="ref-6"></a>[6] **Adaptive Neuron Growth and Pruning for Imbalanced Classification.** arXiv:2507.09940, 2025. <https://arxiv.org/abs/2507.09940>

<a id="ref-7"></a>[7] You, Y. et al. **Large Batch Training of Convolutional Networks (LARS).** arXiv:1708.03888, 2017. <https://arxiv.org/abs/1708.03888>

<a id="ref-8"></a>[8] You, Y. et al. **Large Batch Optimization for Deep Learning: Training BERT in 76 minutes (LAMB).** ICLR, 2020. <https://arxiv.org/abs/1904.00962>

<a id="ref-9"></a>[9] **Not All Layers Are Equal: A Layer-Wise Adaptive Approach Toward Large-Scale DNN Training (LENA).** WWW, 2022. <https://dl.acm.org/doi/fullHtml/10.1145/3485447.3511989>

<a id="ref-10"></a>[10] **One LR Doesn't Fit All: Heavy-Tail Guided Layerwise Learning Rates for LLMs (LLR).** arXiv:2605.22297, 2025. <https://arxiv.org/html/2605.22297v1>

<a id="ref-11"></a>[11] **AdaLip: An Adaptive Learning Rate Method per Layer for Stochastic Optimization.** <https://d-nb.info/1283272997/34>

<a id="ref-12"></a>[12] Gadetsky, A. and Brbić, M. **Let Go of Your Labels with Unsupervised Transfer (TURTLE).** arXiv:2406.07236, 2024. <https://arxiv.org/html/2406.07236v1>

<a id="ref-13"></a>[13] **Assessing Pre-Trained Models for Transfer Learning Through Distribution of Spectral Components (DISCO).** arXiv:2412.19085, 2024. <https://arxiv.org/html/2412.19085v2>

<a id="ref-14"></a>[14] **CODA: Consensus-Driven Active Model Selection.** arXiv:2507.23771, 2025. <https://arxiv.org/abs/2507.23771>

<a id="ref-15"></a>[15] Lin, T.-Y. et al. **Focal Loss for Dense Object Detection.** ICCV, 2017. <https://arxiv.org/abs/1708.02002>

