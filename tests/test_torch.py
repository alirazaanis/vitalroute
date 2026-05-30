"""Tests for PyTorch probe, samplers, and controller.

Skipped automatically when PyTorch is not installed.
"""

from __future__ import annotations

import unittest
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _tiny_mlp(input_dim: int = 16, num_classes: int = 4) -> "nn.Module":
    return nn.Sequential(
        nn.Linear(input_dim, 32), nn.ReLU(),
        nn.Linear(32, 16),        nn.ReLU(),
        nn.Linear(16, num_classes),
    )


def _synthetic(n: int = 120, input_dim: int = 16, num_classes: int = 4,
               seed: int = 0):
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.normal(size=(n, input_dim)).astype(np.float32))
    y = torch.from_numpy(rng.integers(0, num_classes, n).astype(np.int64))
    return X, y


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestVitalityProbe(unittest.TestCase):

    def setUp(self):
        from vitalroute.torch_probe import VitalityProbe
        self.VitalityProbe = VitalityProbe
        self.model = _tiny_mlp()
        self.X, self.y = _synthetic()

    def tearDown(self):
        if hasattr(self, "probe"):
            self.probe.detach()

    def test_attach_detects_relu_layers(self):
        self.probe = self.VitalityProbe(self.model)
        names = self.probe.layer_names()
        # Sequential 0,2,4 are Linear; 0 and 2 should be paired with ReLU
        self.assertGreaterEqual(len(names), 2)

    def test_stasis_rates_shape(self):
        self.probe = self.VitalityProbe(self.model)
        self.probe.observe(self.X)
        rates = self.probe.stasis_rates()
        self.assertEqual(rates.shape[0], len(self.probe.layer_names()))
        self.assertTrue(np.all(rates >= 0))
        self.assertTrue(np.all(rates <= 1))

    def test_composite_stress_shape(self):
        self.probe = self.VitalityProbe(self.model)
        self.probe.observe(self.X)
        cs = self.probe.composite_stress()
        self.assertEqual(cs.shape[0], len(self.probe.layer_names()))

    def test_mean_stasis_scalar(self):
        self.probe = self.VitalityProbe(self.model)
        self.probe.observe(self.X)
        ms = self.probe.mean_stasis()
        self.assertIsInstance(ms, float)
        self.assertGreaterEqual(ms, 0.0)

    def test_per_class_stress_shape(self):
        self.probe = self.VitalityProbe(self.model)
        self.probe.observe(self.X)
        scores = self.probe.per_class_stress(self.X, self.y, num_classes=4)
        self.assertEqual(scores.shape, (4,))
        self.assertTrue(np.all(scores >= 0))

    def test_per_sample_stress_shape(self):
        self.probe = self.VitalityProbe(self.model)
        self.probe.observe(self.X)
        scores = self.probe.per_sample_stress(self.X, self.y)
        self.assertEqual(scores.shape[0], len(self.y))

    def test_detach_clears_units(self):
        self.probe = self.VitalityProbe(self.model)
        self.probe.detach()
        self.assertEqual(len(self.probe._units), 0)

    def test_stasis_nonzero_after_training(self):
        """After training on biased data some ReLU units should show stasis."""
        model = _tiny_mlp()
        self.probe = self.VitalityProbe(model)
        # train only on class 0 — biases the network
        opt = torch.optim.Adam(model.parameters(), lr=0.05)
        X0 = self.X[self.y == 0]
        y0 = self.y[self.y == 0]
        model.train()
        for _ in range(30):
            opt.zero_grad()
            nn.functional.cross_entropy(model(X0), y0).backward()
            opt.step()
        self.probe.observe(self.X)
        # at least one layer should have some stasis
        self.assertGreater(self.probe.mean_stasis(), 0.0)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchSamplers(unittest.TestCase):

    def setUp(self):
        from vitalroute.torch_probe import VitalityProbe
        from vitalroute.torch_samplers import TorchVitalitySampler, TorchHardSampleSampler
        self.VitalityProbe = VitalityProbe
        self.TorchVitalitySampler = TorchVitalitySampler
        self.TorchHardSampleSampler = TorchHardSampleSampler
        self.model = _tiny_mlp()
        self.X, self.y = _synthetic(n=200)

    def test_vitality_sampler_length(self):
        probe = self.VitalityProbe(self.model)
        probe.observe(self.X)
        sampler = self.TorchVitalitySampler(self.y, num_classes=4, probe=probe, seed=0)
        indices = list(sampler)
        self.assertEqual(len(indices), len(self.y))
        self.assertTrue(all(0 <= i < len(self.y) for i in indices))
        probe.detach()

    def test_vitality_sampler_refresh(self):
        probe = self.VitalityProbe(self.model)
        probe.observe(self.X)
        sampler = self.TorchVitalitySampler(self.y, num_classes=4, probe=probe, seed=0)
        scores = sampler.refresh(self.X, self.y, self.model)
        self.assertEqual(scores.shape, (4,))
        probe.detach()

    def test_hard_sampler_length(self):
        probe = self.VitalityProbe(self.model)
        probe.observe(self.X)
        sampler = self.TorchHardSampleSampler(len(self.y), probe=probe, seed=0)
        indices = list(sampler)
        self.assertEqual(len(indices), len(self.y))
        probe.detach()

    def test_hard_sampler_refresh(self):
        probe = self.VitalityProbe(self.model)
        probe.observe(self.X)
        sampler = self.TorchHardSampleSampler(len(self.y), probe=probe, seed=0)
        scores = sampler.refresh(self.X, self.y, self.model)
        self.assertEqual(scores.shape[0], len(self.y))
        probe.detach()


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchController(unittest.TestCase):

    def _run(self, imbalance: bool):
        from vitalroute.torch_controller import torch_adaptive_controller
        if imbalance:
            # 5:1 imbalance to trigger vitality sampler
            y_full = torch.from_numpy(
                np.concatenate([np.zeros(100, int), np.ones(20, int)])
            )
        else:
            y_full = torch.from_numpy(np.repeat(np.arange(4), 30).astype(np.int64))

        num_classes = int(y_full.max().item()) + 1
        model = _tiny_mlp(num_classes=num_classes)

        # stratified probe batch
        X_probe = torch.randn(num_classes * 10, 16)
        y_probe = torch.from_numpy(np.repeat(np.arange(num_classes), 10).astype(np.int64))

        ctrl = torch_adaptive_controller(y_full, num_classes, verbose=False)
        sampler = ctrl.setup(model, X_probe, y_probe,
                             y_full=y_full, num_classes=num_classes)

        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(3):
            ctrl.on_epoch_start(model, X_probe, opt, epoch)
            X_batch = torch.randn(32, 16)
            y_batch = torch.randint(0, num_classes, (32,))
            opt.zero_grad()
            nn.functional.cross_entropy(model(X_batch), y_batch).backward()
            opt.step()
            info = ctrl.after_epoch(model, X_probe, y_probe)
            self.assertIn("mean_stasis", info)

        ctrl.detach()
        return sampler

    def test_imbalanced_returns_sampler(self):
        sampler = self._run(imbalance=True)
        self.assertIsNotNone(sampler)

    def test_balanced_may_return_none_or_hard_sampler(self):
        # balanced small dataset — either hard sampler or None
        sampler = self._run(imbalance=False)
        # just check it doesn't crash; sampler type depends on routing


if __name__ == "__main__":
    unittest.main()
