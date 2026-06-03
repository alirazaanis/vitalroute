"""Tests for CNN probes, transfer, LR scale, and MLPerf hooks."""

from __future__ import annotations

import unittest
import numpy as np

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _tiny_cnn(num_classes: int = 4) -> "nn.Module":
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1), nn.BatchNorm2d(8), nn.ReLU(),
        nn.Conv2d(8, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, num_classes),
    )


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchData(unittest.TestCase):
    def test_stratified_probe_indices(self):
        from vitalroute.torch_data import stratified_probe_indices

        y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2])
        idx = stratified_probe_indices(y, per_class=2, num_classes=3, seed=0)
        picked = y[idx]
        for c in range(3):
            self.assertGreaterEqual((picked == c).sum(), 1)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestCNNVitalityProbe(unittest.TestCase):
    def test_zones(self):
        from vitalroute.torch_probe_cnn import CNNVitalityProbe

        model = _tiny_cnn()
        X = torch.randn(8, 3, 8, 8)

        for zone in ("head", "trunk", "all"):
            probe = CNNVitalityProbe(model, probe_zone=zone)
            probe.observe(X)
            names = probe.layer_names()
            self.assertGreater(len(names), 0)
            rates = probe.stasis_rates()
            self.assertEqual(len(rates), len(names))
            probe.detach()

    def test_detect_and_make_probe(self):
        from vitalroute.torch_probes import detect_architecture, make_probe

        model = _tiny_cnn()
        self.assertEqual(detect_architecture(model), "cnn")
        probe = make_probe(model, probe_zone="all")
        probe.observe(torch.randn(4, 3, 8, 8))
        self.assertGreaterEqual(probe.mean_stasis(), 0.0)
        probe.detach()


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchTransfer(unittest.TestCase):
    def test_warm_start(self):
        from vitalroute.torch_transfer import warm_start_from_parent

        parent = _tiny_cnn()
        child = _tiny_cnn()
        info = warm_start_from_parent(child, parent, fresh_head=True)
        self.assertGreater(info["n_loaded"], 0)
        self.assertGreater(info["n_skipped_head"], 0)

    def test_pick_parent(self):
        from vitalroute.torch_transfer import pick_transfer_parent_torch

        X = torch.randn(16, 3, 8, 8)
        p1 = _tiny_cnn()
        p2 = _tiny_cnn()
        name, _, scores = pick_transfer_parent_torch(
            [("a", p1), ("b", p2)], X, probe_zone="head"
        )
        self.assertIn(name, ("a", "b"))
        self.assertEqual(len(scores), 2)


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchLRScale(unittest.TestCase):
    def test_param_groups_and_scale(self):
        from vitalroute.torch_lr_scale import apply_lr_scales, build_layer_param_groups, make_vitality_optimizer
        from vitalroute.torch_probes import make_probe

        model = _tiny_cnn()
        probe = make_probe(model, probe_zone="head")
        probe.observe(torch.randn(4, 3, 8, 8))
        groups = build_layer_param_groups(model, probe, base_lr=0.1)
        self.assertTrue(any(g.get("name") != "_rest" for g in groups))
        opt = make_vitality_optimizer(model, probe, torch.optim.SGD, lr=0.1, momentum=0.9)
        scales = apply_lr_scales(opt, probe, alpha=4.0)
        self.assertGreater(len(scales), 0)
        probe.detach()


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestMLPerfCallback(unittest.TestCase):
    def test_callback_smoke(self):
        from torch.utils.data import TensorDataset
        from vitalroute.mlperf_hooks import VitalRouteMLPerfCallback

        X = torch.randn(40, 3, 8, 8)
        y = torch.from_numpy(np.repeat(np.arange(4), 10).astype(np.int64))
        ds = TensorDataset(X, y)
        model = _tiny_cnn()

        cb = VitalRouteMLPerfCallback(y.numpy(), 4, architecture="cnn", probe_zone="head", verbose=False)
        cb.before_train(model, ds, device="cpu")
        opt = cb.make_optimizer(model, torch.optim.SGD, lr=0.01)
        cb.on_epoch_begin(0, model, opt)
        loader = cb.get_train_loader(ds, batch_size=8)
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()
        metrics = cb.on_epoch_end(0, model)
        self.assertIn("mean_stasis", metrics)
        tags = cb.log_mlperf_tags()
        self.assertTrue(tags["vitalroute_enabled"])
        cb.after_train()


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchControllerCNN(unittest.TestCase):
    def test_cnn_controller_setup(self):
        from vitalroute.torch_controller import torch_adaptive_controller

        y = np.concatenate([np.zeros(100), np.ones(20)]).astype(np.int64)
        model = _tiny_cnn(num_classes=2)
        X = torch.randn(24, 3, 8, 8)
        y_probe = torch.from_numpy(np.repeat([0, 1], 12).astype(np.int64))

        ctrl = torch_adaptive_controller(y, 2, architecture="cnn", probe_zone="all", verbose=False)
        sampler = ctrl.setup(model, X, y_probe, y_full=y, num_classes=2)
        self.assertIsNotNone(sampler)
        opt = ctrl.make_optimizer(model, torch.optim.SGD, lr=0.01)
        ctrl.on_epoch_start(model, X, opt, 0)
        ctrl.after_epoch(model, X, y_probe)
        ctrl.detach()


if __name__ == "__main__":
    unittest.main()
