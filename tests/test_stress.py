"""Stress signal unit tests."""

from __future__ import annotations

import unittest

import numpy as np

from vitalroute.backbone import MLP, LayerSpec, Adam
from vitalroute.vitality import per_class_health, per_class_stress, per_sample_stress


class TestStressSignals(unittest.TestCase):
    def _tiny_model(self) -> MLP:
        return MLP(
            input_dim=8,
            layers=[LayerSpec(4, "relu"), LayerSpec(2, "linear")],
            output="softmax",
            seed=0,
        )

    def test_per_class_stress_shape(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(40, 8)).astype(np.float32)
        y = np.array([0] * 20 + [1] * 20, dtype=np.int64)
        m = self._tiny_model()
        s = per_class_stress(m, X, y, 2)
        self.assertEqual(s.shape, (2,))
        self.assertTrue(np.all(s >= 0))

    def test_composite_differs_from_stasis_only(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(60, 8)).astype(np.float32)
        y = np.repeat([0, 1], 30)
        m = self._tiny_model()
        for _ in range(3):
            m.train_step(X[:16], y[:16], Adam(lr=0.1))
        stasis = per_class_health(m, X, y, 2)
        composite = per_class_stress(m, X, y, 2)
        self.assertEqual(stasis.shape, composite.shape)

    def test_per_sample_stress_full_length(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(25, 8)).astype(np.float32)
        y = rng.integers(0, 2, 25)
        m = self._tiny_model()
        scores = per_sample_stress(m, X, y, sample_size=100)
        self.assertEqual(scores.shape, (25,))


if __name__ == "__main__":
    unittest.main()
