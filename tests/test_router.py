"""Router unit tests."""

from __future__ import annotations

import unittest

import numpy as np

from vitalroute import profile_task, route_plan


class TestRouter(unittest.TestCase):
    def test_imbalance_route(self):
        y = np.concatenate([np.zeros(3000, int), np.ones(400, int)])
        prof = profile_task(y, 2)
        plan = route_plan(prof, parent_pool_available=False)
        self.assertTrue(plan.use_imbalance_sampler)
        self.assertFalse(plan.use_transfer_pick)

    def test_scarce_transfer(self):
        y = np.repeat(np.arange(5), 10)
        prof = profile_task(y, 5)
        plan = route_plan(prof, parent_pool_available=True)
        self.assertTrue(plan.use_transfer_pick)
        self.assertFalse(plan.use_imbalance_sampler)

    def test_starved_binary_transfer(self):
        y = np.repeat([0, 1], 80)
        prof = profile_task(y, 2)
        plan = route_plan(prof, parent_pool_available=True)
        self.assertTrue(plan.use_transfer_pick)
        self.assertIn("transfer", plan.label)

    def test_easy_monitor(self):
        y = np.repeat(np.arange(10), 140)
        prof = profile_task(y, 10)
        plan = route_plan(prof, parent_pool_available=False)
        self.assertIn("lr_scale", plan.label)
        self.assertTrue(plan.use_lr_scale)
        self.assertTrue(plan.use_hard_sample_sampler)

    def test_scarce_transfer_hard_no_imbalance(self):
        y = np.repeat([0, 1], 80)
        prof = profile_task(y, 2)
        plan = route_plan(prof, parent_pool_available=True)
        self.assertTrue(plan.use_transfer_pick)
        self.assertFalse(plan.use_imbalance_sampler)
        self.assertTrue(plan.use_hard_sample_sampler)


if __name__ == "__main__":
    unittest.main()
