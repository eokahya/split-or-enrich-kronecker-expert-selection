import copy
import unittest

import numpy as np
import torch

from tests._common import TOL_MPS32, UNIT_ROOT, load_cfg, rel
from soe.candidates import CANDIDATES
from soe.early import early_select
from soe.envgen import PilotView, env_specs, make_environment, moments, sample_split
from soe.train import FusedCandidates, moment_loss, export_factors, torch_moments, train_fused


class FakeClock:
    def __init__(self, step):
        self.t, self.step = 0.0, step

    def __call__(self):
        v = self.t
        self.t += self.step
        return v


class TestEarly(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_cfg()
        env = make_environment(cls.cfg, UNIT_ROOT, env_specs(cls.cfg)[5])
        X, Y = sample_split(cls.cfg, env, "pilot")
        cls.pv = PilotView.from_arrays(X, Y, env.task_weights)
        cls.master = np.random.default_rng(UNIT_ROOT).standard_normal((4, 8, 8, 8))

    def test_not_measurable_when_budget_below_one_round(self):
        # clock reads: start 0, after setup 1, after round 1 -> elapsed 2
        r = early_select(self.pv, self.master, self.cfg, budget_s=1.5, clock=FakeClock(1.0))
        self.assertEqual(r["status"], "NOT_MEASURABLE")
        self.assertIsNone(r["selected"])
        self.assertAlmostEqual(r["overrun_s"], 0.5)
        self.assertAlmostEqual(r["setup_s"], 1.0)
        self.assertAlmostEqual(r["first_round_plus_eval_s"], 1.0)
        self.assertGreaterEqual(r["wall_s"], 2.0)

    def test_completed_round_policy(self):
        # round k completes at elapsed 1 + k; budget 3.5 -> rounds 1, 2 fit; round 3 (4.0) overruns
        r = early_select(self.pv, self.master, self.cfg, budget_s=3.5, clock=FakeClock(1.0))
        self.assertEqual(r["status"], "MEASURED")
        self.assertEqual(r["rounds_in_budget"], 2)
        self.assertAlmostEqual(r["overrun_s"], 0.5)
        self.assertAlmostEqual(r["selection_elapsed_s"], 3.0)
        self.assertGreaterEqual(r["wall_s"], 4.0)  # consumed time includes the discarded overrun round
        self.assertEqual(r["rounds_attempted"], 3)
        f = early_select(self.pv, self.master, self.cfg, fixed_rounds=2)
        self.assertIsNone(f["first_round_plus_eval_s"])  # not mislabelled in fixed mode
        self.assertEqual(r["selected"], f["selected"])
        self.assertLess(rel(r["val_losses"], f["val_losses"]), 1e-12)

    def test_round_robin_equals_separate_training(self):
        # structural equivalence is checked in float64; the deployed float32 selector separately below
        k = 7
        cfg64 = copy.deepcopy(self.cfg)
        cfg64["early_selector"]["dtype"] = "float64"
        f = early_select(self.pv, self.master, cfg64, fixed_rounds=k)
        f32 = early_select(self.pv, self.master, self.cfg, fixed_rounds=k)
        self.assertEqual(f32["dtype"], self.cfg["early_selector"]["dtype"])
        self.assertLess(rel(f32["val_losses"], f["val_losses"]), TOL_MPS32)
        mtr = moments(self.pv.X[:, :32], self.pv.Y[:, :32])
        mva = moments(self.pv.X[:, 32:], self.pv.Y[:, 32:])
        vals = []
        for c in CANDIDATES:
            m = FusedCandidates([c], self.master[None], self.pv.task_weights)
            train_fused(m, torch_moments(mtr, torch.float64, "cpu"), self.cfg["training"], k)
            K = export_factors(m)[0][2][0]
            vals.append(moment_loss(K, c.assignment, mva, self.pv.task_weights))
        self.assertLess(rel(f["val_losses"], vals), 1e-9)
        self.assertEqual(f["selected"], int(np.argmin(vals)))

    def test_identical_initialisation_across_candidates(self):
        m = FusedCandidates(CANDIDATES, self.master[None], self.pv.task_weights)
        a0 = m.A_of(0).detach().numpy()[0, 0, :2] * np.sqrt(64)   # E1 expert 0, first two terms
        a8 = m.A_of(8).detach().numpy()[0, 0, :2] * np.sqrt(16)   # E4 expert 0, both terms
        self.assertLess(rel(a0, a8), 1e-12)


if __name__ == "__main__":
    unittest.main()
