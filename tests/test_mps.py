import unittest

import numpy as np
import torch

from tests._common import TOL_MPS32, UNIT_ROOT, load_cfg
from soe.candidates import CANDIDATES
from soe.envgen import env_specs, make_environment, moments, sample_split
from soe.train import FusedCandidates, torch_moments, train_fused


@unittest.skipUnless(torch.backends.mps.is_available(), "MPS not available in this process")
class TestMPSPrecision(unittest.TestCase):
    def test_float32_mps_matches_float64_cpu(self):
        cfg = load_cfg()
        env = make_environment(cfg, UNIT_ROOT, env_specs(cfg)[1])
        mom = moments(*sample_split(cfg, env, "train"))
        masters = np.random.default_rng(UNIT_ROOT).standard_normal((2, 4, 8, 8, 8))
        ref = train_fused(FusedCandidates(CANDIDATES, masters, env.task_weights),
                          torch_moments(mom, torch.float64, "cpu"), cfg["training"], 20)
        m32 = FusedCandidates(CANDIDATES, masters, env.task_weights, torch.float32, "mps")
        got = train_fused(m32, torch_moments(mom, torch.float32, "mps"), cfg["training"], 20)
        self.assertTrue(np.all(np.isfinite(got)))
        scale = env.zero_model_half_loss
        self.assertLess(np.max(np.abs(got - ref)) / scale, TOL_MPS32)


if __name__ == "__main__":
    unittest.main()
