import copy
import unittest

import numpy as np

from tests._common import MC_SE, TOL_ALG, UNIT_ROOT, load_cfg, rel
from soe.envgen import (aniso_spectrum, block_specs, env_specs, make_environment, rng_for, sample_split,
                        STREAM_DATA)
from soe.kron import rearrange


class TestEnvGen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_cfg()
        cls.specs = env_specs(cls.cfg)

    def test_grid_layout(self):
        self.assertEqual(len(self.specs), 24)
        self.assertEqual(len(block_specs(self.cfg)), 12)
        cells = [(s.block.q, s.block.delta) for s in self.specs]
        for q in (2, 6):
            for d in (0.2, 1.0):
                self.assertEqual(cells.count((q, d)), 6)
        for b in range(12):
            kinds = sorted(s.cov_kind for s in self.specs if s.block.block == b)
            self.assertEqual(kinds, ["aniso", "iso"])

    def test_seed_roots_distinct(self):
        s = self.cfg["seeds"]
        self.assertEqual(len({s["scientific_root"], s["technical_root"], s["unit_test_root"], s["warmup_root"]}), 4)
        a = make_environment(self.cfg, s["technical_root"], self.specs[0]).teacher.T
        b = make_environment(self.cfg, s["unit_test_root"], self.specs[0]).teacher.T
        self.assertGreater(rel(a, b), 0.1)

    def test_teacher_structure(self):
        nu = self.cfg["teacher"]["component_frobenius_norm"]
        for spec in (self.specs[0], self.specs[18]):
            env = make_environment(self.cfg, UNIT_ROOT, spec)
            tch = env.teacher
            q, delta = spec.block.q, spec.block.delta
            self.assertLess(abs(np.linalg.norm(tch.common) - nu), TOL_ALG * nu)
            for t in range(4):
                self.assertLess(abs(np.linalg.norm(tch.specific[t]) - nu), TOL_ALG * nu)
                struct = tch.common + delta * tch.specific[t]
                self.assertLess(abs(np.linalg.norm(struct) - tch.struct_norm[t]), TOL_ALG * nu)
                self.assertLess(abs(np.linalg.norm(tch.residual[t]) - 0.1 * tch.struct_norm[t]), TOL_ALG * nu)
                self.assertLess(rel(tch.T[t], struct + tch.residual[t]), TOL_ALG)
            for M in (tch.common, tch.specific[2]):
                s = np.linalg.svd(rearrange(M), compute_uv=False)
                self.assertGreater(s[q - 1] / s[0], 1e-8)
                self.assertLess(s[q] / s[0], 1e-10)
            # residual is dense: its rearranged spectrum is not low rank
            s = np.linalg.svd(rearrange(tch.residual[0]), compute_uv=False)
            self.assertGreater(s[30] / s[0], 0.1)

    def test_teacher_identical_across_covariance_pair(self):
        iso = make_environment(self.cfg, UNIT_ROOT, self.specs[4])
        an = make_environment(self.cfg, UNIT_ROOT, self.specs[5])
        self.assertEqual(iso.spec.block.block, an.spec.block.block)
        self.assertTrue(np.array_equal(iso.teacher.T, an.teacher.T))
        self.assertTrue(np.array_equal(iso.cov.Sigma, np.broadcast_to(np.eye(64), (4, 64, 64))))
        self.assertFalse(np.array_equal(iso.cov.Sigma, an.cov.Sigma))

    def test_aniso_covariance(self):
        lam = aniso_spectrum(self.cfg)
        self.assertLess(abs(lam.sum() - 64), 1e-10)
        self.assertLess(abs(lam.max() / lam.min() - 4.0), 1e-10)
        env = make_environment(self.cfg, UNIT_ROOT, self.specs[1])
        for t in range(4):
            ev = np.sort(np.linalg.eigvalsh(env.cov.Sigma[t]))[::-1]
            self.assertLess(rel(ev, lam), 1e-10)
            self.assertLess(rel(env.cov.L[t] @ env.cov.L[t].T, env.cov.Sigma[t]), 1e-10)
        self.assertGreater(rel(env.cov.Sigma[0], env.cov.Sigma[1]), 0.01)  # task-dependent rotation

    def test_noise_and_normalizer(self):
        env = make_environment(self.cfg, UNIT_ROOT, self.specs[3])
        T, Sig = env.teacher.T, env.cov.Sigma
        sig = np.array([np.trace(T[t] @ Sig[t] @ T[t].T) for t in range(4)])
        self.assertLess(rel(env.signal_energy, sig), TOL_ALG)
        self.assertLess(rel(64 * env.noise_var, 0.05 * sig), TOL_ALG)
        self.assertLess(abs(env.zero_model_half_loss - np.sum(0.25 * 0.5 * 1.05 * sig)) / env.zero_model_half_loss, TOL_ALG)
        X, Y = sample_split(self.cfg, env, "test", n=30000)
        for t in range(4):
            eps = Y[t] - X[t] @ T[t].T
            e = np.sum(eps ** 2, axis=1)
            self.assertLess(abs(e.mean() - 64 * env.noise_var[t]), MC_SE * e.std() / np.sqrt(len(e)))
            s = np.sum((X[t] @ T[t].T) ** 2, axis=1)
            self.assertLess(abs(s.mean() - sig[t]), MC_SE * s.std() / np.sqrt(len(s)))

    def test_stream_integrity(self):
        env = make_environment(self.cfg, UNIT_ROOT, self.specs[7])
        p1 = sample_split(self.cfg, env, "pilot")
        _ = sample_split(self.cfg, env, "test")
        p2 = sample_split(self.cfg, env, "pilot")
        self.assertTrue(np.array_equal(p1[0], p2[0]) and np.array_equal(p1[1], p2[1]))
        # changing another split's size does not change the pilot
        cfg2 = copy.deepcopy(self.cfg)
        cfg2["data_sizes"]["train"] = 7
        env2 = make_environment(cfg2, UNIT_ROOT, self.specs[7])
        self.assertTrue(np.array_equal(sample_split(cfg2, env2, "pilot")[0], p1[0]))
        # all (task, split, env) data streams are distinct
        firsts = set()
        for eid in (6, 7):
            for t in range(4):
                for sp in range(4):
                    key = (STREAM_DATA, 3, eid % 2, t, sp)
                    firsts.add(rng_for(UNIT_ROOT, key).standard_normal())
        self.assertEqual(len(firsts), 32)
        # pilot and train inputs uncorrelated
        tr = sample_split(self.cfg, env, "train", n=64)
        c = np.corrcoef(p1[0][0].ravel(), tr[0][0].ravel())[0, 1]
        self.assertLess(abs(c), 0.05)
        # tasks within a split differ
        self.assertFalse(np.array_equal(p1[0][0], p1[0][1]))


if __name__ == "__main__":
    unittest.main()
