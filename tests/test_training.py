import unittest

import numpy as np
import torch

from tests._common import MC_SE, TOL_ALG, UNIT_ROOT, load_cfg, rel
from soe.candidates import CANDIDATES
from soe.envgen import env_specs, make_environment, moments, sample_split
from soe.kron import dense_from_factors_np
from soe.selectors import naive_scores_captured
from soe.train import (FusedCandidates, init_factors, moment_loss, population_risk, sample_loss_factorized,
                       torch_moments, train_fused)

TOL_FUSED = 1e-9  # fused-vs-separate Adam trajectories after 15 steps (declared)


def analytic_factor_grads(A, B, assignment, mom, w):
    """Independent numpy gradient of sum_t w_t L_t wrt factors A, B of shape (E, r, 8, 8)."""
    K = dense_from_factors_np(A, B)
    gA, gB = np.zeros_like(A), np.zeros_like(B)
    for e in range(A.shape[0]):
        G = np.zeros((64, 64))
        for t, g in enumerate(assignment):
            if g == e:
                G += w[t] * (K[e] @ mom["Sxx"][t] - mom["Syx"][t])
        G4 = G.reshape(8, 8, 8, 8)  # (a, c, b, d)
        for j in range(A.shape[1]):
            gA[e, j] = np.einsum("acbd,cd->ab", G4, B[e, j])
            gB[e, j] = np.einsum("acbd,ab->cd", G4, A[e, j])
    return gA, gB


class TestTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_cfg()
        cls.env = make_environment(cls.cfg, UNIT_ROOT, env_specs(cls.cfg)[3])  # aniso
        X, Y = sample_split(cls.cfg, cls.env, "train", n=200)
        cls.X, cls.Y = X, Y
        cls.mom = moments(X, Y)
        cls.w = cls.env.task_weights
        rng = np.random.default_rng(UNIT_ROOT)
        cls.masters = rng.standard_normal((2, 4, 8, 8, 8))

    def test_init(self):
        for c in CANDIDATES:
            A, B = init_factors(self.masters[0], c)
            self.assertEqual(A.shape, (c.n_experts, c.rank, 8, 8))
            self.assertTrue(np.all(B == 0))
            self.assertLess(rel(A * np.sqrt(8 * c.rank), self.masters[0][:c.n_experts, :c.rank]), TOL_ALG)

    def test_zero_init_gradients(self):
        for c in (CANDIDATES[0], CANDIDATES[3], CANDIDATES[8]):
            m = FusedCandidates([c], self.masters[:1], self.w)
            L = m.weighted_losses(torch_moments(self.mom, torch.float64, "cpu")).sum()
            L.backward()
            A, B = m.A_of(0).detach().numpy()[0], m.B_of(0).detach().numpy()[0]
            gA, gB = analytic_factor_grads(A, B, c.assignment, self.mom, self.w)
            tA, tB = m.grads_of(0)
            self.assertTrue(np.all(tA.numpy() == 0))
            self.assertTrue(np.all(gA == 0))
            self.assertGreater(np.abs(gB).max(), 0)
            self.assertLess(rel(tB.numpy()[0], gB), TOL_ALG)

    def test_moment_loss_equals_sample_loss(self):
        rng = np.random.default_rng(1)
        for c in (CANDIDATES[0], CANDIDATES[5], CANDIDATES[8]):
            A = rng.standard_normal((c.n_experts, c.rank, 8, 8)) * 0.2
            B = rng.standard_normal((c.n_experts, c.rank, 8, 8)) * 0.2
            K = dense_from_factors_np(A, B)
            lm = moment_loss(K, c.assignment, self.mom, self.w)
            ls = sum(self.w[t] * 0.5 * np.mean(np.sum((self.Y[t] - self.X[t] @ K[g].T) ** 2, 1))
                     for t, g in enumerate(c.assignment))
            lf = sample_loss_factorized(A, B, c.assignment, self.X, self.Y, self.w)
            self.assertLess(abs(lm - ls) / ls, TOL_ALG)
            self.assertLess(abs(lf - ls) / ls, TOL_ALG)

    def test_adam_matches_manual_reference(self):
        cfg = self.cfg["training"]
        c = CANDIDATES[2]
        m = FusedCandidates([c], self.masters[:1], self.w)
        train_fused(m, torch_moments(self.mom, torch.float64, "cpu"), cfg, 10)
        A, B = init_factors(self.masters[0], c)
        mA, vA, mB, vB = (np.zeros_like(A) for _ in range(4))
        b1, b2 = cfg["betas"]
        for k in range(1, 11):
            gA, gB = analytic_factor_grads(A, B, c.assignment, self.mom, self.w)
            for p, g, mm, vv in ((A, gA, mA, vA), (B, gB, mB, vB)):
                mm[:] = b1 * mm + (1 - b1) * g
                vv[:] = b2 * vv + (1 - b2) * g * g
                p -= cfg["lr"] * (mm / (1 - b1 ** k)) / (np.sqrt(vv / (1 - b2 ** k)) + cfg["eps"])
        self.assertLess(rel(m.A_of(0).detach().numpy()[0], A), 1e-8)
        self.assertLess(rel(m.B_of(0).detach().numpy()[0], B), 1e-8)

    def test_fused_equals_separate(self):
        cfg = self.cfg["training"]
        mt = torch_moments(self.mom, torch.float64, "cpu")
        fused = FusedCandidates(CANDIDATES, self.masters, self.w)
        tr = train_fused(fused, mt, cfg, 15)
        for ci, c in enumerate(CANDIDATES):
            for s in range(2):
                single = FusedCandidates([c], self.masters[s:s + 1], self.w)
                tr1 = train_fused(single, mt, cfg, 15)
                self.assertLess(rel(single.B_of(0).detach().numpy()[0], fused.B_of(ci).detach().numpy()[s]), TOL_FUSED)
                self.assertLess(rel(single.A_of(0).detach().numpy()[0], fused.A_of(ci).detach().numpy()[s]), TOL_FUSED)
                self.assertLess(rel(tr1[:, 0, 0], tr[:, ci, s]), TOL_FUSED)

    def test_population_risk_against_sampled_loss_aniso(self):
        self.assertEqual(self.env.spec.cov_kind, "aniso")
        c = CANDIDATES[8]  # one expert per task
        T, Sig = self.env.teacher.T, self.env.cov.Sigma
        # error operator concentrated on each task's top-8 covariance directions, so Sigma matters
        Dl = []
        for t in range(4):
            ev, V = np.linalg.eigh(Sig[t])
            top = V[:, np.argsort(ev)[::-1][:8]]
            Dl.append(3.0 * top @ top.T)
        K = T - np.stack(Dl)
        X, Y = sample_split(self.cfg, self.env, "test", n=40000)
        per, var = 0.0, 0.0
        for t, g in enumerate(c.assignment):
            l = 0.5 * np.sum((Y[t] - X[t] @ K[g].T) ** 2, axis=1)
            per += self.w[t] * l.mean()
            var += self.w[t] ** 2 * l.var() / len(l)
        pop = population_risk(K, c.assignment, T, Sig, self.env.noise_var, self.w)
        self.assertLess(abs(per - pop), MC_SE * np.sqrt(var))
        # a deliberately wrong (isotropic) formula is rejected by the same check
        wrong = float(np.sum(self.w * 0.5 * (np.sum(np.stack(Dl) ** 2, axis=(1, 2)) + 64 * self.env.noise_var)))
        self.assertGreater(abs(per - wrong), MC_SE * np.sqrt(var))

    def test_adam_cannot_beat_isotropic_svd_optimum(self):
        env = make_environment(self.cfg, UNIT_ROOT, env_specs(self.cfg)[0])
        T, w = env.teacher.T, env.task_weights
        pop_mom = {"Sxx": np.broadcast_to(np.eye(64), (4, 64, 64)).copy(), "Syx": T.copy(),
                   "syy": np.einsum("tij,tij->t", T, T) + 64 * env.noise_var}
        m = FusedCandidates(CANDIDATES, self.masters[:1], w)
        tr = train_fused(m, torch_moments(pop_mom, torch.float64, "cpu"), self.cfg["training"], 100)
        opt = naive_scores_captured(T, w) + float(np.sum(w * 0.5 * 64 * env.noise_var))
        self.assertTrue(np.all(tr[-1, :, 0] >= opt * (1 - 1e-12)))


if __name__ == "__main__":
    unittest.main()
