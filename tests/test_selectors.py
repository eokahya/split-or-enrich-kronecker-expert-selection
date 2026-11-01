import dataclasses
import inspect
import unittest

import numpy as np

from tests._common import TOL_ALG, UNIT_ROOT, load_cfg, rel
from soe import selectors as S
from soe.candidates import CANDIDATES
from soe.envgen import PilotView, make_environment, env_specs, sample_split
from soe.kron import best_kron_approx, dense_from_factors_np, rearrange
from soe.train import population_risk


def direct_constrained_objective(P, w, cand):
    """sum_t w_t ||P_t - K_g(t)||^2 at the closed-form optimum (independent of the decomposition)."""
    Ks = S.constrained_optimum(P, w, cand)
    return sum(w[t] * np.sum((P[t] - Ks[g]) ** 2) for t, g in enumerate(cand.assignment))


class TestScores(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(UNIT_ROOT)
        self.P = self.rng.standard_normal((4, 64, 64))
        self.w = np.full(4, 0.25)

    def test_two_implementations_agree(self):
        for w in (self.w, np.array([0.1, 0.2, 0.3, 0.4])):
            a = S.naive_scores_decomposed(self.P, w)
            b = S.naive_scores_captured(self.P, w)
            self.assertLess(rel(a, b), TOL_ALG)

    def test_score_equals_half_constrained_objective(self):
        for w in (self.w, np.array([0.1, 0.2, 0.3, 0.4])):
            sc = S.naive_scores_decomposed(self.P, w)
            for c in CANDIDATES:
                self.assertLess(abs(sc[c.index] - 0.5 * direct_constrained_objective(self.P, w, c)) / sc[c.index], TOL_ALG)

    def test_unequal_group_weights_three_plus_one(self):
        c = CANDIDATES[1]
        self.assertEqual(c.assignment, (0, 0, 0, 1))
        P, w = self.P, self.w
        mean3 = (P[0] + P[1] + P[2]) / 3
        s3 = np.linalg.svd(rearrange(mean3), compute_uv=False)
        s1 = np.linalg.svd(rearrange(P[3]), compute_uv=False)
        var = 0.25 * sum(np.sum((P[t] - mean3) ** 2) for t in range(3))
        hand = 0.5 * (var + 0.75 * np.sum(s3[4:] ** 2) + 0.25 * np.sum(s1[4:] ** 2))
        self.assertLess(abs(S.naive_scores_captured(P, w)[1] - hand) / hand, TOL_ALG)
        # a wrong (unweighted) tail combination differs materially
        wrong = 0.5 * (var + 0.5 * np.sum(s3[4:] ** 2) + 0.5 * np.sum(s1[4:] ** 2))
        self.assertGreater(abs(wrong - hand) / hand, 1e-3)

    def test_isotropic_population_identity(self):
        cfg = load_cfg()
        env = make_environment(cfg, UNIT_ROOT, env_specs(cfg)[0])  # iso
        self.assertEqual(env.spec.cov_kind, "iso")
        T, w = env.teacher.T, env.task_weights
        noise = float(np.sum(w * 0.5 * 64 * env.noise_var))
        sc = S.naive_scores_captured(T, w)
        for c in CANDIDATES:
            Ks = np.stack(S.constrained_optimum(T, w, c))
            risk = population_risk(Ks, c.assignment, T, env.cov.Sigma, env.noise_var, w)
            self.assertLess(abs(risk - (noise + sc[c.index])) / risk, TOL_ALG)
            # perturbing the optimum's own Kronecker factors (staying inside the class) never helps
            A0, B0 = [], []
            for K in Ks:
                U, s, Vt = np.linalg.svd(rearrange(K))
                A0.append((U[:, :c.rank] * np.sqrt(s[:c.rank])).T.reshape(c.rank, 8, 8))
                B0.append((Vt[:c.rank].T * np.sqrt(s[:c.rank])).T.reshape(c.rank, 8, 8))
            A0, B0 = np.stack(A0), np.stack(B0)
            self.assertLess(rel(dense_from_factors_np(A0, B0), Ks), 1e-9)
            for _ in range(3):
                Kp = dense_from_factors_np(A0 + 1e-3 * self.rng.standard_normal(A0.shape),
                                           B0 + 1e-3 * self.rng.standard_normal(B0.shape))
                self.assertGreaterEqual(population_risk(Kp, c.assignment, T, env.cov.Sigma, env.noise_var, w), risk)

    def test_crossfit_sign_identity(self):
        P1, P2, w = self.P, self.rng.standard_normal((4, 64, 64)), self.w
        # same half for fit and evaluation -> naive score minus constant
        same = S.crossfit_scores(P1, P1, w)
        const = 0.5 * np.sum(w * np.sum(P1 ** 2, axis=(1, 2)))
        self.assertLess(rel(same, S.naive_scores_captured(P1, w) - const), TOL_ALG)

        def one_way(Pa, Pb):
            out = []
            for c in CANDIDATES:
                Ks = S.constrained_optimum(Pa, w, c)
                out.append(0.5 * sum(w[t] * (np.sum((Pb[t] - Ks[g]) ** 2) - np.sum(Pb[t] ** 2))
                                     for t, g in enumerate(c.assignment)))
            return np.array(out)

        ref = 0.5 * (one_way(P1, P2) + one_way(P2, P1))
        self.assertLess(rel(S.crossfit_scores(P1, P2, w), ref), TOL_ALG)

    def test_covariance_corrected_operator(self):
        X = self.rng.standard_normal((4, 64, 64))
        Y = self.rng.standard_normal((4, 64, 64))
        That = S.covariance_corrected_operators(X, Y, 0.1)
        for t in range(4):
            C = X[t].T @ X[t] / 64
            lam = 0.1 * np.trace(C) / 64
            ref = (Y[t].T @ X[t] / 64) @ np.linalg.inv(C + lam * np.eye(64))
            self.assertLess(rel(That[t], ref), 1e-9)

    def test_ablation_variance_only_prefers_e4(self):
        self.assertEqual(S.argmin_first(S.variance_only_scores(self.P, self.w)), 8)

    def test_tie_break_lowest_index(self):
        self.assertEqual(S.argmin_first([1.0, 0.5, 0.5, 2.0]), 1)


class TestSelectorIsolation(unittest.TestCase):
    def test_pilot_view_has_only_pilot_fields_and_is_readonly(self):
        names = {f.name for f in dataclasses.fields(PilotView)}
        self.assertEqual(names, {"X", "Y", "task_weights"})
        pv = PilotView.from_arrays(np.zeros((4, 64, 64)), np.zeros((4, 64, 64)), [0.25] * 4)
        with self.assertRaises(ValueError):
            pv.X[0, 0, 0] = 1.0
        with self.assertRaises(dataclasses.FrozenInstanceError):
            pv.X = None

    def test_public_selectors_take_only_pilot(self):
        for name in ("select_naive", "select_naive_reference", "select_crossfit", "select_covariance",
                     "select_variance_only", "select_tail_only"):
            params = list(inspect.signature(getattr(S, name)).parameters)
            self.assertEqual(params[0], "pilot")
            self.assertTrue(all(p in ("pilot", "halves", "ridge_fraction") for p in params), (name, params))

    def test_selection_depends_only_on_pilot(self):
        cfg = load_cfg()
        env = make_environment(cfg, UNIT_ROOT, env_specs(cfg)[1])
        X, Y = sample_split(cfg, env, "pilot")
        r1 = S.select_naive(PilotView.from_arrays(X, Y, env.task_weights))
        # corrupt population objects: the selector never sees them, so the result is unchanged
        env.teacher.T[:] = np.nan
        env.cov.Sigma[:] = np.nan
        r2 = S.select_naive(PilotView.from_arrays(X, Y, env.task_weights))
        self.assertEqual(r1, r2)


if __name__ == "__main__":
    unittest.main()
