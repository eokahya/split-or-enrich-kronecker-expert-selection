import unittest

import numpy as np
import torch

from tests._common import FD_H, TOL_ALG, TOL_FD, UNIT_ROOT, rel
from soe.kron import (best_kron_approx, dense_from_factors, dense_from_factors_np, factorized_forward, rearrange,
                      unrearrange)


def rearrange_loops(W, p=8, q=8):
    M = np.zeros((p * p, q * q))
    for i1 in range(p):
        for i2 in range(q):
            for j1 in range(p):
                for j2 in range(q):
                    M[i1 * p + j1, i2 * q + j2] = W[i1 * q + i2, j1 * q + j2]
    return M


class TestKron(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(UNIT_ROOT)

    def test_rearrange_matches_loops_and_inverse(self):
        W = self.rng.standard_normal((64, 64))
        self.assertEqual(rel(rearrange(W), rearrange_loops(W)), 0.0)
        self.assertEqual(rel(unrearrange(rearrange(W)), W), 0.0)
        M = self.rng.standard_normal((64, 64))
        self.assertEqual(rel(rearrange(unrearrange(M)), M), 0.0)
        # isometry
        self.assertLess(abs(np.linalg.norm(rearrange(W)) - np.linalg.norm(W)), TOL_ALG)
        # batched leading axes
        Wb = self.rng.standard_normal((3, 2, 64, 64))
        self.assertEqual(rel(rearrange(Wb)[1, 1], rearrange_loops(Wb[1, 1])), 0.0)

    def test_kron_rank_one_mapping(self):
        A, B = self.rng.standard_normal((8, 8)), self.rng.standard_normal((8, 8))
        R = rearrange(np.kron(A, B))
        self.assertLess(rel(R, np.outer(A.reshape(-1), B.reshape(-1))), TOL_ALG)
        s = np.linalg.svd(R, compute_uv=False)
        self.assertLess(s[1] / s[0], 1e-12)
        # sum of q generic terms has rearranged rank exactly q
        a, b = self.rng.standard_normal((3, 8, 8)), self.rng.standard_normal((3, 8, 8))
        s = np.linalg.svd(rearrange(sum(np.kron(a[j], b[j]) for j in range(3))), compute_uv=False)
        self.assertGreater(s[2] / s[0], 1e-6)
        self.assertLess(s[3] / s[0], 1e-12)

    def test_vectorization_conventions(self):
        A, B, X = (self.rng.standard_normal((8, 8)) for _ in range(3))
        # row-major: (A kron B) vec_r(X) = vec_r(A X B^T)
        self.assertLess(rel(np.kron(A, B) @ X.reshape(-1), (A @ X @ B.T).reshape(-1)), TOL_ALG)
        # column-major (proposal convention): (A kron B) vec_c(X) = vec_c(B X A^T)
        self.assertLess(rel(np.kron(A, B) @ X.reshape(-1, order="F"), (B @ X @ A.T).reshape(-1, order="F")), TOL_ALG)

    def test_dense_from_factors_matches_np_kron(self):
        A, B = self.rng.standard_normal((5, 8, 8)), self.rng.standard_normal((5, 8, 8))
        ref = sum(np.kron(A[j], B[j]) for j in range(5))
        self.assertLess(rel(dense_from_factors_np(A, B), ref), TOL_ALG)
        self.assertLess(rel(dense_from_factors(torch.tensor(A), torch.tensor(B)).numpy(), ref), TOL_ALG)
        A2, B2 = self.rng.standard_normal((2, 3, 4, 8, 8)), self.rng.standard_normal((2, 3, 4, 8, 8))
        K = dense_from_factors_np(A2, B2)
        self.assertLess(rel(K[1, 2], sum(np.kron(A2[1, 2, j], B2[1, 2, j]) for j in range(4))), TOL_ALG)

    def test_factorized_forward_and_backward(self):
        A = torch.tensor(self.rng.standard_normal((4, 8, 8)), requires_grad=True)
        B = torch.tensor(self.rng.standard_normal((4, 8, 8)), requires_grad=True)
        X = torch.tensor(self.rng.standard_normal((50, 64)))
        Yt = torch.tensor(self.rng.standard_normal((50, 64)))
        Kd = sum(torch.kron(A[j], B[j]) for j in range(4))
        f1 = factorized_forward(A, B, X)
        f2 = X @ Kd.T
        self.assertLess(rel(f1.detach().numpy(), f2.detach().numpy()), TOL_ALG)
        l1 = 0.5 * ((Yt - f1) ** 2).sum(1).mean()
        gA1, gB1 = torch.autograd.grad(l1, (A, B))
        l2 = 0.5 * ((Yt - X @ sum(torch.kron(A[j], B[j]) for j in range(4)).T) ** 2).sum(1).mean()
        gA2, gB2 = torch.autograd.grad(l2, (A, B))
        self.assertLess(rel(gA1.numpy(), gA2.numpy()), TOL_ALG)
        self.assertLess(rel(gB1.numpy(), gB2.numpy()), TOL_ALG)

    def test_finite_difference_gradient(self):
        A0 = self.rng.standard_normal((2, 8, 8))
        B0 = self.rng.standard_normal((2, 8, 8))
        X = self.rng.standard_normal((30, 64))
        Y = self.rng.standard_normal((30, 64))

        def loss_np(A, B):
            K = sum(np.kron(A[j], B[j]) for j in range(2))
            return 0.5 * np.mean(np.sum((Y - X @ K.T) ** 2, axis=1))

        A = torch.tensor(A0, requires_grad=True)
        B = torch.tensor(B0, requires_grad=True)
        l = 0.5 * ((torch.tensor(Y) - factorized_forward(A, B, torch.tensor(X))) ** 2).sum(1).mean()
        gA, gB = torch.autograd.grad(l, (A, B))
        idx = [(0, 1, 2), (1, 7, 0), (0, 4, 4), (1, 3, 5)]
        for which, g in (("A", gA), ("B", gB)):
            for ix in idx:
                Ap, Am, Bp, Bm = A0.copy(), A0.copy(), B0.copy(), B0.copy()
                if which == "A":
                    Ap[ix] += FD_H; Am[ix] -= FD_H
                else:
                    Bp[ix] += FD_H; Bm[ix] -= FD_H
                fd = (loss_np(Ap, Bp) - loss_np(Am, Bm)) / (2 * FD_H)
                self.assertLess(abs(fd - g[ix].item()) / max(1.0, abs(fd)), TOL_FD)

    def test_best_kron_approx_eckart_young(self):
        W = self.rng.standard_normal((64, 64))
        for r in (1, 2, 4, 8):
            K = best_kron_approx(W, r)
            s = np.linalg.svd(rearrange(W), compute_uv=False)
            self.assertLess(abs(np.sum((W - K) ** 2) - np.sum(s[r:] ** 2)) / np.sum(W ** 2), TOL_ALG)
            self.assertEqual(int(np.sum(np.linalg.svd(rearrange(K), compute_uv=False) > 1e-9 * s[0])), r)
            # random rank-r Kronecker sums are never better
            a, b = self.rng.standard_normal((r, 8, 8)), self.rng.standard_normal((r, 8, 8))
            self.assertGreaterEqual(np.sum((W - dense_from_factors_np(a, b) * 0.01) ** 2), np.sum((W - K) ** 2))


if __name__ == "__main__":
    unittest.main()
