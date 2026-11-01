"""Kronecker conventions, rearrangement and factorized operators.

Conventions (fixed for the whole package):
  * d = p * q with p = q = 8; vectors are flattened ROW-MAJOR: x[i1*q + i2] = X[i1, i2].
  * (A kron B)[i1*q+i2, j1*q+j2] = A[i1, j1] * B[i2, j2]  (numpy.kron convention).
  * With row-major flattening, (A kron B) vec_r(X) = vec_r(A X B^T).
  * Rearrangement R(W)[i1*p+j1, i2*q+j2] = W[i1*q+i2, j1*q+j2], so
    R(A kron B) = vec_r(A) vec_r(B)^T (rank one). R is a linear isometry and,
    for p == q, an involution.
"""
from __future__ import annotations

import numpy as np
import torch

P = 8
Q = 8
D = P * Q


def rearrange(W: np.ndarray, p: int = P, q: int = Q) -> np.ndarray:
    """R(W) for W of shape (..., p*q, p*q)."""
    lead = W.shape[:-2]
    W4 = W.reshape(*lead, p, q, p, q)  # (i1, i2, j1, j2)
    n = len(lead)
    perm = tuple(range(n)) + (n + 0, n + 2, n + 1, n + 3)  # (i1, j1, i2, j2)
    return W4.transpose(perm).reshape(*lead, p * p, q * q)


def unrearrange(M: np.ndarray, p: int = P, q: int = Q) -> np.ndarray:
    """Inverse of rearrange: M of shape (..., p*p, q*q) -> (..., p*q, p*q)."""
    lead = M.shape[:-2]
    M4 = M.reshape(*lead, p, p, q, q)  # (i1, j1, i2, j2)
    n = len(lead)
    perm = tuple(range(n)) + (n + 0, n + 2, n + 1, n + 3)  # (i1, i2, j1, j2)
    return M4.transpose(perm).reshape(*lead, p * q, p * q)


def best_kron_approx(W: np.ndarray, r: int) -> np.ndarray:
    """Best Frobenius approximation of W by a sum of r Kronecker terms (Van Loan-Pitsianis)."""
    U, s, Vt = np.linalg.svd(rearrange(W), full_matrices=False)
    M = (U[:, :r] * s[:r]) @ Vt[:r]
    return unrearrange(M)


def dense_from_factors_np(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A, B: (..., r, p, p) and (..., r, q, q) -> sum_j kron(A_j, B_j) of shape (..., d, d)."""
    lead = A.shape[:-3]
    p, q = A.shape[-1], B.shape[-1]
    K = np.einsum("...jab,...jcd->...acbd", A, B)
    return K.reshape(*lead, p * q, p * q)


def dense_from_factors(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Torch version of dense_from_factors_np (differentiable)."""
    lead = A.shape[:-3]
    p, q = A.shape[-1], B.shape[-1]
    K = torch.einsum("...jab,...jcd->...acbd", A, B)
    return K.reshape(*lead, p * q, p * q)


def factorized_forward(A: torch.Tensor, B: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """Apply sum_j kron(A_j, B_j) to rows of X (n, d) without forming the dense matrix.

    y_n = vec_r( sum_j A_j X_n B_j^T ) where X_n = x_n reshaped row-major to (p, q).
    A: (r, p, p), B: (r, q, q). Returns (n, d).
    """
    p, q = A.shape[-1], B.shape[-1]
    Xr = X.reshape(X.shape[0], p, q)
    Y = torch.einsum("jab,nbc,jdc->nad", A, Xr, B)
    return Y.reshape(X.shape[0], p * q)
