"""Pilot-only analytic selectors.

Every public selector takes a PilotView (raw pilot X, Y and task weights) and nothing else.

Proxy operators: P_t = Y_t^T X_t / n = -grad_K L_t(0) for L_t(K) = (1/2n) sum ||y - K x||^2.
Under isotropic inputs the population P_t is T_t and, for candidate c with groups e and
ranks r_e,
    min_K sum_t w_t L_t(K) - noise = 1/2 sum_e [ sum_{t in e} w_t ||P_t - Pbar_e||^2
                                                + pi_e sum_{j > r_e} s_j^2(R(Pbar_e)) ]
                                   = 1/2 [ sum_t w_t ||P_t||^2 - sum_e pi_e sum_{j <= r_e} s_j^2(R(Pbar_e)) ],
with pi_e = sum_{t in e} w_t, Pbar_e = sum_{t in e} w_t P_t / pi_e. Scores keep the factor 1/2.
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .candidates import CANDIDATES, distinct_groups
from .envgen import PilotView
from .kron import best_kron_approx, rearrange


def proxy_operators(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.einsum("tni,tnj->tij", Y, X) / X.shape[1]


def argmin_first(scores: Sequence[float]) -> int:
    return int(np.argmin(np.asarray(scores, dtype=np.float64)))


# --- score implementations -------------------------------------------------------------

def naive_scores_decomposed(P: np.ndarray, w: np.ndarray, candidates=CANDIDATES) -> np.ndarray:
    """Explicit within-group variance + rearrangement spectral tail (reference implementation)."""
    out = []
    for c in candidates:
        total = 0.0
        for g in c.groups():
            wg = w[list(g)]
            pi = wg.sum()
            Pbar = np.tensordot(wg, P[list(g)], axes=1) / pi
            var = sum(w[t] * np.sum((P[t] - Pbar) ** 2) for t in g)
            s = np.linalg.svd(rearrange(Pbar), compute_uv=False)
            tail = np.sum(s[c.rank:] ** 2)
            total += var + pi * tail
        out.append(0.5 * total)
    return np.asarray(out)


def _group_spectra(P: np.ndarray, w: np.ndarray, groups):
    pis = np.array([w[list(g)].sum() for g in groups])
    means = np.stack([np.tensordot(w[list(g)], P[list(g)], axes=1) / pis[i] for i, g in enumerate(groups)])
    svals = np.linalg.svd(rearrange(means), compute_uv=False)  # batched
    return pis, means, svals


def naive_scores_captured(P: np.ndarray, w: np.ndarray, candidates=CANDIDATES) -> np.ndarray:
    """Constant minus weighted captured spectral energy, one batched SVD over distinct groups."""
    groups = distinct_groups()
    gidx = {g: i for i, g in enumerate(groups)}
    pis, _, svals = _group_spectra(P, w, groups)
    cum = np.cumsum(svals ** 2, axis=1)
    const = float(np.sum(w * np.sum(P ** 2, axis=(1, 2))))
    out = []
    for c in candidates:
        cap = sum(pis[gidx[g]] * cum[gidx[g], c.rank - 1] for g in c.groups())
        out.append(0.5 * (const - cap))
    return np.asarray(out)


def variance_only_scores(P, w, candidates=CANDIDATES):
    out = []
    for c in candidates:
        tot = 0.0
        for g in c.groups():
            wg = w[list(g)]
            Pbar = np.tensordot(wg, P[list(g)], axes=1) / wg.sum()
            tot += sum(w[t] * np.sum((P[t] - Pbar) ** 2) for t in g)
        out.append(0.5 * tot)
    return np.asarray(out)


def tail_only_scores(P, w, candidates=CANDIDATES):
    out = []
    for c in candidates:
        tot = 0.0
        for g in c.groups():
            wg = w[list(g)]
            pi = wg.sum()
            Pbar = np.tensordot(wg, P[list(g)], axes=1) / pi
            s = np.linalg.svd(rearrange(Pbar), compute_uv=False)
            tot += pi * np.sum(s[c.rank:] ** 2)
        out.append(0.5 * tot)
    return np.asarray(out)


def crossfit_scores(P1: np.ndarray, P2: np.ndarray, w: np.ndarray, candidates=CANDIDATES) -> np.ndarray:
    """Symmetrized held-out risk of the fitted constrained update, up to an additive constant.

    For fit half a and evaluation half b: sum_e sum_{t in e} w_t (1/2)(||K_e^a||^2 - 2 <P_t^b, K_e^a>),
    K_e^a = best rank-r_e Kronecker approximation of Pbar_e^a. Average of (a,b) = (1,2), (2,1).
    """
    out = []
    for c in candidates:
        tot = 0.0
        for Pa, Pb in ((P1, P2), (P2, P1)):
            for g in c.groups():
                wg = w[list(g)]
                pi = wg.sum()
                K = best_kron_approx(np.tensordot(wg, Pa[list(g)], axes=1) / pi, c.rank)
                Pb_bar = np.tensordot(wg, Pb[list(g)], axes=1) / pi
                tot += 0.5 * pi * (np.sum(K * K) - 2.0 * np.sum(Pb_bar * K))
        out.append(0.5 * tot)
    return np.asarray(out)


def covariance_corrected_operators(X: np.ndarray, Y: np.ndarray, ridge_fraction: float) -> np.ndarray:
    """T_hat_t = P_t (C_t + lam_t I)^-1 with C_t = X^T X / n and lam_t = ridge_fraction * tr(C_t) / d."""
    n, d = X.shape[1], X.shape[2]
    P = proxy_operators(X, Y)
    C = np.einsum("tni,tnj->tij", X, X) / n
    lam = ridge_fraction * np.trace(C, axis1=1, axis2=2) / d
    M = C + lam[:, None, None] * np.eye(d)[None]
    # P M^-1 = (M^-1 P^T)^T, M symmetric
    return np.linalg.solve(M, P.transpose(0, 2, 1)).transpose(0, 2, 1)


# --- public selectors ------------------------------------------------------------------

def _result(scores: np.ndarray, **extra) -> Dict:
    return {"scores": [float(s) for s in scores], "selected": argmin_first(scores), **extra}


def select_naive(pilot: PilotView) -> Dict:
    """PRIMARY: naive full-pilot score."""
    P = proxy_operators(pilot.X, pilot.Y)
    return _result(naive_scores_captured(P, pilot.task_weights))


def select_naive_reference(pilot: PilotView) -> Dict:
    P = proxy_operators(pilot.X, pilot.Y)
    return _result(naive_scores_decomposed(P, pilot.task_weights))


def select_crossfit(pilot: PilotView, halves=(32, 32)) -> Dict:
    a, b = halves
    assert a + b == pilot.X.shape[1]
    P1 = proxy_operators(pilot.X[:, :a], pilot.Y[:, :a])
    P2 = proxy_operators(pilot.X[:, a:], pilot.Y[:, a:])
    return _result(crossfit_scores(P1, P2, pilot.task_weights))


def select_covariance(pilot: PilotView, ridge_fraction: float = 0.1) -> Dict:
    That = covariance_corrected_operators(pilot.X, pilot.Y, ridge_fraction)
    return _result(naive_scores_captured(That, pilot.task_weights))


def select_variance_only(pilot: PilotView) -> Dict:
    return _result(variance_only_scores(proxy_operators(pilot.X, pilot.Y), pilot.task_weights))


def select_tail_only(pilot: PilotView) -> Dict:
    return _result(tail_only_scores(proxy_operators(pilot.X, pilot.Y), pilot.task_weights))


def constrained_optimum(P: np.ndarray, w: np.ndarray, candidate) -> List[np.ndarray]:
    """Per-expert dense optimum under the isotropic objective: best rank-r Kronecker fit of each group mean."""
    out = []
    for g in candidate.groups():
        wg = w[list(g)]
        out.append(best_kron_approx(np.tensordot(wg, P[list(g)], axes=1) / wg.sum(), candidate.rank))
    return out
