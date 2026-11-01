"""Moment-based full-batch training of Kronecker expert candidates.

Synthetic backend: the half squared loss of a linear map depends on data only through
Sxx = X^T X / n, Syx = Y^T X / n, syy = sum||y||^2/n:
    L_t(K) = 1/2 ( tr(K Sxx K^T) - 2 <K, Syx> + syy ).
Factors are trained, the dense 64x64 K is materialised per step (documented synthetic shortcut,
no hardware-efficiency claim). Several candidates and init seeds are trained in ONE Adam
optimizer whose loss is the plain sum of independent problem losses; Adam is elementwise,
so this is mathematically identical to separate runs (verified in tests).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch

from .candidates import Candidate
from .kron import D, P, Q, dense_from_factors, dense_from_factors_np, factorized_forward


def init_factors(master: np.ndarray, cand: Candidate):
    """master: (4, 8, P, P) standard normals for one init seed. Returns numpy A, B of shape (E, r, P, P)."""
    E, r = cand.n_experts, cand.rank
    A = master[:E, :r] / math.sqrt(P * r)
    B = np.zeros((E, r, Q, Q))
    return A, B


def torch_moments(mom: Dict[str, np.ndarray], dtype, device) -> Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(mom[k]), dtype=dtype, device=device) for k in ("Sxx", "Syx", "syy")}


class FusedCandidates:
    """Factors for a list of candidates x S init seeds.

    Candidates with the same (E, r) shape are stacked into one leaf tensor of shape
    (S, n_shape, E, r, P, P) (the seven E2 partitions share one tensor). Adam is elementwise and the
    loss is a plain sum over candidates and seeds, so this is identical to separate runs; it only
    removes per-candidate Python/autograd overhead.
    """

    def __init__(self, candidates: Sequence[Candidate], masters: np.ndarray, task_weights, dtype=torch.float64, device="cpu"):
        self.candidates = list(candidates)
        self.S = masters.shape[0]
        self.dtype, self.device = dtype, device
        self.w = torch.as_tensor(np.array(task_weights, dtype=np.float64), dtype=dtype, device=device)
        shapes = []
        for c in self.candidates:
            if (c.n_experts, c.rank) not in shapes:
                shapes.append((c.n_experts, c.rank))
        self.GA, self.GB, self.gassign, self.gmembers, self.loc = [], [], [], [], {}
        for k, (E, r) in enumerate(shapes):
            members = [ci for ci, c in enumerate(self.candidates) if (c.n_experts, c.rank) == (E, r)]
            A = np.stack([np.stack([init_factors(masters[s], self.candidates[ci])[0] for ci in members])
                          for s in range(self.S)])
            self.GA.append(torch.tensor(A, dtype=dtype, device=device, requires_grad=True))
            self.GB.append(torch.zeros(A.shape, dtype=dtype, device=device, requires_grad=True))
            self.gassign.append(torch.as_tensor([self.candidates[ci].assignment for ci in members], device=device))
            self.gmembers.append(members)
            for j, ci in enumerate(members):
                self.loc[ci] = (k, j)
        order = [ci for m in self.gmembers for ci in m]
        self.inv_order = torch.as_tensor(np.argsort(order), device=device)

    def params(self) -> List[torch.Tensor]:
        return [p for pair in zip(self.GA, self.GB) for p in pair]

    def A_of(self, ci: int) -> torch.Tensor:
        k, j = self.loc[ci]
        return self.GA[k][:, j]  # (S, E, r, P, P)

    def B_of(self, ci: int) -> torch.Tensor:
        k, j = self.loc[ci]
        return self.GB[k][:, j]

    def grads_of(self, ci: int):
        k, j = self.loc[ci]
        return self.GA[k].grad[:, j], self.GB[k].grad[:, j]

    def task_losses(self, mom: Dict[str, torch.Tensor]) -> torch.Tensor:
        """(n_cand, S, n_tasks) half-loss per task."""
        out = []
        for k in range(len(self.GA)):
            K = dense_from_factors(self.GA[k], self.GB[k])  # (S, n_k, E, d, d)
            n_k = K.shape[1]
            Kt = K[:, torch.arange(n_k, device=self.device)[:, None], self.gassign[k]]  # (S, n_k, T, d, d)
            quad = torch.sum((Kt @ mom["Sxx"]) * Kt, dim=(-2, -1))
            lin = torch.sum(Kt * mom["Syx"], dim=(-2, -1))
            out.append((0.5 * (quad - 2.0 * lin + mom["syy"])).transpose(0, 1))  # (n_k, S, T)
        return torch.cat(out)[self.inv_order]

    def weighted_losses(self, mom) -> torch.Tensor:
        return self.task_losses(mom) @ self.w  # (n_cand, S)


def make_adam(params, cfg_train: dict):
    return torch.optim.Adam(params, lr=cfg_train["lr"], betas=tuple(cfg_train["betas"]),
                            eps=cfg_train["eps"], weight_decay=cfg_train["weight_decay"])


def train_fused(model: FusedCandidates, mom_t: Dict[str, torch.Tensor], cfg_train: dict, steps: int) -> np.ndarray:
    """Full-batch Adam. Returns trajectory (steps+1, n_cand, S): losses at theta_0..theta_steps."""
    opt = make_adam(model.params(), cfg_train)
    traj = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        L = model.weighted_losses(mom_t)
        traj.append(L.detach())
        L.sum().backward()
        opt.step()
    with torch.no_grad():
        traj.append(model.weighted_losses(mom_t))
    return torch.stack(traj).cpu().numpy()


def export_factors(model: FusedCandidates):
    """List over candidates of (A, B, K) numpy arrays with leading seed axis."""
    out = []
    with torch.no_grad():
        for ci in range(len(model.candidates)):
            A = model.A_of(ci).detach().cpu().double().numpy()
            B = model.B_of(ci).detach().cpu().double().numpy()
            out.append((A, B, dense_from_factors_np(A, B)))
    return out


# --- numpy evaluation (float64) --------------------------------------------------------

def moment_loss(Ks: np.ndarray, assignment, mom: Dict[str, np.ndarray], w) -> float:
    """Ks: (E, d, d) dense experts. Weighted half loss from moments."""
    K = Ks[list(assignment)]
    quad = np.einsum("tij,tjk,tik->t", K, mom["Sxx"], K)
    lin = np.sum(K * mom["Syx"], axis=(1, 2))
    return float(np.sum(np.asarray(w) * 0.5 * (quad - 2 * lin + mom["syy"])))


def sample_loss_factorized(A: np.ndarray, B: np.ndarray, assignment, X: np.ndarray, Y: np.ndarray, w) -> float:
    """Weighted half loss computed sample by sample with the factorized forward (no dense K)."""
    tot = 0.0
    At, Bt = torch.as_tensor(A, dtype=torch.float64), torch.as_tensor(B, dtype=torch.float64)
    for t, e in enumerate(assignment):
        pred = factorized_forward(At[e], Bt[e], torch.as_tensor(X[t], dtype=torch.float64)).numpy()
        tot += w[t] * 0.5 * float(np.mean(np.sum((Y[t] - pred) ** 2, axis=1)))
    return tot


def population_risk(Ks: np.ndarray, assignment, T: np.ndarray, Sigma: np.ndarray, noise_var: np.ndarray, w) -> float:
    """Exact expected weighted half loss: sum_t w_t/2 [tr((T-K) Sigma (T-K)^T) + d sigma_t^2]."""
    Dl = T - Ks[list(assignment)]
    quad = np.einsum("tij,tjk,tik->t", Dl, Sigma, Dl)
    return float(np.sum(np.asarray(w) * 0.5 * (quad + D * noise_var)))
