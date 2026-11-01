"""Synthetic teachers, covariances, data splits and RNG stream layout.

Every random quantity comes from numpy PCG64 seeded by SeedSequence(entropy=root, spawn_key=key).
Keys (all ints):
  teacher      (1, block)
  covariance   (2, block)
  data         (3, block, cov_kind, task, split)   split: 0 pilot, 1 train, 2 val, 3 test
  train init   (4, block, seed_index)              shared by the iso/aniso pair of a block
  early init   (5, block)                          shared by the iso/aniso pair of a block
Streams are therefore independent of generation order and of every other split's size.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from .kron import D, P, Q

STREAM_TEACHER, STREAM_COV, STREAM_DATA, STREAM_INIT, STREAM_EARLY_INIT = 1, 2, 3, 4, 5
SPLITS = {"pilot": 0, "train": 1, "val": 2, "test": 3}
COV_KINDS = ("iso", "aniso")


def rng_for(root: int, key: Tuple[int, ...]) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=root, spawn_key=tuple(int(k) for k in key))))


@dataclass(frozen=True)
class BlockSpec:
    block: int
    cell: int
    replicate: int
    q: int
    delta: float


def block_specs(cfg: dict) -> List[BlockSpec]:
    out = []
    reps = cfg["teacher"]["replicate_blocks_per_cell"]
    for ci, (q, delta) in enumerate(cfg["teacher"]["cells_order"]):
        for r in range(reps):
            out.append(BlockSpec(len(out), ci, r, int(q), float(delta)))
    return out


@dataclass(frozen=True)
class EnvSpec:
    env_id: int
    block: BlockSpec
    cov_kind: str

    @property
    def cov_index(self) -> int:
        return COV_KINDS.index(self.cov_kind)


def env_specs(cfg: dict) -> List[EnvSpec]:
    out = []
    for b in block_specs(cfg):
        for kind in cfg["covariance"]["kinds_order"]:
            out.append(EnvSpec(len(out), b, kind))
    return out


def _kron_rank_component(rng: np.random.Generator, q: int, norm: float) -> np.ndarray:
    a = rng.standard_normal((q, P, P))
    b = rng.standard_normal((q, Q, Q))
    M = sum(np.kron(a[j], b[j]) for j in range(q))
    return M * (norm / np.linalg.norm(M))


@dataclass
class Teacher:
    common: np.ndarray            # (d, d), ||.||_F = component norm
    specific: np.ndarray          # (4, d, d), each ||.||_F = component norm
    residual: np.ndarray          # (4, d, d)
    T: np.ndarray                 # (4, d, d) full teacher = common + delta*specific + residual
    struct_norm: np.ndarray       # (4,)


def make_teacher(cfg: dict, root: int, spec: BlockSpec) -> Teacher:
    n_tasks = cfg["tasks"]["n_tasks"]
    nu = cfg["teacher"]["component_frobenius_norm"]
    ratio = cfg["teacher"]["dense_residual_norm_ratio"]
    rng = rng_for(root, (STREAM_TEACHER, spec.block))
    C = _kron_rank_component(rng, spec.q, nu)
    S = np.stack([_kron_rank_component(rng, spec.q, nu) for _ in range(n_tasks)])
    struct = C[None] + spec.delta * S
    snorm = np.linalg.norm(struct, axis=(1, 2))
    Rraw = rng.standard_normal((n_tasks, D, D))
    R = Rraw * (ratio * snorm / np.linalg.norm(Rraw, axis=(1, 2)))[:, None, None]
    return Teacher(C, S, R, struct + R, snorm)


def aniso_spectrum(cfg: dict) -> np.ndarray:
    kappa = cfg["covariance"]["aniso_condition_number"]
    lam = kappa ** (-np.arange(D) / (D - 1))
    return lam * (cfg["covariance"]["trace"] / lam.sum())


def haar_orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    Z = rng.standard_normal((d, d))
    Qm, Rm = np.linalg.qr(Z)
    return Qm * np.sign(np.diag(Rm))[None, :]


@dataclass
class Covariance:
    Sigma: np.ndarray   # (4, d, d)
    L: np.ndarray       # (4, d, d) with Sigma = L L^T


def make_covariance(cfg: dict, root: int, spec: BlockSpec, kind: str) -> Covariance:
    n_tasks = cfg["tasks"]["n_tasks"]
    if kind == "iso":
        I = np.broadcast_to(np.eye(D), (n_tasks, D, D)).copy()
        return Covariance(I, I.copy())
    rng = rng_for(root, (STREAM_COV, spec.block))
    lam = aniso_spectrum(cfg)
    Qs = np.stack([haar_orthogonal(rng, D) for _ in range(n_tasks)])
    L = Qs * np.sqrt(lam)[None, None, :]
    Sigma = (Qs * lam[None, None, :]) @ Qs.transpose(0, 2, 1)
    return Covariance(Sigma, L)


@dataclass
class Environment:
    """Population objects of one environment. Never handed to a selector."""
    spec: EnvSpec
    root: int
    teacher: Teacher
    cov: Covariance
    noise_var: np.ndarray            # (4,) per-coordinate output noise variance
    signal_energy: np.ndarray        # (4,) E||T x||^2 = tr(T Sigma T^T)
    zero_model_half_loss: float      # sum_t w_t * 0.5 * E||y_t||^2 (evaluation normalizer)
    task_weights: np.ndarray


def make_environment(cfg: dict, root: int, spec: EnvSpec) -> Environment:
    teacher = make_teacher(cfg, root, spec.block)
    cov = make_covariance(cfg, root, spec.block, spec.cov_kind)
    T = teacher.T
    sig = np.einsum("tij,tjk,tik->t", T, cov.Sigma, T)
    frac = cfg["noise"]["output_noise_energy_fraction"]
    noise_var = frac * sig / D
    w = np.asarray(cfg["tasks"]["task_weights"], dtype=float)
    z0 = float(np.sum(w * 0.5 * (sig + D * noise_var)))
    return Environment(spec, root, teacher, cov, noise_var, sig, z0, w)


def sample_split(cfg: dict, env: Environment, split: str, n: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """Return X, Y of shape (4, n, d), row-sample convention y = T x + eps."""
    if n is None:
        n = cfg["data_sizes"][split]
    n_tasks = cfg["tasks"]["n_tasks"]
    Xs, Ys = [], []
    for t in range(n_tasks):
        rng = rng_for(env.root, (STREAM_DATA, env.spec.block.block, env.spec.cov_index, t, SPLITS[split]))
        Z = rng.standard_normal((n, D))
        eps = rng.standard_normal((n, D)) * np.sqrt(env.noise_var[t])
        X = Z @ env.cov.L[t].T
        Y = X @ env.teacher.T[t].T + eps
        Xs.append(X)
        Ys.append(Y)
    return np.stack(Xs), np.stack(Ys)


@dataclass(frozen=True)
class PilotView:
    """The ONLY object selectors receive: raw pilot arrays and the fixed task weights."""
    X: np.ndarray  # (4, n_pilot, d)
    Y: np.ndarray  # (4, n_pilot, d)
    task_weights: np.ndarray

    @staticmethod
    def from_arrays(X: np.ndarray, Y: np.ndarray, w) -> "PilotView":
        X = np.array(X, dtype=np.float64, copy=True)
        Y = np.array(Y, dtype=np.float64, copy=True)
        w = np.array(w, dtype=np.float64, copy=True)
        for a in (X, Y, w):
            a.setflags(write=False)
        return PilotView(X, Y, w)


def moments(X: np.ndarray, Y: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-task sufficient statistics of the half squared loss.

    Sxx = X^T X / n, Syx = Y^T X / n (= negative gradient at K = 0), syy = sum ||y||^2 / n.
    """
    n = X.shape[1]
    return {
        "Sxx": np.einsum("tni,tnj->tij", X, X) / n,
        "Syx": np.einsum("tni,tnj->tij", Y, X) / n,
        "syy": np.einsum("tni,tni->t", Y, Y) / n,
        "n": n,
    }
