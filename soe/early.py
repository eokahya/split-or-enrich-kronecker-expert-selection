"""Early-training / validation selector on the pilot (32 fit / 32 validation per task).

All nine candidates are trained round-robin: one round = one Adam step for every candidate
(fused into one optimizer; Adam is elementwise so this equals nine separate steps), followed
by a validation evaluation of all candidates. Candidates are therefore always compared at an
equal number of completed steps (complete sweeps, no candidate-order bias).

Budgeted mode (budget_s): used both for the PRIMARY early baseline (common 1.0 s ceiling) and the
SECONDARY naive-time-matched cost diagnostic. The clock starts when the selector receives the raw
pilot arrays (tensor conversion, moments, initialisation and every validation are charged) and is
read after each validation (after device synchronisation). The selection uses the LAST round whose
completion time is within budget; a round that finishes after the budget is discarded but its time is
included in the reported consumed wall time. If not even one round + evaluation fits, the result is
NOT_MEASURABLE (never a score).
Fixed mode (fixed_rounds): exactly k rounds then one validation; used for deterministic replay.
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np
import torch

from .candidates import CANDIDATES
from .envgen import PilotView, moments
from .train import FusedCandidates, make_adam, torch_moments


def _sync(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()


def default_clock() -> float:
    return time.perf_counter()


def early_select(pilot: PilotView, init_master: np.ndarray, cfg: dict, *, budget_s: Optional[float] = None,
                 fixed_rounds: Optional[int] = None, clock: Callable[[], float] = default_clock,
                 max_rounds: int = 1_000_000) -> Dict:
    """Exactly one of budget_s or fixed_rounds must be given."""
    assert (budget_s is None) != (fixed_rounds is None)
    ec = cfg["early_selector"]
    dtype = {"float64": torch.float64, "float32": torch.float32}[ec["dtype"]]
    device = ec["device"]
    tcfg = dict(cfg["training"])
    tcfg["lr"] = ec["lr"]
    a, b = ec["pilot_train"], ec["pilot_val"]

    t0 = clock()
    assert a + b == pilot.X.shape[1]
    mtr = torch_moments(moments(pilot.X[:, :a], pilot.Y[:, :a]), dtype, device)
    mva = torch_moments(moments(pilot.X[:, a:], pilot.Y[:, a:]), dtype, device)
    model = FusedCandidates(CANDIDATES, init_master[None], pilot.task_weights, dtype, device)
    opt = make_adam(model.params(), tcfg)
    _sync(device)
    setup_s = clock() - t0

    snapshots = []  # (round, elapsed_s, val_losses)
    overrun_s = None
    first_round_s = None
    rounds = 0
    limit = fixed_rounds if fixed_rounds is not None else max_rounds
    while rounds < limit:
        opt.zero_grad(set_to_none=True)
        model.weighted_losses(mtr).sum().backward()
        opt.step()
        rounds += 1
        if fixed_rounds is not None and rounds < fixed_rounds:
            continue
        with torch.no_grad():
            val = model.weighted_losses(mva)[:, 0].cpu().double().numpy()
        _sync(device)
        el = clock() - t0
        if first_round_s is None and fixed_rounds is None:
            first_round_s = el - setup_s
        if budget_s is not None and el > budget_s:
            overrun_s = el - budget_s
            break
        snapshots.append((rounds, el, val))
    if snapshots:
        k, el_sel, val = snapshots[-1]
        selected = int(np.argmin(val))
    wall_s = clock() - t0  # consumed time including any discarded overrun round and the final choice

    base = {"setup_s": setup_s, "first_round_plus_eval_s": first_round_s, "device": str(device),
            "dtype": ec["dtype"], "budget_s": budget_s, "fixed_rounds": fixed_rounds, "wall_s": wall_s,
            "rounds_attempted": rounds, "overrun_s": overrun_s,
            "min_granularity_s": None if first_round_s is None else setup_s + first_round_s}
    if not snapshots:
        return {**base, "status": "NOT_MEASURABLE", "selected": None, "rounds_in_budget": 0,
                "selection_elapsed_s": None, "val_losses": None}
    return {**base, "status": "MEASURED", "selected": selected, "rounds_in_budget": k,
            "selection_elapsed_s": el_sel, "val_losses": [float(v) for v in val]}
