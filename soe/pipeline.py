"""End-to-end experiment pipeline (selection phase -> lock -> training/evaluation -> analysis).

Phase order is the isolation mechanism: for every environment only the pilot split is sampled
before ALL selections are written and hashed; train/val/test are sampled afterwards. Selectors
receive a read-only PilotView. Population objects (teacher, covariance, noise) are used only for
data generation and post-lock evaluation diagnostics.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import time
import unittest
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from . import evaluate as ev
from .candidates import CANDIDATES
from .early import early_select
from .envgen import (STREAM_EARLY_INIT, STREAM_INIT, PilotView, env_specs, make_environment, moments, rng_for,
                     sample_split)
from .provenance import (create_new_dir, environment_versions, output_manifest, save_npz_exclusive, sha256_file,
                         verify_freeze, write_bytes_exclusive, write_json_exclusive)
from .selectors import (naive_scores_captured, proxy_operators, select_covariance, select_crossfit, select_naive,
                        select_naive_reference, select_tail_only, select_variance_only)
from .train import (FusedCandidates, export_factors, moment_loss, population_risk, sample_loss_factorized,
                    torch_moments, train_fused)

PKG_ROOT = Path(__file__).resolve().parent.parent
SELECTOR_METHODS = ("naive", "crossfit", "covariance", "variance_only", "tail_only",
                    "early_ceiling", "early_time_matched")


def run_unit_tests() -> dict:
    suite = unittest.TestLoader().discover(str(PKG_ROOT / "tests"), top_level_dir=str(PKG_ROOT))
    buf = io.StringIO()
    res = unittest.TextTestRunner(stream=buf, verbosity=1).run(suite)
    return {"ran": res.testsRun, "failures": len(res.failures), "errors": len(res.errors),
            "skipped": len(res.skipped), "ok": res.wasSuccessful(), "output_tail": buf.getvalue()[-4000:]}


def init_masters(root: int, block: int, n_seeds: int) -> np.ndarray:
    return np.stack([rng_for(root, (STREAM_INIT, block, s)).standard_normal((4, 8, 8, 8)) for s in range(n_seeds)])


def early_master(root: int, block: int) -> np.ndarray:
    return rng_for(root, (STREAM_EARLY_INIT, block)).standard_normal((4, 8, 8, 8))


def configure_runtime(cfg: dict) -> dict:
    torch.set_num_threads(cfg["runtime"]["torch_threads"])
    return {"torch_threads": torch.get_num_threads(),
            "blas_env": {k: os.environ.get(k) for k in cfg["runtime"]["blas_threads_env"]}}


def warmup(cfg: dict) -> None:
    """Untimed warmup on an unrelated environment (warmup seed root)."""
    root = cfg["seeds"]["warmup_root"]
    env = make_environment(cfg, root, env_specs(cfg)[1])
    X, Y = sample_split(cfg, env, "pilot")
    pv = PilotView.from_arrays(X, Y, env.task_weights)
    for _ in range(3):
        select_naive(pv)
        select_crossfit(pv, tuple(cfg["selectors"]["crossfit_halves"]))
        select_covariance(pv, cfg["selectors"]["covariance_ridge_fraction"])
    early_select(pv, early_master(root, 0), cfg, fixed_rounds=5)
    early_select(pv, early_master(root, 0), cfg, budget_s=0.05)
    model = FusedCandidates(CANDIDATES, init_masters(root, 0, 2), env.task_weights)
    train_fused(model, torch_moments(moments(X, Y), torch.float64, "cpu"), cfg["training"], 5)


def _timed(fn, reps):
    times, results = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        times.append(time.perf_counter() - t0)
        results.append(r)
    assert all(r == results[0] for r in results), "analytic selector not deterministic"
    return results[0], times


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(np.asarray(a))).astype(float)
    rb = np.argsort(np.argsort(np.asarray(b))).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def half_pilot_reliability(pilot: PilotView, halves) -> dict:
    """Descriptive: naive scores on each 32-row half vs each other and vs the full 64-row score."""
    a = halves[0]
    w = pilot.task_weights
    full = naive_scores_captured(proxy_operators(pilot.X, pilot.Y), w)
    hA = naive_scores_captured(proxy_operators(pilot.X[:, :a], pilot.Y[:, :a]), w)
    hB = naive_scores_captured(proxy_operators(pilot.X[:, a:], pilot.Y[:, a:]), w)
    return {"argmin_full": int(np.argmin(full)), "argmin_half_a": int(np.argmin(hA)), "argmin_half_b": int(np.argmin(hB)),
            "spearman_half_a_half_b": _spearman(hA, hB), "spearman_half_a_full": _spearman(hA, full),
            "spearman_half_b_full": _spearman(hB, full)}


def run_selectors(pilot: PilotView, cfg: dict, emaster: np.ndarray) -> dict:
    sc, ec = cfg["selectors"], cfg["early_selector"]
    reps = ec["analytic_timing_repeats"]
    halves = tuple(sc["crossfit_halves"])
    naive, t_naive = _timed(lambda: select_naive(pilot), reps)
    cross, t_cross = _timed(lambda: select_crossfit(pilot, halves), reps)
    cov, t_cov = _timed(lambda: select_covariance(pilot, sc["covariance_ridge_fraction"]), reps)
    var_only, t_var = _timed(lambda: select_variance_only(pilot), 1)
    tail_only, t_tail = _timed(lambda: select_tail_only(pilot), 1)
    naive_median = float(np.median(t_naive))
    ref = select_naive_reference(pilot)
    two_impl = max(abs(a - b) / max(1.0, abs(b)) for a, b in zip(naive["scores"], ref["scores"]))
    out = {
        "naive": naive, "crossfit": cross, "covariance": cov, "variance_only": var_only, "tail_only": tail_only,
        "naive_reference_scores": ref["scores"], "naive_two_impl_max_rel_diff": two_impl,
        "timing_s": {"naive": t_naive, "crossfit": t_cross, "covariance": t_cov, "variance_only": t_var,
                     "tail_only": t_tail},
        "naive_median_s": naive_median,
        "half_pilot_reliability": half_pilot_reliability(pilot, halves),
    }
    # PRIMARY early baseline: common ALLOTTED ceiling (not equal consumed time).
    out["early_ceiling"] = early_select(pilot, emaster, cfg, budget_s=ec["primary_common_ceiling_s"])
    # SECONDARY cost diagnostic: budget = measured naive median wall time.
    out["early_time_matched"] = early_select(pilot, emaster, cfg, budget_s=naive_median)
    e = out["early_ceiling"]
    if e["status"] == "MEASURED":  # deterministic replay with the recorded round count
        rep = early_select(pilot, emaster, cfg, fixed_rounds=e["rounds_in_budget"])
        out["early_ceiling_replay"] = {"selected_equal": rep["selected"] == e["selected"],
                                       "max_abs_val_diff": float(np.max(np.abs(np.subtract(rep["val_losses"], e["val_losses"]))))}
    return out


def _selected_map(sel: dict) -> Dict[str, Optional[int]]:
    return {m: sel[m]["selected"] for m in SELECTOR_METHODS}


def train_and_evaluate(cfg: dict, env, root: int, out_dir: Path) -> dict:
    tc = cfg["training"]
    w = env.task_weights
    block = env.spec.block.block
    data = {s: sample_split(cfg, env, s) for s in ("train", "val", "test")}
    mom = {s: moments(*data[s]) for s in data}
    model = FusedCandidates(CANDIDATES, init_masters(root, block, tc["n_init_seeds"]), w, torch.float64, tc["device"])
    t0 = time.perf_counter()
    traj = train_fused(model, torch_moments(mom["train"], torch.float64, tc["device"]), tc, tc["steps"])
    train_s = time.perf_counter() - t0
    exp = export_factors(model)
    Z = env.zero_model_half_loss
    win = tc["convergence_window_steps"]
    noise_floor = float(np.sum(w * 0.5 * 64 * env.noise_var))
    # exact class optimum exists in closed form only for isotropic inputs
    class_opt = naive_scores_captured(env.teacher.T, w) + noise_floor if env.spec.cov_kind == "iso" else None
    rows, cand_loss, cand_pop, max_ms, max_tr = [], [], [], 0.0, 0.0
    arrays = {"T": env.teacher.T, "Sigma": env.cov.Sigma, "noise_var": env.noise_var, "traj": traj}
    for s_name in ("train", "val", "test"):
        for k in ("Sxx", "Syx", "syy"):
            arrays[f"{s_name}_{k}"] = mom[s_name][k]
    for s in range(tc["n_init_seeds"]):
        lrow, prow = [], []
        for ci, c in enumerate(CANDIDATES):
            A, B, K = exp[ci]
            tr_np = moment_loss(K[s], c.assignment, mom["train"], w)
            te_m = moment_loss(K[s], c.assignment, mom["test"], w)
            te_s = sample_loss_factorized(A[s], B[s], c.assignment, *data["test"], w)
            pop = population_risk(K[s], c.assignment, env.teacher.T, env.cov.Sigma, env.noise_var, w)
            max_ms = max(max_ms, abs(te_m - te_s) / Z)
            max_tr = max(max_tr, abs(tr_np - float(traj[-1, ci, s])) / Z)
            rows.append({
                "env_id": env.spec.env_id, "candidate": ci, "label": c.label, "seed": s,
                "train_final": tr_np, "val": moment_loss(K[s], c.assignment, mom["val"], w),
                "test_moment": te_m, "test_sample": te_s, "population_risk": pop,
                "conv_delta_norm": float((traj[-1 - win, ci, s] - traj[-1, ci, s]) / Z),
                "iso_population_gap_over_Z": None if class_opt is None else float((pop - class_opt[ci]) / Z),
                "Z": Z,
            })
            lrow.append(te_m)
            prow.append(pop)
        cand_loss.append(lrow)
        cand_pop.append(prow)
    for ci in range(len(CANDIDATES)):
        A, B, K = exp[ci]
        arrays[f"A_{ci}"], arrays[f"B_{ci}"], arrays[f"K_{ci}"] = A, B, K
    npz_hash = save_npz_exclusive(out_dir / "envs" / f"env{env.spec.env_id:02d}_train_eval.npz", **arrays)
    oracle_P = env.teacher.T @ env.cov.Sigma  # population negative gradient; DIAGNOSTIC ONLY (post-lock)
    oracle = naive_scores_captured(oracle_P, w)
    return {
        "rows": rows, "cand_loss": cand_loss, "cand_pop": cand_pop, "Z": Z, "train_s": train_s,
        "max_moment_vs_sample_norm": max_ms, "max_traj_vs_numpy_norm": max_tr,
        "finite": bool(np.all(np.isfinite(traj)) and all(math.isfinite(r[k]) for r in rows for k in
                                                          ("train_final", "val", "test_moment", "test_sample", "population_risk"))),
        "oracle_population_gradient_selected": int(np.argmin(oracle)),
        "npz_sha256": npz_hash,
    }


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    buf = io.StringIO()
    wr = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
    wr.writeheader()
    wr.writerows(rows)
    write_bytes_exclusive(path, buf.getvalue().encode())


def full_env_ids(cfg: dict) -> List[int]:
    return list(range(len(env_specs(cfg))))


def _check_scientific(cfg, root, env_ids, freeze_manifest, run_tests) -> None:
    if root != cfg["seeds"]["scientific_root"] or freeze_manifest is None:
        raise RuntimeError("scientific mode requires the scientific root and a verified freeze manifest")
    if list(env_ids) != full_env_ids(cfg):
        raise RuntimeError("scientific mode requires exactly all environment ids of the frozen grid")
    with open(PKG_ROOT / "config.json") as f:
        if json.load(f) != cfg:
            raise RuntimeError("scientific mode requires the unmodified on-disk config.json")
    if freeze_manifest["files"].get("config.json") != sha256_file(PKG_ROOT / "config.json"):
        raise RuntimeError("config.json does not match the freeze manifest")
    if not run_tests:
        raise RuntimeError("scientific mode must run the unit tests")


def run(cfg: dict, *, root: int, env_ids: List[int], out_dir: Path, mode: str, run_tests: bool,
        freeze_manifest: Optional[dict] = None, analyze: bool = True, log: Callable[[str], None] = print) -> dict:
    if mode == "scientific":
        _check_scientific(cfg, root, env_ids, freeze_manifest, run_tests)
        expected_envs = len(full_env_ids(cfg))
    elif mode == "technical":
        if root == cfg["seeds"]["scientific_root"]:
            raise RuntimeError("technical mode may not use the scientific seed root")
        expected_envs = len(env_ids)
    else:
        raise ValueError(mode)
    t_start = time.perf_counter()
    cap = cfg["gates"]["runtime_cap_seconds"]
    out = create_new_dir(out_dir)
    (out / "pilot").mkdir()
    (out / "envs").mkdir()
    rt = configure_runtime(cfg)
    header = {"mode": mode, "root": root, "env_ids": env_ids, "versions": environment_versions(), "runtime": rt,
              "freeze_manifest": freeze_manifest, "config": cfg, "started_unix": time.time()}
    write_json_exclusive(out / "run_header.json", header)

    tests = run_unit_tests() if run_tests else {"ok": None, "note": "not run in this mode"}
    log(f"unit tests: {tests.get('ran')} ran ok={tests['ok']}")
    if tests["ok"] is False:  # never spend (or burn) seeds on a failing implementation
        summary = {"mode": mode, "complete": False, "valid": False, "unit_tests": tests,
                   "verdict": {"verdict": "INVALID", "reason": "unit tests failed; no data generated"}}
        write_json_exclusive(out / "summary.json", summary)
        write_json_exclusive(out / "output_manifest.json", output_manifest(out))
        return summary
    warmup(cfg)

    # ---------------- phase A: pilot-only selection ----------------
    specs = env_specs(cfg)
    envs, selections = {}, {}
    complete = True
    for eid in env_ids:
        if time.perf_counter() - t_start > cap:
            complete = False
            break
        spec = specs[eid]
        env = make_environment(cfg, root, spec)
        Xp, Yp = sample_split(cfg, env, "pilot")
        pilot = PilotView.from_arrays(Xp, Yp, env.task_weights)
        h = save_npz_exclusive(out / "pilot" / f"env{eid:02d}_pilot.npz", X=Xp, Y=Yp)
        sel = run_selectors(pilot, cfg, early_master(root, spec.block.block))
        sel["pilot_npz_sha256"] = h
        sel["env"] = {"env_id": eid, "block": spec.block.block, "cell": spec.block.cell, "q": spec.block.q,
                      "delta": spec.block.delta, "cov_kind": spec.cov_kind}
        selections[eid] = sel
        envs[eid] = env
        log(f"selected env {eid}")
    sel_hash = write_json_exclusive(out / "selections.json", {str(k): v for k, v in selections.items()})
    write_bytes_exclusive(out / "selections.sha256", (sel_hash + "  selections.json\n").encode())
    log(f"selections locked sha256={sel_hash}")

    # ---------------- phase B: training and evaluation (uses the locked file) ----------------
    with open(out / "selections.json", "rb") as f:
        raw = f.read()
    if hashlib.sha256(raw).hexdigest() != sel_hash:
        raise RuntimeError("selections.json changed after locking")
    locked = json.loads(raw)
    records, pop_records, all_rows, timing_rows = [], [], [], []
    num = {"finite": True, "max_moment_vs_sample_norm": 0.0, "max_traj_vs_numpy_norm": 0.0,
           "max_naive_two_impl": 0.0, "early_replay_ok": True}
    for eid in list(selections):
        if time.perf_counter() - t_start > cap:
            complete = False
            break
        env = envs[eid]
        res = train_and_evaluate(cfg, env, root, out)
        sel = locked[str(eid)]
        rec = {"env_id": eid, "cov_kind": env.spec.cov_kind, "block": env.spec.block.block, "Z": res["Z"],
               "cand_loss": res["cand_loss"], "selected": _selected_map(sel),
               "oracle_population_gradient_selected": res["oracle_population_gradient_selected"]}
        records.append(rec)
        pop_records.append({**rec, "cand_loss": res["cand_pop"]})
        all_rows += res["rows"]
        num["finite"] &= res["finite"]
        num["max_moment_vs_sample_norm"] = max(num["max_moment_vs_sample_norm"], res["max_moment_vs_sample_norm"])
        num["max_traj_vs_numpy_norm"] = max(num["max_traj_vs_numpy_norm"], res["max_traj_vs_numpy_norm"])
        num["max_naive_two_impl"] = max(num["max_naive_two_impl"], sel["naive_two_impl_max_rel_diff"])
        if "early_ceiling_replay" in sel:
            num["early_replay_ok"] &= bool(sel["early_ceiling_replay"]["selected_equal"])
        trow = {"env_id": eid}
        for m, ts in sel["timing_s"].items():
            trow[f"{m}_median_s"] = float(np.median(ts))
        for m in ("early_ceiling", "early_time_matched"):
            e = sel[m]
            trow.update({f"{m}_{k}": e[k] for k in ("status", "budget_s", "rounds_in_budget", "wall_s", "setup_s",
                                                       "min_granularity_s", "overrun_s", "selection_elapsed_s")})
        trow["train_s"] = res["train_s"]
        timing_rows.append(trow)
        log(f"trained env {eid} ({res['train_s']:.1f}s)")
    total_s = time.perf_counter() - t_start
    complete &= len(records) == expected_envs and total_s <= cap

    tol = cfg["gates"]
    num["moment_vs_sample_ok"] = num["max_moment_vs_sample_norm"] <= tol["moment_vs_sample_loss_rel_tol"]
    num["two_impl_ok"] = num["max_naive_two_impl"] <= tol["naive_two_impl_rel_tol"]
    num["traj_consistency_ok"] = num["max_traj_vs_numpy_norm"] <= tol["moment_vs_sample_loss_rel_tol"]
    valid = bool(num["finite"] and num["moment_vs_sample_ok"] and num["two_impl_ok"] and num["traj_consistency_ok"]
                 and num["early_replay_ok"] and (tests["ok"] is not False))
    post_freeze = None
    if mode == "scientific":
        valid &= tests["ok"] is True
        try:
            verify_freeze(PKG_ROOT, Path(freeze_manifest["manifest_path"]))
            post_freeze = "verified"
        except Exception as exc:  # sources changed during the run
            post_freeze = f"FAILED: {exc}"
            valid = False

    _write_csv(out / "candidate_seed_losses.csv", all_rows)
    _write_csv(out / "timing.csv", timing_rows)
    conv = np.array([r["conv_delta_norm"] for r in all_rows]) if all_rows else np.zeros(0)
    iso_gap = np.array([r["iso_population_gap_over_Z"] for r in all_rows if r["iso_population_gap_over_Z"] is not None])
    summary = {"mode": mode, "complete": complete, "valid": valid, "total_runtime_s": total_s,
               "runtime_cap_s": cap, "unit_tests": tests, "numerical_gates": num, "selections_sha256": sel_hash,
               "post_run_freeze_check": post_freeze, "n_envs_completed": len(records), "expected_envs": expected_envs,
               "optimization_diagnostics": {
                   "role": "diagnostic only; the verdict changes only through the pre-registered INCONCLUSIVE views",
                   "n_runs": int(conv.size),
                   "n_last_window_decrease_over_threshold": int(np.sum(conv > cfg["training"]["convergence_flag_threshold_normalized"])),
                   "max_last_window_decrease_over_Z": float(conv.max()) if conv.size else None,
                   "iso_population_gap_over_Z_max": float(iso_gap.max()) if iso_gap.size else None,
                   "iso_population_gap_over_Z_median": float(np.median(iso_gap)) if iso_gap.size else None,
                   "iso_population_gap_note": "finite-training-sample + optimization gap to the exact class optimum; not by itself an optimization failure"},
               }
    if analyze and records:
        g = cfg["gates"]
        summary["verdict"] = ev.verdict(records, g, valid=valid, complete=complete, expected_envs=expected_envs,
                                        population_records=pop_records if g["robustness_population_risk"] else None)
        summary["winner_diversity"] = ev.winner_diversity(records)
        summary["descriptive_test_regret"] = ev.descriptive(records, g)
        summary["descriptive_population_regret"] = ev.descriptive(pop_records, g)
        an = [i for i, r in enumerate(records) if r["cov_kind"] == "aniso"]
        summary["hindsight_best_fixed_all"] = ev.hindsight_best_fixed(records)
        summary["hindsight_best_fixed_aniso"] = ev.hindsight_best_fixed(records, an)
        summary["half_pilot_reliability"] = {k: v["half_pilot_reliability"] for k, v in locked.items()}
        env_rows = []
        tab = ev.regret_table(records, "mean")
        for i, r in enumerate(records):
            L = ev.agg_losses(r["cand_loss"], "mean") / r["Z"]
            row = {"env_id": r["env_id"], "block": r["block"], "cov_kind": r["cov_kind"], "Z": r["Z"]}
            row.update({f"norm_test_loss_c{c}": float(L[c]) for c in range(len(L))})
            row["best_mean"] = int(np.argmin(L))
            row["best_seed0"] = int(np.argmin(ev.agg_losses(r["cand_loss"], "seed0")))
            row["best_seed1"] = int(np.argmin(ev.agg_losses(r["cand_loss"], "seed1")))
            row.update({f"sel_{m}": r["selected"][m] for m in r["selected"]})
            row["oracle_population_gradient_selected"] = r["oracle_population_gradient_selected"]
            row.update({f"regret_{m}": tab[m][i] for m in tab})
            env_rows.append(row)
        _write_csv(out / "env_table.csv", env_rows)
        write_json_exclusive(out / "records.json", {"test": records, "population": pop_records})
    write_json_exclusive(out / "summary.json", summary)
    write_json_exclusive(out / "output_manifest.json", output_manifest(out))
    return summary
