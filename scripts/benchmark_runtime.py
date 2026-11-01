"""TECHNICAL runtime benchmark (technical seed root only; no scientific data).

Under the frozen runtime settings (torch threads and BLAS env from config.json) it measures analytic
selector wall times, the early selector's setup / round granularity / rounds within the 1.0 s common
ceiling and within the naive-time-matched budget (frozen device cpu float64, plus cpu float32 and
mps float32 for information), full-training time per environment, and float32 vs float64 agreement.
Output: JSON (refuses to overwrite).

    python scripts/benchmark_runtime.py --out technical_benchmark.json
"""
from _path import ROOT  # noqa: E402,F401  (must precede numpy)

import argparse
import copy
import json
import statistics
import time

import numpy as np
import torch

from soe.candidates import CANDIDATES
from soe.early import early_select
from soe.envgen import PilotView, env_specs, make_environment, moments, sample_split
from soe.pipeline import configure_runtime, early_master, init_masters, warmup
from soe.provenance import environment_versions, write_json_exclusive
from soe.selectors import select_covariance, select_crossfit, select_naive
from soe.train import FusedCandidates, torch_moments, train_fused

ENV_IDS = (0, 1, 22, 23)


def _median_time(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(ts), "min_s": min(ts), "max_s": max(ts)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=30)
    a = ap.parse_args(argv)
    cfg = json.load(open(ROOT / "config.json"))
    root = cfg["seeds"]["technical_root"]
    assert root != cfg["seeds"]["scientific_root"]
    rt = configure_runtime(cfg)
    warmup(cfg)
    specs = env_specs(cfg)
    ceiling = cfg["early_selector"]["primary_common_ceiling_s"]
    devices = [("cpu", "float64"), ("cpu", "float32")] + ([("mps", "float32")] if torch.backends.mps.is_available() else [])
    res = {"label": "TECHNICAL BENCHMARK - technical seed root", "versions": environment_versions(), "runtime": rt,
           "root": root, "env_ids": ENV_IDS, "frozen_early_device": [cfg["early_selector"]["device"], cfg["early_selector"]["dtype"]],
           "analytic": {}, "early": {}, "training": {}, "precision": {}}
    for eid in ENV_IDS:
        env = make_environment(cfg, root, specs[eid])
        X, Y = sample_split(cfg, env, "pilot")
        pv = PilotView.from_arrays(X, Y, env.task_weights)
        res["analytic"][eid] = {
            "naive": _median_time(lambda: select_naive(pv), a.repeats),
            "crossfit": _median_time(lambda: select_crossfit(pv, (32, 32)), a.repeats),
            "covariance": _median_time(lambda: select_covariance(pv, 0.1), a.repeats),
        }
        naive_budget = res["analytic"][eid]["naive"]["median_s"]
        em = early_master(root, specs[eid].block.block)
        per_dev = {}
        for dev, dt in devices:
            c2 = copy.deepcopy(cfg)
            c2["early_selector"]["device"], c2["early_selector"]["dtype"] = dev, dt
            early_select(pv, em, c2, fixed_rounds=3)  # device warmup
            tm = early_select(pv, em, c2, budget_s=naive_budget)
            ce = early_select(pv, em, c2, budget_s=ceiling)
            per_dev[f"{dev}_{dt}"] = {
                "time_matched_status": tm["status"], "time_matched_rounds": tm["rounds_in_budget"],
                "time_matched_wall_s": tm["wall_s"], "min_granularity_s": tm["min_granularity_s"],
                "setup_s": tm["setup_s"], "ceiling_status": ce["status"], "ceiling_rounds": ce["rounds_in_budget"],
                "ceiling_wall_s": ce["wall_s"], "ceiling_overrun_s": ce["overrun_s"],
                "approx_round_plus_eval_s": (ce["selection_elapsed_s"] - ce["setup_s"]) / max(1, ce["rounds_in_budget"]),
            }
        res["early"][eid] = per_dev
        mom = moments(*sample_split(cfg, env, "train"))
        masters = init_masters(root, specs[eid].block.block, 2)
        tr = {}
        for dev, dt in devices:
            dtype = {"float64": torch.float64, "float32": torch.float32}[dt]
            mt = torch_moments(mom, dtype, dev)
            train_fused(FusedCandidates(CANDIDATES, masters, env.task_weights, dtype, dev), mt, cfg["training"], 5)
            m = FusedCandidates(CANDIDATES, masters, env.task_weights, dtype, dev)
            t0 = time.perf_counter()
            traj = train_fused(m, mt, cfg["training"], cfg["training"]["steps"])
            if dev == "mps":
                torch.mps.synchronize()
            tr[f"{dev}_{dt}"] = (time.perf_counter() - t0, traj[-1])
        res["training"][eid] = {k: {"wall_s": v[0]} for k, v in tr.items()}
        ref = tr["cpu_float64"][1]
        for k, (_, fin) in tr.items():
            if k != "cpu_float64":
                res["precision"][f"{eid}_{k}_max_abs_final_diff_over_Z"] = float(np.max(np.abs(fin - ref)) / env.zero_model_half_loss)
        d = per_dev["cpu_float64"]
        print(f"env {eid}: naive {naive_budget*1e3:.2f} ms, cov {res['analytic'][eid]['covariance']['median_s']*1e3:.2f} ms; "
              f"cpu64 early: granularity {d['min_granularity_s']*1e3:.2f} ms, time-matched {d['time_matched_status']}, "
              f"ceiling {d['ceiling_rounds']} rounds ({d['ceiling_wall_s']:.3f}s consumed); train "
              + ", ".join(f"{k}={v['wall_s']:.2f}s" for k, v in res["training"][eid].items()))
    write_json_exclusive(ROOT / a.out if not a.out.startswith("/") else a.out, res)


if __name__ == "__main__":
    main()
