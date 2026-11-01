"""TECHNICAL smoke (technical seed root only, never the scientific root).

1. End-to-end pipeline on technical environments (default: one teacher block = iso + aniso pair),
   including the in-pipeline unit-test run, selection lock, training and numerical gates.
2. Optimization-adequacy check: on independent technical teachers (one per cell), train all nine
   candidates x two init seeds with POPULATION isotropic moments (Sxx = I, Syx = T) for the frozen
   400 Adam steps and compare with the exact rearrangement-SVD optimum.
3. Design diagnostic: population-moment-trained candidate losses for all 24 TECHNICAL environments
   (selector-free: how separable the candidates are relative to the 0.02 regret threshold).
Only engineering/design quantities are written to technical_smoke.json (no selection/regret outcomes).

    python scripts/technical_smoke.py --out technical_smoke.json --run-dir runs_technical/smoke_1
"""
from _path import ROOT  # noqa: E402,F401  (must precede numpy)

import argparse
import csv
import json

import numpy as np
import torch

from soe.candidates import CANDIDATES
from soe.envgen import block_specs, env_specs, make_environment
from soe.pipeline import init_masters, run
from soe.provenance import _default, write_json_exclusive
from soe.selectors import naive_scores_captured
from soe.train import FusedCandidates, torch_moments, train_fused


def optimisation_check(cfg, root):
    out = []
    specs = env_specs(cfg)
    for b in block_specs(cfg):
        if b.replicate != 0:
            continue
        spec = next(s for s in specs if s.block.block == b.block and s.cov_kind == "iso")
        env = make_environment(cfg, root, spec)
        T, w = env.teacher.T, env.task_weights
        mom = {"Sxx": np.broadcast_to(np.eye(64), (4, 64, 64)).copy(), "Syx": T.copy(),
               "syy": np.einsum("tij,tij->t", T, T) + 64 * env.noise_var}
        m = FusedCandidates(CANDIDATES, init_masters(root, b.block, 2), w)
        traj = train_fused(m, torch_moments(mom, torch.float64, "cpu"), cfg["training"], cfg["training"]["steps"])
        opt = naive_scores_captured(T, w) + float(np.sum(w * 0.5 * 64 * env.noise_var))
        Z = env.zero_model_half_loss
        gap = (traj[-1] - opt[:, None]) / Z  # (9, 2)
        win = cfg["training"]["convergence_window_steps"]
        out.append({"block": b.block, "q": b.q, "delta": b.delta,
                    "gap_over_Z": gap.tolist(), "max_gap": float(gap.max()), "min_gap": float(gap.min()),
                    "seed_gap_diff_max": float(np.abs(gap[:, 0] - gap[:, 1]).max()),
                    "argmin_trained_equals_argmin_opt": [int(np.argmin(traj[-1, :, s])) == int(np.argmin(opt)) for s in range(2)],
                    "last_window_decrease_over_Z_max": float(((traj[-1 - win] - traj[-1]) / Z).max()),
                    "gap_at_step": {str(k): float(((traj[k] - opt[:, None]) / Z).max()) for k in (100, 200, 300, 400)}})
    return out


def design_diagnostic(cfg, root):
    """Selector-free property of the frozen design on TECHNICAL teachers: population-moment-trained
    candidate losses (normalized by Z), i.e. how separable the nine candidates are at all."""
    out = []
    for spec in env_specs(cfg):
        env = make_environment(cfg, root, spec)
        T, Sig, w = env.teacher.T, env.cov.Sigma, env.task_weights
        mom = {"Sxx": Sig.copy(), "Syx": T @ Sig, "syy": env.signal_energy + 64 * env.noise_var}
        m = FusedCandidates(CANDIDATES, init_masters(root, spec.block.block, 2), w)
        traj = train_fused(m, torch_moments(mom, torch.float64, "cpu"), cfg["training"], cfg["training"]["steps"])
        L = traj[-1].min(axis=1) / env.zero_model_half_loss
        exc = L - L.min()
        srt = np.sort(exc)
        out.append({"env_id": spec.env_id, "q": spec.block.q, "delta": spec.block.delta, "cov": spec.cov_kind,
                    "best_norm_loss": float(L.min()), "excess_over_best": exc.tolist(),
                    "second_best_gap": float(srt[1]), "spread": float(srt[-1]),
                    "n_within_0.02": int(np.sum(exc <= 0.02)), "n_within_0.01": int(np.sum(exc <= 0.01))})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--env-ids", default="0,1")
    a = ap.parse_args(argv)
    cfg = json.load(open(ROOT / "config.json"))
    root = cfg["seeds"]["technical_root"]
    env_ids = [int(x) for x in a.env_ids.split(",")]
    summ = run(cfg, root=root, env_ids=env_ids, out_dir=ROOT / a.run_dir, mode="technical", run_tests=True,
               analyze=False)
    rows = list(csv.DictReader(open(ROOT / a.run_dir / "candidate_seed_losses.csv")))
    conv = np.array([float(r["conv_delta_norm"]) for r in rows])
    timing = list(csv.DictReader(open(ROOT / a.run_dir / "timing.csv")))
    thr = cfg["training"]["convergence_flag_threshold_normalized"]
    opt = optimisation_check(cfg, root)
    design = design_diagnostic(cfg, root)
    res = {
        "label": "TECHNICAL SMOKE - technical seed root, not scientific evidence",
        "technical_root": root, "env_ids": env_ids, "run_dir": a.run_dir,
        "pipeline": {k: summ[k] for k in ("complete", "valid", "total_runtime_s", "numerical_gates", "selections_sha256")},
        "unit_tests": {k: summ["unit_tests"][k] for k in ("ran", "failures", "errors", "skipped", "ok")},
        "convergence": {"n_runs": int(conv.size), "max_last50_decrease_over_Z": float(conv.max()),
                        "min_last50_decrease_over_Z": float(conv.min()),
                        "n_flagged_over_threshold": int(np.sum(conv > thr)), "threshold": thr},
        "timing": timing,
        "isotropic_population_optimum_check": opt,
        "isotropic_population_optimum_summary": {
            "max_gap_over_Z": max(o["max_gap"] for o in opt), "min_gap_over_Z": min(o["min_gap"] for o in opt),
            "max_seed_gap_diff": max(o["seed_gap_diff_max"] for o in opt)},
        "design_diagnostic_population_moments": design,
        "design_diagnostic_summary": {
            "n_envs": len(design),
            "n_envs_where_all_9_within_0.02": int(sum(d["n_within_0.02"] == 9 for d in design)),
            "median_n_within_0.02": float(np.median([d["n_within_0.02"] for d in design])),
            "median_spread": float(np.median([d["spread"] for d in design])),
            "median_second_best_gap": float(np.median([d["second_best_gap"] for d in design]))},
        "projected_full_runtime_s": None,
    }
    per_env = summ["total_runtime_s"] / max(1, len(env_ids))
    res["projected_full_runtime_s"] = {"naive_linear_extrapolation_24_envs_s": per_env * 24,
                                       "note": "includes one in-pipeline unit-test run and warmup, so it overestimates"}
    write_json_exclusive(ROOT / a.out, res)
    print(json.dumps({k: res[k] for k in ("pipeline", "unit_tests", "convergence", "isotropic_population_optimum_summary",
                                          "design_diagnostic_summary", "projected_full_runtime_s")}, indent=1, default=_default))


if __name__ == "__main__":
    main()
