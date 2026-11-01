"""Regrets, pre-registered gates and verdict labels.

Input is a list of per-environment records (plain dicts) so that gates can be tested on
handcrafted tables:
  {"env_id", "cov_kind", "Z",
   "cand_loss": [[...9 test losses, seed 0], [... seed 1]],
   "selected": {method_name: int | None}}      None = NOT_MEASURABLE in that environment
Regret of a method in an environment, for aggregation a in {mean, seed0, seed1, min}:
   (Lbar_a[selected] - min_c Lbar_a[c]) / Z,    Lbar_mean = mean over the two seeds, etc.
The non-mean aggregations are sensitivity views only, never a seed choice for reporting.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

PRIMARY = "naive"
GATE3_DEPLOYABLE = ("covariance", "early_ceiling")
FIXED_PREFIX = "fixed_"

VERDICTS = (
    "INVALID", "INCOMPLETE", "INCONCLUSIVE", "FAIL_PREDICTION", "NO_ADDED_UTILITY",
    "PASS_GATES_NO_PRACTICAL_ADVANTAGE_CLAIM", "PASS_CONSIDER_STAGE_B_DESIGN",
)


def agg_losses(cand_loss, agg: str) -> np.ndarray:
    L = np.asarray(cand_loss, dtype=np.float64)
    if agg == "mean":
        return L.mean(axis=0)
    if agg == "seed0":
        return L[0]
    if agg == "seed1":
        return L[1]
    if agg == "min":
        return L.min(axis=0)
    raise ValueError(agg)


def regret(rec: dict, sel: Optional[int], agg: str = "mean") -> Optional[float]:
    if sel is None:
        return None
    L = agg_losses(rec["cand_loss"], agg)
    return float((L[sel] - L.min()) / rec["Z"])


def method_names(records: List[dict], n_cand: int) -> List[str]:
    names = list(records[0]["selected"].keys())
    return names + [f"{FIXED_PREFIX}{c}" for c in range(n_cand) if f"{FIXED_PREFIX}{c}" not in names]


def selection(rec: dict, method: str) -> Optional[int]:
    if method.startswith(FIXED_PREFIX):
        return int(method[len(FIXED_PREFIX):])
    return rec["selected"][method]


def regret_table(records: List[dict], agg: str = "mean") -> Dict[str, List[Optional[float]]]:
    n_cand = len(records[0]["cand_loss"][0])
    return {m: [regret(r, selection(r, m), agg) for r in records] for m in method_names(records, n_cand)}


def _mean_or_none(vals):
    return None if any(v is None for v in vals) else float(np.mean(vals))


def gates(records: List[dict], gcfg: dict, agg: str = "mean") -> dict:
    tab = regret_table(records, agg)
    prim = tab[PRIMARY]
    n_ok = int(sum(v <= gcfg["gate2_regret_threshold"] for v in prim))
    g2 = n_ok >= gcfg["gate2_min_env_count"]

    an = [i for i, r in enumerate(records) if r["cov_kind"] == "aniso"]
    prim_an = _mean_or_none([prim[i] for i in an])
    baselines, not_measurable = {}, []
    names = list(GATE3_DEPLOYABLE) + [m for m in tab if m.startswith(FIXED_PREFIX)]
    for m in names:
        v = _mean_or_none([tab[m][i] for i in an]) if m in tab else None
        if v is None:
            not_measurable.append(m)
        else:
            baselines[m] = v
    strongest = min(baselines, key=lambda k: (baselines[k], k))
    g3 = prim_an <= baselines[strongest] - gcfg["gate3_margin"]
    return {
        "aggregation": agg, "gate2_pass": bool(g2), "gate2_envs_within_threshold": n_ok,
        "gate2_n_envs": len(prim), "gate3_pass": bool(g3), "primary_aniso_mean_regret": prim_an,
        "gate3_baseline_aniso_means": baselines, "gate3_strongest_baseline": strongest,
        "gate3_strongest_mean": baselines[strongest], "gate3_not_measurable": not_measurable,
    }


def _label(g: dict) -> str:
    if not g["gate2_pass"]:
        return "FAIL_PREDICTION"
    if not g["gate3_pass"]:
        return "NO_ADDED_UTILITY"
    return "PASS"


def verdict(records: List[dict], gcfg: dict, *, valid: bool, complete: bool, expected_envs: int,
            population_records: Optional[List[dict]] = None) -> dict:
    """population_records: same records with exact population risks in place of test losses; when given,
    their mean-aggregation label is one more pre-registered robustness view (test-sampling noise)."""
    if not valid:
        return {"verdict": "INVALID", "reason": "tests, numerical gates or finiteness failed"}
    if not complete or len(records) != expected_envs:
        return {"verdict": "INCOMPLETE", "reason": f"{len(records)}/{expected_envs} environments completed"}
    per_agg = {a: gates(records, gcfg, a) for a in gcfg["robustness_aggregations"]}
    if population_records is not None:
        per_agg["population_risk_mean"] = gates(population_records, gcfg, "mean")
    labels = {a: _label(g) for a, g in per_agg.items()}
    base = {"gates_by_aggregation": per_agg, "labels_by_aggregation": labels}
    if len(set(labels.values())) > 1:
        return {"verdict": "INCONCLUSIVE", "reason": "seed/aggregation/test-sampling view changes the gate label", **base}
    lab = labels["mean"]
    if lab != "PASS":
        return {"verdict": lab, **base}
    if "early_ceiling" in per_agg["mean"]["gate3_not_measurable"]:
        return {"verdict": "PASS_GATES_NO_PRACTICAL_ADVANTAGE_CLAIM",
                "reason": "primary early baseline NOT_MEASURABLE in some aniso environment", **base}
    return {"verdict": "PASS_CONSIDER_STAGE_B_DESIGN", **base}


def hindsight_best_fixed(records: List[dict], subset=None, agg: str = "mean") -> dict:
    """Evaluation-only comparator: ONE fixed candidate with the lowest mean regret over `subset`."""
    idx = range(len(records)) if subset is None else subset
    n_cand = len(records[0]["cand_loss"][0])
    means = [float(np.mean([regret(records[i], c, agg) for i in idx])) for c in range(n_cand)]
    best = int(np.argmin(means))
    return {"candidate": best, "mean_regret": means[best], "all_means": means}


def winner_diversity(records: List[dict]) -> dict:
    """Which candidate is best per environment (seed-mean and per init seed) and per-method choices."""
    def counts(vals):
        c = Counter("NOT_MEASURABLE" if v is None else int(v) for v in vals)
        ints = sorted(k for k in c if k != "NOT_MEASURABLE")
        out = {str(k): c[k] for k in ints}
        if "NOT_MEASURABLE" in c:
            out["NOT_MEASURABLE"] = c["NOT_MEASURABLE"]
        return out
    out = {}
    for name, sel in (("all", lambda r: True), ("iso", lambda r: r["cov_kind"] == "iso"),
                      ("aniso", lambda r: r["cov_kind"] == "aniso")):
        rs = [r for r in records if sel(r)]
        w_mean = [int(np.argmin(agg_losses(r["cand_loss"], "mean"))) for r in rs]
        w_s0 = [int(np.argmin(agg_losses(r["cand_loss"], "seed0"))) for r in rs]
        w_s1 = [int(np.argmin(agg_losses(r["cand_loss"], "seed1"))) for r in rs]
        out[name] = {"best_mean": counts(w_mean), "best_seed0": counts(w_s0), "best_seed1": counts(w_s1),
                     "seed_winner_agreement": int(sum(a == b for a, b in zip(w_s0, w_s1))),
                     "n_distinct_winners": len(set(w_mean)), "single_winner": len(set(w_mean)) == 1,
                     "method_choices": {m: counts([r["selected"][m] for r in rs]) for m in rs[0]["selected"]}}
    return out


def descriptive(records: List[dict], gcfg: dict) -> dict:
    tab = regret_table(records, "mean")
    out = {}
    for m, vals in tab.items():
        row = {}
        for name, sel in (("all", lambda r: True), ("iso", lambda r: r["cov_kind"] == "iso"),
                          ("aniso", lambda r: r["cov_kind"] == "aniso")):
            v = [x for x, r in zip(vals, records) if sel(r)]
            meas = [x for x in v if x is not None]
            row[name] = {"n": len(v), "n_measured": len(meas),
                         "mean": float(np.mean(meas)) if meas else None,
                         "max": float(np.max(meas)) if meas else None,
                         "n_within_gate2": int(sum(x <= gcfg["gate2_regret_threshold"] for x in meas))}
        out[m] = row
    return out
