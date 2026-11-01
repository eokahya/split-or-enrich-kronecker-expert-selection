import copy
import unittest

from tests._common import load_cfg
from soe import evaluate as ev

METHODS = ("naive", "crossfit", "covariance", "variance_only", "tail_only", "early_ceiling",
           "early_time_matched")


def table(primary_regrets, baseline_aniso=0.05, early=0, seed_split=None):
    """24 envs (even iso, odd aniso), Z = 1, candidate c has loss 1 + c/100 except the forced ones.

    Candidate 0 is always the best (loss 1.0). A method with regret x selects candidate 8 whose
    loss is set to 1 + x (otherwise 1 + baseline). Baselines (covariance, early, fixed 1..8) are made bad.
    """
    recs = []
    for i in range(24):
        kind = "aniso" if i % 2 else "iso"
        last = primary_regrets[i] if primary_regrets[i] > 0 else baseline_aniso
        L = [1.0] + [1.0 + baseline_aniso] * 7 + [1.0 + last]
        Ls = [list(L), list(L)]
        if seed_split is not None and i in seed_split:
            d = seed_split[i]
            Ls[0][8] += d
            Ls[1][8] -= d
        sel = {m: 1 for m in METHODS}
        sel["naive"] = 8 if primary_regrets[i] > 0 else 0
        sel["early_ceiling"] = early
        sel["early_time_matched"] = None  # secondary diagnostic, never in gate 3
        recs.append({"env_id": i, "cov_kind": kind, "Z": 1.0, "cand_loss": Ls, "selected": sel})
    return recs


class TestGates(unittest.TestCase):
    def setUp(self):
        self.g = load_cfg()["gates"]

    def V(self, recs, **kw):
        return ev.verdict(recs, self.g, valid=kw.get("valid", True), complete=kw.get("complete", True),
                          expected_envs=24)["verdict"]

    def test_regret_values(self):
        recs = table([0.0] * 24)
        r = recs[0]
        self.assertEqual(ev.regret(r, 0), 0.0)
        self.assertAlmostEqual(ev.regret(r, 3), 0.05, places=12)
        self.assertIsNone(ev.regret(r, None))
        tab = ev.regret_table(recs)
        self.assertTrue(all(v >= 0 for m in tab for v in tab[m] if v is not None))
        self.assertIn("fixed_8", tab)

    def test_pass(self):
        # fixed_0 is the optimum everywhere: the hindsight-best fixed candidate is unbeatable -> no utility
        recs = table([0.0] * 24)
        self.assertEqual(self.V(recs), "NO_ADDED_UTILITY")
        # make the optimum vary across environments so no fixed candidate is optimal everywhere
        recs = table([0.0] * 24)
        for i, r in enumerate(recs):
            j = 1 + i % 7
            for s in range(2):
                r["cand_loss"][s][0], r["cand_loss"][s][j] = r["cand_loss"][s][j], r["cand_loss"][s][0]
            r["selected"]["naive"] = j
        self.assertEqual(self.V(recs), "PASS_CONSIDER_STAGE_B_DESIGN")
        hb = ev.hindsight_best_fixed(recs, [i for i in range(24) if i % 2])
        self.assertGreater(hb["mean_regret"], 0.01)
        # primary early baseline not measurable in one aniso env -> no practical-advantage claim
        recs[3]["selected"]["early_ceiling"] = None
        self.assertEqual(self.V(recs), "PASS_GATES_NO_PRACTICAL_ADVANTAGE_CLAIM")

    def _varying(self, prim):
        recs = table(prim)
        for i, r in enumerate(recs):
            j = 1 + i % 7
            for s in range(2):
                r["cand_loss"][s][0], r["cand_loss"][s][j] = r["cand_loss"][s][j], r["cand_loss"][s][0]
            if r["selected"]["naive"] == 0:
                r["selected"]["naive"] = j
        return recs

    def test_gate2_boundary(self):
        prim = [0.0] * 24
        for i in (0, 2, 4, 6):
            prim[i] = 0.03
        self.assertEqual(self.V(self._varying(prim)), "PASS_CONSIDER_STAGE_B_DESIGN")  # 20/24
        prim[8] = 0.03
        self.assertEqual(self.V(self._varying(prim)), "FAIL_PREDICTION")  # 19/24
        prim[8] = 0.0199
        self.assertEqual(self.V(self._varying(prim)), "PASS_CONSIDER_STAGE_B_DESIGN")
        # threshold is inclusive: a regret of exactly 0.02 (exactly representable construction) counts
        rec = {"env_id": 0, "cov_kind": "iso", "Z": 1.0, "cand_loss": [[0.0, 0.02], [0.0, 0.02]], "selected": {}}
        self.assertEqual(ev.regret(rec, 1), 0.02)
        self.assertTrue(ev.regret(rec, 1) <= self.g["gate2_regret_threshold"])

    def test_gate3_margin(self):
        recs = self._varying([0.0] * 24)
        # covariance selector becomes nearly optimal on aniso envs -> fails margin
        for r in recs:
            if r["cov_kind"] == "aniso":
                r["selected"]["covariance"] = r["selected"]["naive"]
        self.assertEqual(self.V(recs), "NO_ADDED_UTILITY")
        g = ev.gates(recs, self.g)
        self.assertEqual(g["gate3_strongest_baseline"], "covariance")
        # the primary early baseline (common ceiling) does enter gate 3
        recs = self._varying([0.0] * 24)
        for r in recs:
            if r["cov_kind"] == "aniso":
                r["selected"]["early_ceiling"] = r["selected"]["naive"]
        self.assertEqual(self.V(recs), "NO_ADDED_UTILITY")
        # crossfit (secondary), ablations and the time-matched diagnostic never enter gate 3
        recs = self._varying([0.0] * 24)
        for r in recs:
            r["selected"]["crossfit"] = r["selected"]["naive"]
            r["selected"]["variance_only"] = r["selected"]["naive"]
            r["selected"]["early_time_matched"] = r["selected"]["naive"]
        self.assertEqual(self.V(recs), "PASS_CONSIDER_STAGE_B_DESIGN")

    def test_inconclusive_seed_sensitivity(self):
        prim = [0.0] * 24
        for i in (0, 2, 4, 6):
            prim[i] = 0.019
        recs = self._varying(prim)
        for i in (0, 2, 4, 6, 8):
            for s, d in ((0, 0.005), (1, -0.005)):
                recs[i]["cand_loss"][s][8] += d
            recs[i]["selected"]["naive"] = 8
        recs[8]["cand_loss"][0][8] = 1.019 + 0.005
        recs[8]["cand_loss"][1][8] = 1.019 - 0.005
        # mean regret 0.019 everywhere among the 5 -> gate2 pass (24/24); seed0 -> 0.024 fails 5 envs
        self.assertEqual(self.V(recs), "INCONCLUSIVE")

    def test_inconclusive_population_risk_view(self):
        recs = self._varying([0.0] * 24)
        self.assertEqual(self.V(recs), "PASS_CONSIDER_STAGE_B_DESIGN")
        pop = copy.deepcopy(recs)
        for i in range(0, 24, 2)[:5]:
            for s in range(2):
                pop[i]["cand_loss"][s][8] = 1.0 - 0.03  # another candidate is better by 0.03 in population risk
        v = ev.verdict(recs, self.g, valid=True, complete=True, expected_envs=24, population_records=pop)
        self.assertEqual(v["verdict"], "INCONCLUSIVE")
        self.assertEqual(v["labels_by_aggregation"]["population_risk_mean"], "FAIL_PREDICTION")
        v = ev.verdict(recs, self.g, valid=True, complete=True, expected_envs=24, population_records=recs)
        self.assertEqual(v["verdict"], "PASS_CONSIDER_STAGE_B_DESIGN")

    def test_winner_diversity(self):
        recs = self._varying([0.0] * 24)
        wd = ev.winner_diversity(recs)
        self.assertEqual(wd["all"]["n_distinct_winners"], 7)
        self.assertFalse(wd["all"]["single_winner"])
        self.assertEqual(wd["all"]["seed_winner_agreement"], 24)
        wd1 = ev.winner_diversity(table([0.0] * 24))
        self.assertTrue(wd1["all"]["single_winner"])
        # regression: a method measurable in some environments and not others (mixed None / int)
        for i, r in enumerate(recs):
            r["selected"]["early_time_matched"] = None if i % 3 else 2
        wd2 = ev.winner_diversity(recs)
        self.assertEqual(wd2["all"]["method_choices"]["early_time_matched"], {"2": 8, "NOT_MEASURABLE": 16})
        self.assertEqual(wd2["aniso"]["method_choices"]["early_time_matched"]["NOT_MEASURABLE"], 8)

    def test_invalid_incomplete(self):
        recs = self._varying([0.0] * 24)
        self.assertEqual(self.V(recs, valid=False), "INVALID")
        self.assertEqual(self.V(recs, complete=False), "INCOMPLETE")
        self.assertEqual(self.V(recs[:23]), "INCOMPLETE")
        self.assertEqual(self.V(recs, valid=False, complete=False), "INVALID")


if __name__ == "__main__":
    unittest.main()
