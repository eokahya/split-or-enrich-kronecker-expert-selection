import itertools
import unittest

from tests._common import load_cfg
from soe.candidates import CANDIDATES, distinct_groups


def set_partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for p in set_partitions(rest):
        yield [[first]] + p
        for i in range(len(p)):
            yield p[:i] + [[first] + p[i]] + p[i + 1:]


class TestCandidates(unittest.TestCase):
    def test_nine_candidates_and_order(self):
        self.assertEqual(len(CANDIDATES), 9)
        self.assertEqual([c.index for c in CANDIDATES], list(range(9)))
        self.assertEqual(["".join(map(str, c.assignment)) for c in CANDIDATES],
                         ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "0123"])
        self.assertEqual([(c.n_experts, c.rank) for c in CANDIDATES], [(1, 8)] + [(2, 4)] * 7 + [(4, 2)])

    def test_all_seven_two_group_partitions_independently(self):
        ref = {frozenset(frozenset(b) for b in p) for p in set_partitions([0, 1, 2, 3]) if len(p) == 2}
        self.assertEqual(len(ref), 7)
        got = {frozenset(frozenset(g) for g in c.groups()) for c in CANDIDATES if c.n_experts == 2}
        self.assertEqual(got, ref)
        # restricted-growth canonical form: first task in expert 0, labels appear in order
        for c in CANDIDATES:
            seen = -1
            for g in c.assignment:
                self.assertLessEqual(g, seen + 1)
                seen = max(seen, g)

    def test_param_count_1024(self):
        cfg = load_cfg()
        for c in CANDIDATES:
            self.assertEqual(c.n_params(), 1024)
            self.assertEqual(c.n_params(), cfg["candidates"]["factor_scalars_per_candidate"])
            self.assertEqual(c.n_experts * c.rank, 8)

    def test_distinct_groups(self):
        g = distinct_groups()
        self.assertEqual(len(g), 15)  # full set + 4 singletons + 6 pairs + 4 triples
        self.assertEqual(len(set(g)), 15)


if __name__ == "__main__":
    unittest.main()
