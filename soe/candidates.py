"""The nine expert-count / capacity candidates.

Each candidate is a restricted-growth string (RGS) a over tasks 0..3:
a[0] = 0, a[i] <= max(a[:i]) + 1. Group label e is the expert index. Candidates are
the RGS with 1, 2 or 4 blocks, sorted lexicographically, which yields
E1 (0000), the seven E2 partitions (0001 ... 0111), E4 (0123).
Every expert in a candidate with E experts has r = 8 / E Kronecker terms, so every
candidate has E * r * (8*8 + 8*8) = 1024 trainable factor scalars.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

N_TASKS = 4
TOTAL_TERMS = 8
RANK_BY_E = {1: 8, 2: 4, 4: 2}


@dataclass(frozen=True)
class Candidate:
    index: int
    assignment: Tuple[int, ...]  # task -> expert
    n_experts: int
    rank: int

    @property
    def label(self) -> str:
        return f"E{self.n_experts}r{self.rank}:" + "".join(map(str, self.assignment))

    def groups(self) -> List[Tuple[int, ...]]:
        return [tuple(t for t, g in enumerate(self.assignment) if g == e) for e in range(self.n_experts)]

    def n_params(self, p: int = 8, q: int = 8) -> int:
        return self.n_experts * self.rank * (p * p + q * q)


def _rgs(n: int):
    def rec(prefix, mx):
        if len(prefix) == n:
            yield tuple(prefix)
            return
        for v in range(mx + 2):
            yield from rec(prefix + [v], max(mx, v))

    yield from rec([0], 0)


def build_candidates() -> List[Candidate]:
    out = []
    for a in sorted(_rgs(N_TASKS)):
        E = max(a) + 1
        if E in RANK_BY_E:
            out.append(Candidate(len(out), a, E, RANK_BY_E[E]))
    return out


CANDIDATES: List[Candidate] = build_candidates()


def distinct_groups() -> List[Tuple[int, ...]]:
    """All distinct (task-set) groups across candidates, in first-appearance order."""
    seen = []
    for c in CANDIDATES:
        for g in c.groups():
            if g not in seen:
                seen.append(g)
    return seen
