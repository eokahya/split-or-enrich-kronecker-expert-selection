# Implementation report — Split or Enrich Stage A pilot (Claude Code Opus 5.5 High)

Date 2026-10-31. **No scientific-root data were generated, no scientific selection or evaluation was
run, nothing was committed, pushed, downloaded or placed in a worktree.** All work is inside
`split_or_enrich/`; the parent project was not modified. `docs/final_design_clarifications.md` and
`docs/coordinator_pre_run_audit.md` (including the mixed None/int counting bug) are incorporated.

## Files

| file | role |
|---|---|
| `config.json` | every pinned size, seed root, threshold, optimizer and timing setting |
| `PROTOCOL.md` | definitions, isolation, gates, verdict labels, resolved ambiguities, technical-seed limitations |
| `README.md` | commands |
| `soe/kron.py` | row-major Kronecker convention, rearrangement R and inverse, Van Loan best-r fit, dense and factorized forward |
| `soe/candidates.py` | nine RGS candidates (E1r8, seven E2r4, E4r2), 1024 factor scalars each |
| `soe/envgen.py` | teachers, covariances, noise, Z, RNG stream layout, read-only `PilotView`, moments |
| `soe/selectors.py` | primary naive score (two independent implementations), crossfit, covariance heuristic, ablations |
| `soe/early.py` | round-robin early selector: 1 s common ceiling (primary, gate 3) and naive-time-matched (secondary), deterministic replay |
| `soe/train.py` | moment backend, shape-grouped fused Adam (identical to separate runs), factorized sample loss, population risk |
| `soe/evaluate.py` | regrets, gates, INCONCLUSIVE views, verdicts, hindsight-best fixed, winner diversity |
| `soe/provenance.py` | freeze manifest, verification (changed/missing/extra files, versions), exclusive-create outputs |
| `soe/pipeline.py` | phase A pilot-only selection, then lock + SHA-256, then phase B training/evaluation and analysis |
| `scripts/run_pilot.py` | scientific runner; requires an approved, matching freeze manifest |
| `scripts/freeze.py` | coordinator writes the freeze manifest |
| `scripts/technical_smoke.py`, `scripts/benchmark_runtime.py` | technical-root-only checks |
| `tests/` (59 tests) | see below; tolerances declared in `tests/_common.py` |
| `technical_benchmark.json`, `technical_smoke.json`, `runs_technical/smoke_02/` | current technical outputs |
| `runs_technical/superseded_v1/`, `runs_technical/smoke_01/` | pre-clarification technical outputs, preserved |

## Checks run (technical/unit roots only)

- `python -m unittest discover -s tests -t .`: **59 passed, 0 skipped** (MPS test executed).
  The in-pipeline test run in the smoke reports the same.
- Tests cover: rearrangement vs explicit loops, inverse and isometry; kron rank-one mapping and exact
  rank q; row- and column-major vectorization identities; factorized forward/backward vs dense kron
  autograd; central finite differences (h = 1e-6); Eckart–Young optimality of the Kronecker fit; all 7
  partitions vs an independent set-partition enumerator; 1024 parameters; both score implementations
  agree and equal half the directly computed constrained objective (equal and unequal task weights);
  hand-computed 3+1 weighting; exact isotropic risk identity plus in-class perturbations never improving;
  crossfit sign identity vs an independent one-way formula; covariance-heuristic solve orientation;
  zero-init gradients vs an analytic numpy gradient; Adam vs a manual numpy Adam; fused vs separate
  training; anisotropic population-risk formula vs 40k-sample Monte Carlo (5 SE, plus rejection of the
  wrong isotropic formula); Adam never beating the SVD optimum; stream integrity and order or size
  independence; teacher norms and residual ratio; covariance spectrum; noise energy; Z; selector
  signatures and PilotView read-only; corrupting population objects leaves selections unchanged;
  phase-order spy (no train/val/test sampled before `selections.sha256` exists); hash mutation, extra
  and missing file, unapproved and version-mismatch rejection; no overwrite; the runner rejecting a
  non-matching manifest before creating output; scientific-mode guards (subset of environments,
  modified config, config hash); failing tests abort before any data; gates and verdicts on
  handcrafted tables (boundaries, margin, INCONCLUSIVE seed and population views, winner diversity
  including mixed None/int); early-selector budget logic with a fake clock; round-robin equivalence;
  float32 agreement.
- Smoke (`technical_smoke.json`, 2 technical environments, end to end): valid and complete. Numerical
  gates: moment vs factorized sample test loss 1.3e-15 Z; trajectory vs numpy 7.3e-16 Z; naive two
  implementations 4.3e-16; early replay deterministic. Largest last-50-step decrease 4.5e-4 Z (0/36
  flagged). Time-matched early NOT_MEASURABLE in both environments; the 1 s ceiling completed 290 and
  294 rounds. Training took about 2.0–2.1 s per environment. Projected full run is about 2–3 min
  (cap 60 min).
- Benchmark (`technical_benchmark.json`, 1 thread): naive 3.5–3.6 ms, covariance 4.4–4.6 ms; early
  CPU float64 granularity 4.8–6.3 ms, so time-matched is NOT_MEASURABLE in 4/4; the 1 s ceiling gives
  284–321 rounds and consumed 1.001–1.006 s. CPU float32 and MPS float32 were measured for information
  only (float32 vs float64 final-loss difference ≤ 4e-7 Z).
- Isotropic population optimum check (technical): after 400 steps the gap to the exact SVD optimum is
  ≤ 0.0017 Z; one near-tie winner flipped for one seed.

## Critical design objections and limitations (for the coordinator / reviewers)

1. **Weak discriminability of the frozen grid (technical seeds, selector-free).** All nine candidates
   are within 0.02 Z in every (q=2, δ=.2) environment, E4 is never optimal, and E1 wins every q=6
   environment by 0.045–0.17 Z. A constant-E1 rule is within 0.02 in 18/24, with about 0.021 mean aniso
   excess. Expected consequence: gate 2 is partly free, and gate 3 is a contest against "always E1".
   I did not change the grid. Changing it now would be design tuning on development evidence and needs
   reviewer sign-off.
2. **Pilot noise (Fable dissent).** At n = d = 64 the pilot-gradient noise energy is of the order of the
   signal energy. Small-δ structure is likely below the selector's resolution. This is a limitation of
   the deployed 64-example selector, quantified by the oracle and half-pilot diagnostics. The pilot was
   not enlarged.
3. **Condition number 4 is mild.** No conclusion about strong anisotropy follows.
4. **Early-training comparison is budget-unequal by design.** Time-matched is NOT_MEASURABLE. The
   1 s ceiling consumes about 280–300× the naive wall time. Report it as an equal allotted ceiling only.
   The 1 s round count varies with timing jitter (about 280–320). Each selection is reproducible from
   its recorded round count, which the run checks.
5. **Optimization floor about 0.002 Z.** The best trained finite candidate is not a certified optimum.
   Regret differences below about 0.002 Z are not interpretable. The INCONCLUSIVE views (seed 0, seed 1,
   min, population risk) guard the conclusion, but not effect sizes.
6. **Moment backend.** Training materializes the dense 64x64 K per step. Factorized code is used for
   verification and test recomputation only, so there is no efficiency claim.
7. **24 environments = 12 independent teacher blocks.** There is no power or coverage statement. The
   iso and aniso pair shares teachers and init streams, while data streams are independent.
8. **Process honesty.**
   - Three test bugs were fixed during development, and no tolerance was loosened. A perturbation test
     stepped outside the candidate class, a Monte Carlo discriminator was too weak, and a float boundary
     case was not exactly representable.
   - One test was re-scoped. Fused-equivalence is checked in float64, and float32 agreement is checked
     separately against the declared 1e-4.
   - Early probe outputs (`runs_technical/probe_*`, the first benchmark probe) were deleted by me before
     the preserve-outputs instruction arrived. That includes `probe_smoke.json`, which the coordinator
     had inspected. Everything since is preserved.

## Hashes (provisional, NOT a freeze)

These are the first 16 hex digits of SHA-256 at report time. The coordinator's `scripts/freeze.py`
manifest is authoritative.
config.json 0dc15170a8b4b325 · PROTOCOL.md 5772d4000cd7c2f0 (this report and README.md are not hashed
by design) · soe/pipeline.py 4a3ba54ca160e8df · soe/selectors.py 6c34e798134b6589 · soe/train.py
a0f773adf2f00beb · soe/early.py 58eec7ed641ee3c4 · soe/evaluate.py c2803aaff1204354. Any further
edit to PROTOCOL.md changes its hash, so freeze only after the final read.

## Coordinator steps to run (not executed)

```bash
cd /Users/emre/Documents/Antropic_Mech_Int/split_or_enrich
../.venv-stage1a-mps-py311/bin/python -m unittest discover -s tests -t . -v
../.venv-stage1a-mps-py311/bin/python scripts/freeze.py --approver "<approval ref>" --out FREEZE_MANIFEST.json
../.venv-stage1a-mps-py311/bin/python scripts/run_pilot.py --freeze FREEZE_MANIFEST.json --out runs/scientific_01
```
