# Split or Enrich — Stage A synthetic falsification pilot: PROTOCOL

Status: **PROVISIONAL — not frozen.** It becomes binding only when the coordinator writes a freeze
manifest (`scripts/freeze.py`) over this file, `config.json`, `soe/*.py`, `scripts/*.py` and `tests/*.py`.
No scientific-seed data has been generated. Numbers in `config.json` are the single source of truth;
this document defines them.

## 1. Question and non-claims

Does the naive pilot-gradient score (pooled-within-group variance + rearrangement-SVD spectral tail,
computed from 64 pilot examples per task) select, among nine equal-parameter Kronecker expert
configurations, a configuration whose trained held-out loss is close to the best trained candidate,
and does it beat strong practical baselines on anisotropic inputs?

Not claimed under any outcome: an original theorem; covariance-weighted Kronecker optimality; anything
about nonlinear networks, LLMs, token routing or MI; hardware savings; independence of the 24
environments (there are 12 independent teacher blocks); statistical power, coverage or significance.
Passing permits only *considering* a Stage B design. Failing stops this naive selector.

## 2. Environment grid

- d = 64 = 8 x 8, four tasks, fixed task IDs, task weights w_t = 1/4.
- Teacher block (cell (q, delta), replicate rep): independent draws of a common component C and
  task-specific components S_t, each `sum_{j=1}^{q} kron(a_j, b_j)` with a_j, b_j 8x8 i.i.d. N(0,1),
  then rescaled to Frobenius norm nu = 8 (= sqrt(d)) **before** combination.
  Structured teacher: `C + delta * S_t`. Dense residual `R_t`: an i.i.d. N(0,1) 64x64 matrix per task,
  rescaled so `||R_t||_F = 0.1 * ||C + delta S_t||_F` (norm ratio 0.1, energy ratio 0.01).
  `T_t = C + delta S_t + R_t`. Generic draws have exact Kronecker rank q per component (tested).
- Cells, in order: (q, delta) = (2, .2), (2, 1), (6, .2), (6, 1); 3 replicate blocks each -> 12 blocks;
  block id = 3*cell + rep.
- Each block appears in two environments with **identical teachers**: `iso` (Sigma_t = I) and `aniso`
  (Sigma_t = Q_t diag(lambda) Q_t^T, lambda_i ∝ 4^{-i/63}, trace 64, condition number 4, Q_t independent
  Haar per task, via QR with sign fix). env id = 2*block + (0 iso | 1 aniso) -> 24 environments.
- Output noise: isotropic Gaussian, per-coordinate variance `sigma_t^2 = 0.05 tr(T_t Sigma_t T_t^T) / 64`
  (population signal energy of that environment/task; so iso and aniso noise levels differ slightly).
- Inputs are zero-mean Gaussian; no biases anywhere.

## 3. Data, RNG streams and isolation

numpy PCG64 seeded by `SeedSequence(entropy=root, spawn_key=key)`. Roots: scientific 2026103101,
technical 9900731, unit tests 7700123, warmup 5500001 (distinct; the pipeline refuses the scientific
root unless in frozen scientific mode, and technical mode refuses it always).

| stream | key | content |
|---|---|---|
| teacher | (1, block) | C, S_0..S_3, R_0..R_3, in that order |
| covariance | (2, block) | Q_0..Q_3 (aniso only) |
| data | (3, block, cov 0/1, task, split 0..3) | z (n x 64) then noise (n x 64); x = L z, y = T x + eps |
| training init | (4, block, seed 0/1) | master (4,8,8,8) N(0,1); shared by the iso/aniso pair |
| early-selector init | (5, block) | master (4,8,8,8) N(0,1); shared by the iso/aniso pair |

Per task and environment: pilot 64, train 512, val 128, test 2048 — separate streams, so each split is
independent of the others' sizes and of generation order (tested).

Isolation is enforced by phase order: Phase A samples only pilots for all 24 environments, runs every
selector on a read-only `PilotView` (pilot X, Y, task weights; nothing else), and writes
`selections.json` + its SHA-256 **before any train/val/test array exists** (tested with a spy).
Phase B re-reads the locked file from disk and uses it for evaluation. Teachers, covariances, noise
levels and the normalizer Z are never passed to any selector. Training loss is not normalized.

## 4. Loss and normalization

Row-sample convention. Per task `L_t(K) = (1/2n) sum_i ||y_i - K x_i||^2`; the objective is
`sum_t w_t L_t(K_{g(t)})`. Hence `-grad_K L_t(0) = Y_t^T X_t / n =: P_t` (population `T_t Sigma_t`).
Per-environment normalizer (evaluation only):
`Z = sum_t w_t (1/2) E||y_t||^2 = sum_t w_t (1/2)(tr(T_t Sigma_t T_t^T) + 64 sigma_t^2)`, the known
expected zero-model half loss. No task-wise gradient normalization anywhere.

## 5. Candidates

Restricted-growth strings over tasks 0..3 with 1, 2 or 4 blocks, sorted lexicographically:
`0 E1r8:0000, 1 0001, 2 0010, 3 0011, 4 0100, 5 0101, 6 0110, 7 0111 (E2r4), 8 E4r2:0123`.
Expert e of a candidate is `K_e = sum_{j<r} kron(A_ej, B_ej)`, A, B in R^{8x8}; every candidate has
exactly 1024 factor scalars. Active compute differs across E (not equalized, reported as such).
Ties in any argmin: lowest canonical index among exactly equal float64 values (numpy argmin).

## 6. Selectors (all pilot-only) and comparators

Notation: for candidate c with groups e (task sets), `pi_e = sum_{t in e} w_t`,
`Pbar_e = sum_{t in e} w_t P_t / pi_e`, `R(.)` the rearrangement with `R(kron(A,B)) = vec(A) vec(B)^T`.

1. **PRIMARY — naive full-64 score**
   `S(c) = 1/2 sum_e [ sum_{t in e} w_t ||P_t - Pbar_e||_F^2 + pi_e sum_{j>r_e} s_j^2(R(Pbar_e)) ]`
   `     = 1/2 [ sum_t w_t ||P_t||^2 - sum_e pi_e sum_{j<=r_e} s_j^2(R(Pbar_e)) ]`.
   Both forms are implemented independently; their agreement is a numerical gate. Under isotropy with
   population P_t = T_t this equals the exact minimum excess risk of the candidate class (tested).
2. **Secondary — symmetric crossfit (not a gate).** Halves = pilot rows 0–31 and 32–63. For fit half a,
   `K_e^a` = best rank-r_e Kronecker approximation of `Pbar_e^a`; score
   `1/2 * sum_{(a,b)} sum_e sum_{t in e} w_t (1/2)(||K_e^a||^2 - 2<P_t^b, K_e^a>)`. Estimates the risk of
   the fitted update up to a constant; not an unbiased oracle minimum; selection still optimistic.
3. **Strong comparator — covariance-corrected heuristic.** `C_t = X_t^T X_t / 64` (uncentered, same
   pilot), `lambda_t = 0.1 tr(C_t)/64`, `That_t = P_t (C_t + lambda_t I)^{-1}`, scored by formula 1.
   Explicitly heuristic, not the covariance-weighted Kronecker optimum.
4. **Strong comparator — early training / validation, PRIMARY early baseline: common allotted
   ceiling of 1.0 s per environment** (in gate 3). See §7. Secondary: the same search time-matched to
   the measured naive time (cost diagnostic only, never in gate 3, never given a score when
   NOT_MEASURABLE).
5. **Fixed candidates 0..8** (always choose c). The fixed candidate with the lowest mean regret on the
   analysed subset ("hindsight-best fixed") is an evaluation-only comparator, not deployable.
6. **Ablations (diagnostics only, never gates):** variance-only (first term; trivially selects E4),
   tail-only (second term; tends to E1).
7. **Half-pilot reliability (descriptive):** formula 1 on rows 0–31 and on rows 32–63 separately;
   Spearman rank correlations half/half and half/full, and argmin agreement. Never an invalidity rule.
8. **Oracle diagnostic (post-lock, not a selector):** formula 1 on population `T_t Sigma_t`
   ("infinite pilot"), separating finite-pilot noise from proxy bias. Reported only.

## 7. Early-training selector and timing

- Same 64 pilot rows: rows 0–31 fit, 32–63 validation (per task). The separate 128-example validation
  split is NOT early-selector data. All nine candidates start from the identical initialization scheme
  of §8 (early-init stream), Adam lr 0.01, full batch, moment backend, **CPU float64, one torch thread**
  (BLAS/OpenMP threads pinned to 1 by the scripts before numpy loads), explicit process warmup on a
  separate warmup-root environment.
- One **round** = one Adam step for every candidate (fused in one optimizer; Adam is elementwise so this
  equals nine separate steps — tested), then validation losses of all nine. Only complete sweeps count,
  so candidates are always compared at equal completed steps and there is no candidate-order bias.
  Selection = argmin validation loss at the last round completed within the budget.
- **Timing scope (all methods):** the shared PilotView (raw pilot arrays) is created outside timing.
  Each method is timed from those raw arrays to its chosen index: gradients/moments, covariance solve,
  SVDs, model setup, all rounds, validation and the final choice. Analytic selectors: median of 5
  repeats (results asserted identical); ablations: one timing.
- **PRIMARY early baseline:** budget = common allotted ceiling 1.0 s. This is an equal *allotted*
  ceiling, not equal consumed time: analytic selectors finish naturally (milliseconds) and are never
  delayed; reports must not describe the 1 s search as costing the same as the naive score. The
  consumed wall time (including the discarded overrun round) is recorded as `wall_s`, the time at the
  selected round as `selection_elapsed_s`, plus rounds, setup, granularity and overrun. The selection
  is replayed deterministically with the recorded round count (numerical gate `early_replay_ok`).
- **SECONDARY cost diagnostic:** identical search with budget = this environment's median naive time.
  If setup + one round + one validation exceeds it, the status is `NOT_MEASURABLE` (no score, excluded).
- If the primary early baseline is NOT_MEASURABLE in any aniso environment it is excluded from gate 3
  and the verdict cannot exceed `PASS_GATES_NO_PRACTICAL_ADVANTAGE_CLAIM`.
- **Technical benchmark (technical seeds, M2 Max, 1 thread, `technical_benchmark.json`):** naive
  median 3.5–3.6 ms, covariance 4.4–4.6 ms; early cpu-float64 minimum granularity 4.8–6.3 ms, so the
  time-matched diagnostic is NOT_MEASURABLE in all four benchmark environments; the 1.0 s ceiling
  completes 284–321 rounds (~3.1–3.5 ms per round incl. validation), consumed 1.001–1.006 s.
  CPU float32 is faster and MPS float32 slower; neither is used.

## 8. Candidate training

- Each of the nine candidates x 2 init seeds, 400 full-batch Adam steps on the 512 training examples
  (moment backend, CPU float64), lr 0.01, betas (0.9, 0.999), eps 1e-8, weight decay 0, no schedule,
  no early stopping, final iterate used. All 18 runs of an environment share one optimizer instance
  (identical to separate runs; tested).
- Initialization: `A_ej ~ N(0, 1/(8 r))` i.i.d., `B_ej = 0`, no output multiplier (so K = 0 at start
  and the initial gradient reaches only B). Common random numbers across candidates: A of candidate
  (E, r) = master[:E, :r] / sqrt(8r). The 1/sqrt(r) scale keeps the initial update magnitude roughly
  rank-independent under Adam (CDKA's rank-dependent-gradient concern). Fixed before any run.
- Synthetic moment backend: the dense 64x64 K is materialized per step; a factorized forward is
  implemented and tested against dense kron and used to recompute every test loss sample-by-sample
  (numerical gate). No hardware-efficiency claim.

## 9. Evaluation

- Candidate test loss `L_c` = mean over the two init seeds of the test half loss (2048 independent test
  examples per task). No test-based seed choice.
- Regret of a method in an environment: `(L_selected - min_c L_c) / Z` (absolute gaps also saved).
  Reference = best *trained finite* candidate, not a certified global optimum.
- Also saved: validation loss (diagnostic only — never used for selection or retuning), exact
  population risk `sum_t w_t/2 [tr((T_t-K)Sigma_t(T_t-K)^T) + 64 sigma_t^2]` (diagnostic), training
  trajectories, factors and dense K for every candidate/seed, all moments, pilots, scores.

## 10. Gates (engineering research-investment thresholds, not power claims)

1. Valid and complete: unit tests pass in-process, all numerical gates (§11) pass, all values finite,
   all 24 environments x 9 candidates x 2 seeds completed within 3600 s total (from runner start,
   including tests and warmup). Otherwise `INVALID` / `INCOMPLETE`. Environments are never dropped.
2. At least 20 of 24 environments have primary normalized test regret <= 0.02 (inclusive).
3. Over the 12 aniso environments: `mean primary regret <= min(baseline means) - 0.01`, baselines =
   covariance heuristic, the primary 1.0 s-ceiling early selector (if MEASURED in all 12), and each fixed
   candidate 0..8 (each one candidate across all 12; hence including the hindsight-best fixed).
   Crossfit (even if best), ablations, oracle and the time-matched diagnostic are excluded. A near-zero
   baseline mean makes this gate intentionally unattainable -> `NO_ADDED_UTILITY` (no adaptive-selection
   value demonstrated in this design; not a universal theoretical falsification).

## 11. Numerical validity, optimization diagnostics and INCONCLUSIVE rule (pre-registered)

Numerical gates (any failure -> INVALID): naive two-implementation relative difference <= 1e-10;
moment-based vs factorized sample-based test loss |diff|/Z <= 1e-9; final training-trajectory loss vs
numpy recomputation |diff|/Z <= 1e-9; deterministic replay of every early-ceiling selection; all
finite; in-process unit tests pass (scientific mode aborts **before generating any data** if they fail);
freeze hashes re-verified after the run. Declared unit-test tolerances are in `tests/_common.py`.

Optimization diagnostics — reported, **never verdict-changing on their own**:
- last-window criterion: `(L_350 - L_400)/Z` per environment x candidate x seed on the training loss;
  flagged if > 0.001 (count reported);
- seed instability: per environment the winner under seed 0, seed 1 and the seed mean, their agreement,
  and `|L_seed0 - L_seed1|/Z` per candidate (from `candidate_seed_losses.csv`);
- finite-sample oracle gap (iso environments only): population risk of the trained candidate minus the
  exact class optimum, /Z. It mixes finite-training-sample estimation error with optimization error
  and is **not** by itself an optimization failure. The last-window criterion measures recent loss
  decrease, not the unknown optimization error, and does not certify convergence.
- Finite-pilot noise or half-pilot disagreement is a property of the 64-example method, not invalidity.

INCONCLUSIVE (the only optimization/seed rule that changes the verdict, symmetric for PASS and FAIL):
compute the gate-2/3 label (FAIL_PREDICTION / NO_ADDED_UTILITY / PASS) under each of: seed mean
(primary), seed 0 only, seed 1 only, per-candidate min over seeds, and exact population risk instead
of test loss. If the labels disagree, the verdict is `INCONCLUSIVE` (seed/optimization or test-sampling
ambiguity alters the conclusion). These views are sensitivity analyses only; no seed is chosen by test.

## 12. Verdict labels (precedence top-down)

`INVALID` > `INCOMPLETE` > `INCONCLUSIVE` > `FAIL_PREDICTION` (gate 2) > `NO_ADDED_UTILITY`
(gate 3) > `PASS_GATES_NO_PRACTICAL_ADVANTAGE_CLAIM` (gates pass but the primary 1 s early baseline was
NOT_MEASURABLE somewhere in aniso) > `PASS_CONSIDER_STAGE_B_DESIGN`. Winner diversity is always reported;
a single candidate winning throughout means no adaptive-selection value is demonstrated in this design. No automatic LLM run, whitening rescue or threshold revision follows
any label.

## 13. Provenance and outputs

`scripts/run_pilot.py --freeze M --out DIR` verifies every hashed file (no missing, changed or extra
files matching the frozen globs) and python/numpy/torch versions before creating anything, refuses an
existing DIR, and opens every output file in exclusive-create mode. Scientific mode additionally
requires the scientific root, exactly all 24 environment ids, the unmodified on-disk `config.json`
whose hash is in the manifest (so 400 steps, 2 inits, sizes and thresholds cannot be altered
programmatically), and an in-process test run; it re-verifies the freeze after the run. Failed and
earlier runs are never deleted or overwritten. Outputs: `run_header.json`,
`pilot/*.npz`, `selections.json` + `selections.sha256` (written before Phase B), `envs/*.npz`
(moments, teacher, Sigma, noise, trajectories, factors, dense K), `candidate_seed_losses.csv`,
`timing.csv` (all selector times, early statuses/rounds/consumed wall/overrun, training time),
`env_table.csv` (normalized candidate losses, per-seed winners, every method's choice and regret),
`records.json`, `summary.json` (verdict, all gate views, optimization diagnostics, winner diversity,
half-pilot reliability, descriptive test- and population-risk regrets, hindsight-best fixed),
`output_manifest.json` (SHA-256 of all outputs).

## 14. Resolved ambiguities (my resolutions; coordinator may overrule before freeze)

R1 component norm nu = 8 and "delta multiplies task-specific component only"; R2 dense residual
independent per task, norm ratio relative to that task's structured part; R3 noise fraction relative to
population signal energy of the specific environment; R4 iso/aniso pair shares teachers and init
streams but not data streams; R5 crossfit halves and early fit/val split both = first/last 32 rows;
R6 covariance estimate uncentered; R7 analytic timing = median of 5 repeats after process warmup, timing scope as in §7;
R8 early selector validates after every complete round (anytime) and uses the last in-budget round;
R9 primary early baseline counts in gate 3 only if MEASURED in all 12 aniso envs; R10 hindsight-best
fixed = one candidate per analysed subset; R11 gate inclusivity: `<= 0.02` and `<= strongest - 0.01`;
R12 INCONCLUSIVE views as in §11; R13 runtime cap includes tests and warmup; R14 every process runs with
1 torch thread and BLAS/OpenMP threads pinned to 1 (training too, for one consistent timing regime).

## 15. Known design properties and limitations (development evidence on TECHNICAL seeds only)

These were measured on independent technical seeds before any scientific data existed. They authorize
no tuning of the grid, pilot size, thresholds, learning rate or step count.

L1 **Discriminability:** selector-free population-moment training over all 24 technical environments:
in all 6 (q=2, delta=.2) environments every candidate is within 0.02 Z of the best (gate 2 is free
there); E4 is never best; E1 is best by 0.045–0.17 Z in all 12 q=6 environments; the constant-E1 rule
is within 0.02 in 18/24 environments with mean aniso excess ~0.021. Gate 2 alone is weak evidence and
gate 3 is likely dominated by "always E1".
L2 **Optimization floor:** 400 Adam steps on exact isotropic population moments leave a gap to the SVD
optimum of up to 0.0017 Z and flipped a near-tie winner for one seed in a (2, .2) block. This cautions
against interpreting small measured gaps as differences between globally optimal architectures;
it does not establish a universal resolution floor for measured finite-training regret. §11 views
partly guard the investment decision.
L3 **Pilot noise:** at n = d = 64 the error energy of `X^T X/n` is about (d+1)/n ≈ 1 times the identity's,
so the noise energy in P_t is of the order of the signal energy. This is a limitation of the deployed
64-example selector; the oracle and half-pilot diagnostics quantify it.
L4 **Mild anisotropy:** condition number 4; a null aniso effect says nothing about strong anisotropy.
L5 **Time-matched early search is NOT_MEASURABLE** (granularity ~1.4–1.8x the naive time); practical
comparisons rest on the 1.0 s allotted ceiling, which consumes ~280–300x the naive wall time.
L6 **Dependence:** 24 environments = 12 independent teacher blocks; no power or coverage claims.

## 16. Versions (benchmark machine)

Python 3.11.13, numpy 2.4.6 (Accelerate), torch 2.6.0, macOS 26.6.2 arm64 (Apple M2 Max); runs use
1 torch thread and VECLIB_MAXIMUM_THREADS = OMP_NUM_THREADS = 1. The runner refuses a version mismatch with the freeze manifest.
