# Coordinator read-only source audit (before any scientific result)

Reviewed initial candidates, Kronecker operations, environment generator, selectors, trainer and pipeline. Algebra checked: row-major forward convention; rearrangement; task-probability weighting of3+1 partitions; negative-gradient sign; crossfit sign; ridge solve orientation; quadratic sufficient-statistic loss; fused optimizer independence. No substantive algebra error found in these initial versions. Final updated files still require review.

Before freeze, ensure:

- Incorporate final_design_clarifications.md, especially primary common1s early comparator versus secondary actual-naive-time diagnostic.
- Report actual consumed early-selector wall time INCLUDING terminal overrun step. Snapshot elapsed time is not actual consumed time. Existing budget/overrun fields permit reconstruction but explicit wall_s is clearer. Fixed-step first_round_plus_eval_s must not mislabel400-round total as first-round time.
- Training convergence threshold in config currently appears diagnostic-only. Final protocol must say so or make a concrete predeclared technical-inconclusive rule; no hidden discretion after results. Reviewer asked for operational last-window and seed-instability criteria. Distinguish finite-sample oracle gap from convergence error.
- Scientific CLI must enforce exactly all24 environment IDs, scientific root,400steps and2initializations through frozen configuration; no subset with expected_envs=len(subset) promoted to full result.
- Do not proceed to costly scientific training if unit tests fail. Technical smoke and separate full tests must pass before freeze.
- Include half-pilot score/rank reliability diagnostics (descriptive, not automatic invalidity), actual covariance-selector timing, winner distribution, and technical-only benchmark.
- Final freeze must include code, tests, config and protocol. Recheck hashes after run. Persist all selection hashes before full training. Preserve prior outputs and failed runs.

No scientific source edits by coordinator; fixes remain assigned to the requested Claude Code Opus5.5 High author.

Concrete follow-up bug found while final edits were underway: `evaluate.winner_diversity.counts` uses `sorted(Counter(vals).items())`, which raises TypeError for mixed `None` and integer keys. Secondary `early_time_matched` may be measurable in some environments and not others. Use an explicit safe sort/count representation and add a mixed None/int regression test before final smoke/freeze. Native UI message delivery was not confirmed; this file is the authoritative audit item.

Coordinator independently ran the then-current suite on M2 Max outside sandbox:
`../.venv-stage1a-mps-py311/bin/python -m unittest discover -s tests -t . -v`
Result:56 tests passed in6.533s, none skipped; MPS test executed. This predates final queued clarifications and is not a freeze of changing source.

Inspected technical-only `runs_technical/probe_smoke.json`:36 candidate/seed fits in the2-environment smoke; largest last50-step loss decrease/Z=.0004483 (<.001 flag); population-moment isotropic optimum maximum gap/Z=.00170996. Opus additionally ran a24-environment selector-free design diagnostic on independent TECHNICAL seeds. Its outcomes are development evidence, not the held-out scientific grid. No scientific-root data have yet been run, and no teacher-grid, pilot-size, utility-threshold, learning-rate or training-step tuning is authorized from these diagnostics.

Final implementation audit: common 1-second primary early baseline, secondary naive-time diagnostic, explicit consumed wall time including discarded overrun, one torch thread, full-grid scientific guard, fail-fast unit tests, pre-training selection hash, post-run source verification, half-pilot reliability and mixed missing-value winner counts are implemented. Coordinator independently reran all 59 tests after these changes: PASS in 4.554 seconds, no skips, MPS precision test executed. Final technical smoke v2 also passes all 59 tests and two complete environments; the primary early baseline fits approximately 290 rounds. No scientific-root run has been performed as of this entry.

Two wording cautions for final protocol/report: last-window loss decrease measures recent progress, not the unknown optimization error or a certificate of convergence; a ~0.0017 Z gap on technical cases is a warning about near-tie interpretation, not a universal lower bound on resolvable measured regret.

Coordinator corrected these two prose claims directly in PROTOCOL.md before freeze, after reading the delivered implementation report; no implementation or test source was edited. Opus's report discloses deletion of initial `probe_*` technical artifacts before the preserve-outputs instruction. Retained v1/v2 artifacts and this audit log support the reported development checks; do not claim a complete archive of every development probe. Scientific outputs will all be preserved.
