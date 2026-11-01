# Final pre-outcome clarifications — 2026-10-31

No scientific outcomes have been run or inspected. These clarifications supersede provisional brief where stated.

1. Primary accuracy-utility gates remain: >=20/24 normalized regrets <=.02; and mean naive regret across12 anisotropic environments <= minimum of comparator MEANS across those same12 minus.01. Comparators include ridge score, early selection and each of9 fixed candidates. Best-fixed uses ONE candidate across all12. Near-zero baseline regret makes the utility gate intentionally unattainable: classify NO_ADDED_UTILITY, not universal theoretical falsification.

2. Normalization is the known population zero-model **half squared loss INCLUDING output noise**, available only to evaluator. Raw pilot alone to selectors. Primary uses full64; early selector uses32+32 of the SAME64. Additional128 validation samples are not early-selector data.

3. Timing: retain actual end-to-end naive-time-matched early search as a secondary cost diagnostic. It can be NOT_MEASURABLE and must never receive an artificial bad score. To ensure a meaningful strong early comparator, the PRIMARY early baseline receives a **common maximum selection budget of1.0 second per environment**, a ceiling available to every selector. Analytic methods finish naturally without artificial delay; actual times reported. This is equal ALLOTTED ceiling, not equal consumed time; never describe a1s early search as consuming the same time as a millisecond naive score. The stronger-baseline utility gate remains conservative for naive. CPUfloat64, one torch CPU thread, explicit warmup; consistently define whether shared pilot array creation is outside timing, and include gradient/moment construction, nine-candidate search/setup, validation and choice inside method timing. Use complete round-robin sweeps to avoid candidate-order bias; report granularity/overruns. If1s cannot yield a complete fair comparison on technical benchmark, report unresolved before freeze rather than silently enlarging budget.

4. Declare numerical optimization diagnostics before run. Tail-loss change and per-init candidate choices must be reported; finite-training/population-oracle gap alone is not an optimization failure. Finite-pilot gradient noise or disagreement is a diagnostic of this64-example method, not automatic invalidation of the experiment. Gates on optimization validity must not simply excuse an unpromising outcome.

5. The paired covariance environments share teachers and training-init streams. They are12 independent teacher blocks, not24 independent observations. No population coverage/power assertion. Crossfit stays secondary even if best-performing.

6. Do not expand to condition100, add LLM/LoRA runs, increase pilot samples or tune teacher grid after outcomes. Report winner diversity to reveal whether the candidate-selection problem is degenerate. A single fixed candidate winning throughout means no adaptive selection value demonstrated in this design; it is not a universal capacity result.

7. Final protocol/config/code must be frozen after tests and technical benchmark and before any scientific selection/evaluation. Coordinator performs final freeze and run. Opus prepares implementation and tests; no scientific run until explicit follow-up.
