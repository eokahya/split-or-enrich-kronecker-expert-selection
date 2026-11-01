# Split or Enrich — measured scientific results

Main run: `runs/scientific_01`; frozen before scientific data: `FREEZE_MANIFEST.json`.

Complete and valid: True, True; runtime 112.500753 s; 59 unit tests, no skips; 24 environments × 9 candidates × 2 seeds = 432 fits, 400 steps each.

Formal frozen verdict: **NO_ADDED_UTILITY** in all five sensitivity views. This means the required improvement over the strongest baseline was not established. It does NOT mean adaptation has no value over a fixed architecture.

| Anisotropic mean normalized regret | Value |
|---|---:|
| naive | 0.000399224930 |
| covariance | 0.000380419324 |
| early_ceiling | 0.000957333012 |
| crossfit | 0.001593190678 |
| fixed_0 | 0.021003397674 |

Gate 2: 24/24 within 0.02; required >=20. PASS.
Gate 3: naive <= strongest baseline - 0.01; 0.000399224930 <= -0.009619580676. FAIL. Even zero regret cannot pass this gate on this near-saturated benchmark.

| Selection method | Median consumed time (ms) |
|---|---:|
| naive | 3.660104 |
| covariance | 4.590771 |
| crossfit | 17.965479 |
| early_ceiling | 1002.066958 |

The early comparator has a common allotted 1-second ceiling, not equal consumed compute. The last complete in-budget round is selected; actual consumed time includes a discarded overrun round. Secondary matching to the naive elapsed time is NOT_MEASURABLE in all 24 environments. No artificial poor scores are assigned.

12 independent teacher blocks, each paired iso/condition-4 covariance, not 24 independent replications. Winner counts: E1 12, E2 12, E4 0; seed winners agree 21/24. Half-pilot choices agree 16/24, mean rank correlation 0.654167. These are descriptive diagnostics, not generalization or confidence statements.

Optimization diagnostics: {"role": "diagnostic only; the verdict changes only through the pre-registered INCONCLUSIVE views", "n_runs": 432, "n_last_window_decrease_over_threshold": 2, "max_last_window_decrease_over_Z": 0.0016796946981768284, "iso_population_gap_over_Z_max": 0.007164048423444039, "iso_population_gap_over_Z_median": 0.0022957602675189553, "iso_population_gap_note": "finite-training-sample + optimization gap to the exact class optimum; not by itself an optimization failure"}

All scientific outputs and source hashes passed an independent coordinator audit. The audit reconstructs Kronecker matrices and test losses without importing study modules, checks Z and all five gate decisions. See `analysis/independent_audit.json` and its reproducible script.

Development disclosure: separate technical seeds were inspected before freezing; no grid/threshold tuning followed. Opus disclosed deletion of initial probe artifacts before a preservation instruction; later v1/v2 development outputs and all scientific outputs remain. The main scientific experiment has not been rerun.

Actual post-result reviews from Fable Extra, GPT Pro and Antigravity are complete. All three advise NO-GO for direct LLM Stage B, with different reasons and separate-study proposals. Unsupported reviewer claims are rejected in docs/coordination_log.md; see RESULTS_TR.md for the coordinator decision.
