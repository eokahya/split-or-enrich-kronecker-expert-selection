# Split or Enrich: a pre-registered synthetic test of a gradient-based expert-count selector

A small, pre-registered experiment asking whether a cheap measurement taken before training can tell you how to split a fixed parameter budget between "more experts" and "bigger experts". The result is negative.

[Türkçe sürüm: README_TR.md](README_TR.md)

## Bottom line

- The frozen experiment (24 synthetic environments, 432 training runs) ended with the pre-registered verdict **NO_ADDED_UTILITY**.
- The proposed selector did choose good configurations: in 24 of 24 environments its choice was within the allowed tolerance of the best trained option. But a simple competing heuristic did just as well (its mean regret on the decisive subset was even slightly lower), so the extra accuracy benefit that the protocol demanded before any further investment was not shown.
- The planned follow-up on a real language model ("Stage B") was judged **NO-GO**. No Stage B experiment was run.
- This does not prove that the method can never help. It is a bounded result: one synthetic, linear problem family, mild input anisotropy, one pre-registered decision rule. It says the method did not earn further investment on this setup, nothing more.

## The problem in everyday language

Imagine you have a fixed budget to hire help for four jobs. You can hire one very capable generalist who handles all four jobs, two mid-level people who each handle two jobs, or four juniors who each handle one job. The total salary is the same in every case.

Which is best depends on the work. If the four jobs are really the same complicated job, one capable person is best, because splitting the budget leaves each junior too weak for it. If the four jobs are simple but different from each other, four juniors are best, because one person keeps getting pulled in different directions. The question is whether you can tell which situation you are in from a quick first look, before paying for a full trial of every option.

In machine-learning terms the "people" are experts in a mixture-of-experts layer, "capability" is how many parameters each expert gets, and the "quick first look" is the gradient of the loss computed on a few examples before any training.

Concrete toy version used here: four tasks, each a linear map from 64-dimensional inputs to 64-dimensional outputs. Every candidate model has exactly 1,024 trainable numbers. The candidates are one expert with capacity 8 (E1/r8), two experts with capacity 4 each (E2/r4, in all seven ways of splitting four tasks into two groups), or four experts with capacity 2 each (E4/r2), so nine candidates in total.

## The hypothesis that was tested

A score computed from initial gradients on 64 "pilot" examples per task (the "naive Kronecker score") would:

1. pick a configuration whose trained test loss is close to the best of the nine trained candidates, and
2. do so clearly better than strong, simple alternatives when the inputs are not isotropic (have unequal variance in different directions).

Passing would only have allowed designing a Stage B on a language model; it would not by itself have justified one.

## What the method does

**In plain words.** Each candidate makes two kinds of mistakes. If tasks that need different behaviour share one expert, the expert can only learn a compromise ("too many different jobs for one person"). If an expert has too little capacity, it cannot represent even a single job well ("job too complex for the person"). The score estimates both mistakes from the first gradient of each task and picks the candidate whose estimated total is smallest.

**Precisely.** Each expert is a 64x64 matrix written as a sum of r Kronecker products of 8x8 matrices; one term costs 128 parameters, and E experts of r terms cost 128·E·r = 1,024. With the row-sample convention, the negative initial gradient of task t is P_t = Yᵀ X / n from the pilot data. For a candidate with task groups e (weight π_e, group mean P̄_e), the score is

S(c) = ½ Σ_e [ Σ_{t∈e} w_t ‖P_t − P̄_e‖²_F + π_e Σ_{j>r_e} s_j²(R(P̄_e)) ],

where R is the rearrangement that maps a Kronecker product to a rank-one matrix and s_j are singular values. The first term is the within-group disagreement, the second is the Kronecker approximation tail. With isotropic inputs and exact (population) gradients this equals the exact minimum excess risk of the candidate class. The precise definitions are in [PROTOCOL.md](PROTOCOL.md).

## What it was compared against, and the decision rule

All comparisons select among the same nine trained candidates (each trained twice, 400 full-batch Adam steps, on separate training data).

| Comparator | Role |
|---|---|
| Covariance-corrected score: same score after multiplying the gradient by an inverse (ridge-regularized) input covariance estimated from the same pilot | Strong baseline, counts in the decision |
| Early training: briefly train all nine candidates on 32 pilot examples, validate on the other 32, with a common 1.0 second ceiling per environment | Strong baseline, counts in the decision |
| Each of the nine fixed candidates (always pick the same one) | Baselines, count in the decision; the best one in hindsight is E1/r8 |
| Cross-fitted score on two pilot halves | Secondary, reported only |
| Variance-only and tail-only ablations, half-pilot reliability, oracle score | Diagnostics only |

Regret is the selected candidate's test loss minus the best trained candidate's test loss, divided by Z, the expected loss of a model that always predicts zero (including noise). So a regret of 0.01 means 1% of that reference loss.

The gates were fixed before any scientific data existed (PROTOCOL.md sections 10 to 12):

1. Validity: all tests and numerical checks pass, all 24 environments complete within 3,600 seconds.
2. Good selection: in at least 20 of 24 environments the selector's regret is at most 0.02.
3. Added utility: on the 12 anisotropic environments, the selector's mean regret must be at least 0.01 below the best comparator mean.

If gate 3 fails the verdict is NO_ADDED_UTILITY. The verdict was also computed under five sensitivity views (seed mean, seed 0, seed 1, per-candidate minimum, exact population risk); if they disagreed, the verdict would have been INCONCLUSIVE.

The 24 environments are 12 independent "teacher" blocks, each run once with isotropic inputs and once with anisotropic inputs (condition number 4). They are not 24 independent replications.

## Results

Main run: `runs/scientific_01/`. "Aniso mean" is the mean normalized regret over the 12 anisotropic environments.

| Gate | Requirement | Measured | Outcome | Source (field) |
|---|---|---|---|---|
| 1. Validity | tests, numerical gates, 24 environments, ≤ 3,600 s | valid, complete, 112.50 s, 59 tests, 0 skipped | pass | `runs/scientific_01/summary.json` (`valid`, `complete`, `total_runtime_s`, `unit_tests`) |
| 2. Good selection | ≥ 20 of 24 with regret ≤ 0.02 | 24 of 24; worst 0.0030454 | pass | `summary.json` (`verdict.gates_by_aggregation.mean.gate2_envs_within_threshold`, `descriptive_test_regret.naive.all.max`) |
| 3. Added utility | naive ≤ best comparator − 0.01 = −0.0096196 | naive 0.0003992 vs covariance 0.0003804 | fail | `summary.json` (`verdict.gates_by_aggregation.mean.primary_aniso_mean_regret`, `gate3_strongest_mean`) |
| Verdict | | NO_ADDED_UTILITY in all 5 views | | `summary.json` (`verdict.verdict`, `verdict.labels_by_aggregation`) |

| Method | Aniso mean regret | Median selection time | Source |
|---|---:|---:|---|
| Naive Kronecker score (tested method) | 0.000399225 | 3.660 ms | `summary.json` `gate3_baseline_aniso_means` / `primary_aniso_mean_regret`; times: median over 24 environments of `runs/scientific_01/timing.csv` columns `naive_median_s`, `covariance_median_s`, `crossfit_median_s`, `early_ceiling_wall_s` |
| Covariance-corrected score | 0.000380419 | 4.591 ms | same |
| Early training, 1 s ceiling | 0.000957333 | 1,002.067 ms | same |
| Cross-fit (secondary) | 0.001593191 | 17.965 ms | `summary.json` `descriptive_test_regret.crossfit.aniso.mean` |
| Best fixed candidate (E1/r8) | 0.021003398 | none (fixed) | `summary.json` `gate3_baseline_aniso_means.fixed_0` |

Other facts from the same run:

- Adaptive selection was clearly better than always using the best fixed configuration: 0.021003 − 0.000399 ≈ 0.020604 Z on the anisotropic mean (difference of the two `summary.json` values above).
- Best candidate per environment (seed mean): E1 in 12, an E2 split in 12, E4 in 0 (`summary.json` `winner_diversity.all.best_mean`; candidate 0 is E1, 1 to 7 are E2, 8 is E4). The "four small experts" regime was therefore never actually tested as a winner.
- The two training seeds agreed on the winner in 21 of 24 environments (`winner_diversity.all.seed_winner_agreement`). Choices from the two pilot halves agreed in 16 of 24, with a mean half-to-half Spearman rank correlation of 0.654 (computed from the 24 entries of `summary.json` `half_pilot_reliability`).
- The early-training baseline was given a common 1-second ceiling; it is not compute-matched. A version matched to the naive method's time was NOT_MEASURABLE in all 24 environments because one round already took longer (`timing.csv` `early_time_matched_status`).
- Independent audit: `analysis/independent_audit.json` reports `status` PASS, 29 frozen source hashes verified, 432 rows, maximum recomputed test-loss error 7.0e-16 Z, and the same NO_ADDED_UTILITY label in all five views, recomputed without importing the study code.

Why gate 3 was unreachable here: the covariance baseline's regret (0.00038) is already far below 0.01, so even a perfect selector with zero regret could not beat it by 0.01. The threshold was not changed after seeing this. A tiny naive-versus-covariance difference is not evidence of either superiority or equivalence.

## Why the project was not continued

- The decisive gate failed: no accuracy benefit over a simple covariance-aware heuristic was shown.
- The four-small-experts regime never won, so the three-way "split versus enrich" story was only exercised between E1 and E2.
- The core ingredients are already in the literature (Kronecker mixtures of experts, Kronecker adapter component design with initial-gradient alignment, gradient-based rank allocation among experts), so a positive result would also have needed a clearer novelty case.
- Three AI chat assistants that reviewed the measured results all recommended NO-GO for the language-model stage, though for different reasons. The coordinator's decision was to close the pilot under its frozen label and not start Stage B. Any new hypothesis (for example stronger anisotropy, designs where E4 can win, or data-efficiency at other pilot sizes) would need its own pre-registration and could not be presented as a success of this run.

## Three kinds of evidence, kept separate

**(a) The frozen scientific experiment.** 29 files (code, tests, `config.json`, `PROTOCOL.md`) were hashed in `FREEZE_MANIFEST.json` before scientific-seed data existed; the runner refuses changed code or different Python/numpy/torch versions. Selections were hashed (`selections.sha256`) before any training or test data were generated. Outputs: `runs/scientific_01/`, audited in `analysis/`. This is the only source of the scientific claims above.

**(b) Engineering checks.** The unit tests (59), the technical smoke runs (`technical_smoke.json`, `runs_technical/smoke_02/`), the runtime benchmark (`technical_benchmark.json`) and the superseded pre-clarification versions (`runs_technical/superseded_v1/`, `runs_technical/smoke_01/`) used separate technical seeds. They show that the code computes what it claims (Kronecker identities, gradients, isolation of test data, hashing, timing) and fit within the runtime budget. They do not test the hypothesis and are not scientific evidence for or against it. The implementer disclosed that some very early probe outputs were deleted before a preservation rule was issued, so the development archive is not complete (see `IMPLEMENTATION_REPORT.md`).

**(c) Literature and judgement commentary.** The novelty check, the interpretation in `RESULTS_TR.md`, and the post-result opinions of three AI chat assistants (application labels recorded in the project as Claude Chat "Fable 5.1 Extra", ChatGPT "Pro", and Antigravity "Gemini 4 Argon High") are judgement, not measurement. The reviews were based on the same results summary, not on independent audits of raw files. One reviewer's first reply contained invented numbers, which the project explicitly rejected. The decision rests on the frozen gates and verified losses, not on a vote.

## Limitations

- Synthetic, linear teachers only; nothing is claimed about nonlinear networks, language models, token routing or interpretability.
- Mild anisotropy (condition number 4). Nothing follows for strong anisotropy.
- 12 independent teacher blocks; no statistical power, coverage or significance claim.
- The reference is the best trained finite candidate, not a certified optimum. Technical-seed checks found optimization gaps of up to about 0.0017 Z, and the scientific run reports isotropic population gaps up to 0.0072 Z (finite-sample plus optimization), so very small regret differences should not be over-read.
- With 64 pilot examples in 64 dimensions the gradient estimate is noisy; the half-pilot agreement (16 of 24) reflects this.
- The early-training baseline uses an equal allotted ceiling, not equal consumed compute.
- The selected-architecture space never had E4 as a winner.
- Parameters are equalized across candidates; active compute is not.

## Repository map

| Path | Contents |
|---|---|
| `soe/` | Study code: Kronecker utilities, candidates, environment generator, selectors, early selector, training, evaluation, provenance, pipeline |
| `scripts/` | `run_pilot.py` (scientific runner), `freeze.py`, `technical_smoke.py`, `benchmark_runtime.py` |
| `tests/` | 59 unit tests (stdlib `unittest`) |
| `config.json`, `PROTOCOL.md` | Frozen configuration and protocol (hashed in the freeze manifest) |
| `FREEZE_MANIFEST.json` | SHA-256 of the 29 frozen files and the Python/numpy/torch versions |
| `runs/scientific_01/` | The frozen scientific run: `summary.json`, `env_table.csv`, `candidate_seed_losses.csv`, `timing.csv`, `selections.json`, `records.json`, headers and manifests |
| `runs/scientific_01/envs/` | 24 per-environment arrays (about 2.2 MB each): teacher matrices, input covariances, noise levels, train/val/test sufficient statistics, training trajectories, and the trained factors and dense matrices of all nine small candidate models for both seeds. Read by the audit |
| `runs/scientific_01/pilot/` | 24 pilot arrays (about 0.25 MB each): the raw 64-example pilot inputs and outputs per task, the only data the selectors saw |
| `analysis/` | `independent_audit.py` and its output `independent_audit.json`; `frozen_source.zip` (archive of the 29 frozen files plus the manifest) and its SHA-256 |
| `technical_smoke.json`, `technical_benchmark.json`, `runs_technical/` | Engineering checks on technical seeds (not scientific evidence) |
| `RESULTS_TR.md` | Final results and decision report (Turkish) |
| `IMPLEMENTATION_REPORT.md` | Implementer's report: files, tests, design objections, process disclosures |
| `docs/` | Implementation brief, pre-outcome clarifications, pre-run audit, English results packet |
| `PROJECT_NOTES.md` | The original project README, unchanged (internal working notes; its interpreter path is machine-specific) |

`PROTOCOL.md` still carries the header "PROVISIONAL — not frozen" because the file was frozen byte for byte as written; its hash in `FREEZE_MANIFEST.json` is what makes it binding.

Not included in this public copy: the array files of the technical smoke runs (`runs_technical/smoke_01/` and `smoke_02/`, `envs/*.npz` and `pilot/*.npz`, about 9.8 MB; same format as the scientific arrays, technical seeds only, regenerable with `technical_smoke.py`) an internal coordination log (`docs/coordination_log.md`, which contains private AI-chat session links), and the original proposal (`docs/original_proposal.txt`, a Turkish AI-assistant answer pasted verbatim into the project; its idea is summarised in this README and formalised in `PROTOCOL.md`). The `output_manifest.json` files of those smoke runs still list the omitted arrays, `RESULTS_TR.md` links to the omitted log, and `docs/implementation_brief.md` refers to both omitted files.

## Reproducing

Dependencies, from `FREEZE_MANIFEST.json` and `technical_benchmark.json`: Python 3.11.13, numpy 2.4.6, torch 2.6.0, and the standard library (`unittest`). No other packages are used. The original runs were on macOS 26.6.2 arm64 (Apple M2 Max), CPU, float64, one thread. There is no requirements file; use an environment with exactly these versions, because the scientific runner refuses any Python, numpy or torch version that differs from the manifest.

Paths: the project's own notes use a machine-specific interpreter path (`/Users/emre/.../.venv-stage1a-mps-py311/bin/python`). Replace it with your own `python` and run everything from the repository root. The code itself locates files relative to the repository (`scripts/_path.py`), so no edits are needed.

What can be checked from the included files alone, without re-running any training:

- `analysis/independent_audit.py` needs only numpy and imports no study code. Run as `python analysis/independent_audit.py . runs/scientific_01`, it re-hashes the 29 frozen files against the run header, re-hashes all 56 output files (including every `.npz`) against `output_manifest.json`, checks the selection hash, rebuilds every trained candidate's dense matrix from its saved Kronecker factors, recomputes all 432 test losses and the normalizer Z from the saved test statistics, and recomputes gate 2, gate 3 and the verdict in all five views against `summary.json`. It prints the result (the content of `analysis/independent_audit.json`) and writes nothing.
- The pilot arrays (`runs/scientific_01/pilot/envNN_pilot.npz`, members `X`, `Y`) are exactly what the selectors saw, so the selector scores and choices in `selections.json` can be recomputed independently from the formulas in `PROTOCOL.md`; no script for this is included.
- The early-training baseline's choices cannot be recomputed exactly from files, because they depend on how many rounds fitted into one second; `selections.json` records the round counts used.

Re-running from scratch:

```bash
# unit tests (about 5 s; the MPS test only runs where Apple MPS is available)
python -m unittest discover -s tests -t . -v

# engineering checks on technical seeds; outputs are never overwritten, so use new names
python scripts/benchmark_runtime.py --out technical_benchmark_rerun.json
python scripts/technical_smoke.py --out technical_smoke_rerun.json --run-dir runs_technical/smoke_rerun

# re-run the frozen scientific experiment into a NEW directory (the runner refuses existing ones)
python scripts/run_pilot.py --freeze FREEZE_MANIFEST.json --out runs/scientific_rerun

# independent audit of the included run, or of a new run directory
python analysis/independent_audit.py . runs/scientific_01
python analysis/independent_audit.py . runs/scientific_rerun
```

Notes:

- `run_pilot.py` rejects any changed, missing or extra `.py` file under `soe/`, `scripts/` or `tests/`, and any change to `config.json` or `PROTOCOL.md`. Do not edit or add files there if you want to use the original freeze manifest.
- The early-training baseline depends on wall-clock time, so its round counts, and possibly its choices, can differ slightly on other hardware. All timings will differ.
- A from-scratch re-run uses the same seeds, so teachers, data and analytic selections should be reproduced, but bit-identical floating-point results across different hardware or BLAS builds are not guaranteed; the included arrays are the recorded evidence.
- Some recorded files contain absolute paths from the original machine (`/Users/emre/...`): `runs/scientific_01/run_header.json` (the freeze manifest path), `IMPLEMENTATION_REPORT.md`, `docs/implementation_brief.md` and `PROJECT_NOTES.md`. They are kept unmodified as evidence; `run_header.json` is also covered by the hashes in `output_manifest.json`, so editing it would make the audit fail. These paths do not affect re-running.

## References

Sources cited in the project's own documents. The first four were checked by the project (titles, abstracts and relevant sections; not an exhaustive novelty search). The rest are cited in the original proposal (`docs/original_proposal.txt`, not included in this copy) and were not separately re-verified.

Checked by the project:

- TensorPoly: Mixture of Latent Experts Using Tensor Products (2024). https://arxiv.org/abs/2405.16671
- MoKA (COLING 2025), Kronecker sparse mixture of experts. https://aclanthology.org/2025.coling-main.679/
- CDKA, Diving into Kronecker Adapters: Component Design Matters (ICML 2026). https://arxiv.org/abs/2602.01267 (checked version: https://arxiv.org/html/2602.01267v3)
- DR-LoRA (COLM 2026), rank allocation among expert LoRAs from routing and gradient importance. https://arxiv.org/abs/2601.04823 (checked version: https://arxiv.org/html/2601.04823v6)

Cited in the original proposal only:

- MoKA: Mixture of Kronecker Adapters (2025). https://arxiv.org/abs/2508.03527
- MAdaKron (2026). https://www.sciencedirect.com/science/article/abs/pii/S0950705125021240
- Scaling Laws for Fine-Grained Mixture of Experts (ICML 2024). https://proceedings.mlr.press/v235/ludziejewski24a.html
- Cornell eCommons record cited for the standard best-Kronecker-approximation result. https://ecommons.cornell.edu/entities/publication/ad6a5c71-2632-4652-b33a-72016d7abc4d
- arXiv paper cited for the link between tensor separability and network expressivity. https://arxiv.org/abs/1704.01552
- ACL Anthology paper cited for limits of tensor parametrizations in dense and MoE models. https://aclanthology.org/2026.emnlp-main.2679/
- GLUE benchmark (proposed Stage B data). https://arxiv.org/abs/1804.07461
- Qwen2.5-0.5B-Instruct configuration (proposed Stage B model). https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/raw/main/config.json

## Provenance and license

- The study was run at the request of, and under the authorization of, the repository owner.
- Implementation and tests were written by Claude Code (application label "Opus 5.5 High"), as stated in `IMPLEMENTATION_REPORT.md` and `RESULTS_TR.md`.
- A separate coordinating AI agent audited the source, froze it, ran the scientific experiment and wrote the independent audit; the freeze manifest records the approver as "Codex coordinator: user-authorized pilot". According to the project records, the coordinator did not author or modify the scientific implementation or test code; it corrected two prose statements about convergence in `PROTOCOL.md` before the freeze.
- The design was critiqued before the run, and the results reviewed after it, by three AI chat assistants (see "Three kinds of evidence" above).
- All experiments ran locally on one Apple M2 Max machine on 2026-10-31.

No license file is included; all rights reserved by the repository owner unless stated otherwise. Contact the owner before reuse.
