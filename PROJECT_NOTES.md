# Split or Enrich — Stage A synthetic pilot

Completed 2026-10-31: **NO_ADDED_UTILITY**, large LLM Stage B **NO-GO**. All 24 environments completed in 112.50 s; 59 tests passed. Read [Turkish result and decision](RESULTS_TR.md), [main summary](runs/scientific_01/summary.json), and [independent audit](analysis/independent_audit.json). The scientific output folder already exists; never overwrite or rerun it as if unseen. Original pre-run implementation instructions follow.

Pilot-gradient selector for expert count vs Kronecker capacity, tested on synthetic linear teachers.
Read `PROTOCOL.md` (definitions, gates, verdicts, open decisions) and `IMPLEMENTATION_REPORT.md`.

Python: `PY=/Users/emre/Documents/Antropic_Mech_Int/.venv-stage1a-mps-py311/bin/python` (numpy, torch; stdlib unittest).
Run everything from this folder.

```bash
cd /Users/emre/Documents/Antropic_Mech_Int/split_or_enrich
```

Unit tests (59 tests, ~5 s; the MPS test runs only when MPS is visible to the process):

```bash
$PY -m unittest discover -s tests -t . -v
```

Technical runtime benchmark (technical seed root only; refuses to overwrite):

```bash
$PY scripts/benchmark_runtime.py --out technical_benchmark.json
```

Technical smoke (technical seeds: 2-environment end-to-end run, optimization check, design diagnostic):

```bash
$PY scripts/technical_smoke.py --out technical_smoke.json --run-dir runs_technical/smoke_02
```

Coordinator only, after final approval — freeze, then the scientific run (24 environments, ~3 min expected incl. ~2x1 s early search per environment, 60 min cap):

```bash
$PY scripts/freeze.py --approver "<coordinator approval reference>" --out FREEZE_MANIFEST.json
```

```bash
$PY scripts/run_pilot.py --freeze FREEZE_MANIFEST.json --out runs/scientific_01
```

`run_pilot.py` rejects any changed, missing or extra file under `soe/`, `scripts/`, `tests/`, plus
`config.json` / `PROTOCOL.md`, and any python/numpy/torch version change. It refuses an existing output
directory. Results: `runs/scientific_01/summary.json` (verdict), `env_table.csv`, `candidate_seed_losses.csv`,
`timing.csv`, `selections.json` (+ `.sha256`, written before training), `envs/*.npz`, `output_manifest.json`.

Layout: `soe/kron.py` (conventions, rearrangement, factorized forward), `soe/candidates.py`,
`soe/envgen.py` (teachers, covariances, RNG streams, PilotView), `soe/selectors.py`, `soe/early.py`,
`soe/train.py` (moment backend, fused Adam), `soe/evaluate.py` (regret, gates, verdicts),
`soe/provenance.py` (hash lock, exclusive writes), `soe/pipeline.py` (phase order).

Outputs are never overwritten: rerunning a technical script needs a new `--out` / `--run-dir`. Superseded
pre-clarification technical outputs are kept in `runs_technical/superseded_v1/` and `runs_technical/smoke_01/`.
