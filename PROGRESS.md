# Sovereign evidence and hardening progress

This ledger is updated at the end of every requested phase. A phase is not
considered complete until its tree is committed and tagged. Benchmark results
are accepted only when the independent verifier passes on the original model;
timeouts, errors, missing inputs, and unverified statuses remain failures.

## Phase 0 — baseline

### Status

Complete as a preservation/inventory baseline. No solver code was changed in
this phase. The starting repository commit was `bec7497` (`Complete MIPLIB
Stage 2 diagnostics and sweep`). The starting tree was dirty and contained
both tracked report edits and untracked raw benchmark outputs/logs.

The baseline snapshot intentionally preserves the existing current state,
including generated evidence and failed/partial runs. It is not a claim that
all artifacts already satisfy the later phase protocol.

### Existing benchmark harnesses

Runners under `benchmarks/runners/`:

- `build_corpus_manifest.py` — corpus metadata and hash manifest generation.
- `coverage_report.py` — broad coverage report generation.
- `run_benchmarks.py` — legacy smoke/Netlib/robustness/scale suite runner.
- `run_coverage.py` — Netlib, MIPLIB, infeasible-corpus, and Maros-Meszaros
  coverage runner with HiGHS comparison and independent primal checks.
- `run_evidence_pack.py` — headline evidence-pack regeneration.
- `run_extended_lp.py` — Kennington/Mittelmann showcase runner.
- `run_gpu_paired.py` — paired CPU/CUDA timing harness.
- `run_miplib_official.py` — earlier five-instance official MIPLIB runner.
- `run_miplib_stage2.py` — locked 30-instance official MIPLIB Stage 2 runner.
- `run_netlib_sweep.py` — Netlib original-model KKT/certificate sweep.
- `run_robustness.py` — robustness property and ablation runner.
- `run_scale_ladder.py` — scale ladder runner.
- `run_showcase.py` — industrial/showcase model runner.
- `run_sih_online.py` — online SIH corpus runner.

Tools under `benchmarks/tools/`:

- Dataset/conversion: `fetch_coverage_corpus.py`, `mps_to_json.py`,
  `compare_mps_parse.py`, `check_offline_assets.py`.
- LISWET/QP: `run_liswet1.py`, `run_liswet_step6.py`,
  `verify_liswet_independent.py`, `verify_liswet_pycutest.py`,
  `run_pycutest_container.sh`, `liswet_small_reference.py`,
  `diagnose_liswet_paths.py`, `crosscheck_liswet_real.py`,
  `crosscheck_liswet_nnls.py`, `compare_liswet_c_sources.py`,
  `compare_liswet_x_sources.py`, `highs_qps_crosscheck.py`.
- Synthetic/scale/GPU generation and timing:
  `generate_datasets.py`, `generate_scale_ladder.py`,
  `generate_sparse_scale.py`, `generate_large_scale_smoke.py`,
  `generate_gpu_showcase.py`, `time_transport.py`.
- Site/support: `fetch_fonts.py`, `fetch_site_fonts.py`.

### Existing manifests and input families

- `benchmarks/manifests/miplib_stage2.json` — official MIPLIB 2017 Stage 2
  selection of 30 names and official instance/solution URLs.
- `benchmarks/reports/corpus-manifest.json` — existing 20-input corpus lock.
- `benchmarks/datasets/catalog.json` — repository dataset catalog.
- `benchmarks/data/LISWET1.SIF` — official LISWET1 SIF input.
- `benchmarks/datasets/netlib/` — tracked AFIRO MPS/JSON and FINNIS MPS.
- `benchmarks/datasets/miplib/official/` — tracked official MIPLIB sample
  MPS/JSON pairs and additional MPS inputs.
- `benchmarks/datasets/miplib/` — synthetic knapsack and set-partition
  fixtures, labeled as synthetic in the existing evidence.
- `benchmarks/datasets/robustness/` — synthetic degeneracy,
  ill-conditioning, and weak-relaxation fixtures.
- `benchmarks/datasets/scale/` — synthetic transportation fixtures.
- `benchmarks/datasets/sparse/` and `benchmarks/datasets/gpu/` — synthetic
  sparse/GP​​U/QP fixtures.
- `benchmarks/datasets/coverage/` — fetched coverage corpus location,
  currently ignored by Git; missing files must be fetched from the public URLs
  recorded by the harness rather than replaced.

### Existing reports and raw artifacts

- `benchmarks/reports/EVIDENCE.md` — earlier headline LP/MILP/QP evidence
  pack, including explicit synthetic labels.
- `benchmarks/reports/HONESTY.md` — claim limitations and data provenance.
- `benchmarks/reports/COVERAGE.md` — broad Netlib/MIPLIB/infeasible/QP
  coverage report and failure tables.
- `benchmarks/reports/coverage/` — suite JSONL, metadata, CSVs, and
  performance profiles for Netlib, MIPLIB, MIPLIB infeasible, and QP.
- `benchmarks/reports/miplib_stage2/` — 30-instance official MIPLIB corpus
  lock, results, raw Sovereign/HiGHS logs, profile, and `FINAL_REPORT.md`.
- `benchmarks/reports/liswet_real/` and `liswet_real_cutest/` — official
  LISWET1 runs, independent outputs, source comparisons, and cross-checks.
- `benchmarks/reports/liswet_step6/` — synthetic scalability and NNLS
  experiments at several dimensions.
- `benchmarks/reports/robustness/` — synthetic robustness properties and
  ablation JSONL.
- `benchmarks/reports/showcase/` — showcase/extended-LP JSONL and logs.
- `benchmarks/reports/scale-ladder.md` and
  `scale-lp-1m-result.json` — scale evidence, including synthetic runs.
- `benchmarks/reports/gpu-*.{md,json,jsonl.gz}` and
  `cuda-selection-roundtrip.*` — paired GPU and online CUDA evidence.
- `benchmarks/reports/sih-online.*` and `hard-pc-roundtrip.*` — online
  coordinator/worker evidence.
- `benchmarks/reports/latest.csv`, `mps_audit.json`, and
  `flugpl-node-presolve.*` — earlier summary and diagnostic artifacts.
- `results/netlib.csv`, `results/netlib.md`, and
  `results/netlib-performance-profile.png` — expanded Netlib result table.
- `results/netlib-raw/` — per-instance primary/certificate/record JSON and
  stderr logs for the available Netlib corpus.
- `results/netlib-triage/` — presolve-off and targeted Netlib diagnostics.
- `results/netlib-history/` — preserved pre-baseline Netlib report/CSV.

The baseline also contains existing untracked solver/build logs and the
generated explainer video. They are preserved as part of the current-state
snapshot; they are not benchmark inputs and do not substitute for missing
evidence.

### Baseline gaps carried into Phase 1

- No debug-solution mode is present in the baseline inventory.
- The Stage 2 report records `qnet1_o` with an invalid best-bound check and
  reports the finnis temporary-bound concern; these are Phase 1 work items.
- The existing MIPLIB Stage 2 report was produced with a dirty working tree
  and does not yet satisfy the final artifact metadata requirements.
- The existing infeasible coverage is a MIPLIB infeasible corpus, not the
  requested Netlib infeasible set; no Netlib infeasible set is invented here.
- Extended-LP, QP, scale, and industrial-demo evidence exists only in the
  listed partial forms and must be re-audited against the requested gates.
- No final evidence report or dependency-audit result exists yet.

### Commit and next step

- Phase 0 preservation commit: recorded after this ledger was added.
- Phase 0 progress-record commit: recorded after the preservation commit.
- Tag: `phase0-baseline`.
- Next step: Phase 1 soundness audit and fixes. The Phase 1 gate is zero
  debug-solution violations and zero invalid bounds across the requested
  30-instance reference-solution MIPLIB run.

## Phase 1 — soundness

Not started.

## Phase 2 — MIPLIB improvement

Not started.

## Phase 3 — extended LP

Not started.

## Phase 4 — Netlib infeasible set

Not started.

## Phase 5 — QP evidence

Not started.

## Phase 6 — scale and real-world problems

Not started.

## Phase 7 — compliance and final report

Not started.
