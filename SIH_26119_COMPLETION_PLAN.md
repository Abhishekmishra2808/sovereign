# SIH 26119 completion plan

Status: active plan, 2026-09-27. This maps the supplied MRPL problem statement to
work that can be implemented and verified. "Done" means recorded evidence meets a
gate below, not that a feature name exists in source. Keep the C++ optimization
algorithms original: HiGHS or another established solver is a benchmark oracle
only, never the production solve implementation.

The user also requires an online Render website with outbound CPU/GPU connectors.
That delivery requirement is included below, though the SIH statement itself says
a CLI or basic API is sufficient and does not require a polished interface.

## Current baseline

- Implemented: C++ LP (revised simplex and interior point), convex QP (interior
  point and Frank-Wolfe), MILP (branch-and-bound/cut, presolve, heuristics), MPS
  and JSON input, CLI, HTTP coordinator, and outbound CPU/CUDA workers.
- Verified end-to-end: CPU and CUDA jobs pass through the website queue to this
  PC and return stored results. A 40,000-variable transport LP returned matching
  verified CPU/CUDA objectives. CUDA kernels ran, but did not speed up the solve.
- SIH corpus: 43 CPU algorithm runs, 23 verified optimum matches, 9 numerical
  errors, 10 timeouts, 1 verified feasible point. In the 3-second run, no tested
  MIPLIB profile produced a verified optimum. Two MIPLIB jobs also timed out at
  20 seconds in the PC round-trip test.
- Scale evidence ends at 40,000 variables and 400 constraints. This is not
  million-variable or million-constraint evidence. Current JSON job submission
  is capped at 5 MB, and core LP/QP paths still build dense systems.
- Dataset coverage: one Netlib LP, five official MIPLIB instances, small
  synthetic robustness/QP/industrial-style examples. No recorded Mittelmann
  or QPLIB run and no published or MRPL-supplied industrial case.
- Deployment: Render configuration exists; a real Render service and remote
  company-server connection have not been exercised.

Evidence:
[SIH corpus](benchmarks/reports/sih-online.md) |
[PC and CUDA responses](benchmarks/reports/hard-pc-roundtrip.md) |
[full job payloads](benchmarks/reports/hard-pc-roundtrip.json) |
[GPU kernel timings](benchmarks/reports/gpu-spmv.txt).

## Completion matrix

| SIH requirement | Current state | Completion gate |
| --- | --- | --- |
| Sovereign LP/MILP/QP algorithms | Initial implementations run | Curated LP/QP/MILP corpus meets the accuracy, status, and time gates below, with no production dependency on an existing solver |
| Numerical robustness | Small synthetic cases pass; MIPLIB node relaxations and some simplex cases fail | No unexplained numerical errors on the acceptance corpus; every remaining failure has a minimized reproducer and a truthful non-optimal status |
| Sparse linear algebra and multi-core scale | Sparse storage exists, but dense LU/normal equations/KKT and mostly serial MILP search dominate | Sparse memory/factorization and controlled parallel search show measured improvements without changing answers |
| Thousands to millions of variables and constraints | 40k variables / 400 rows demonstrated | Reproducible 10k, 100k, and 1m-scale sparse workloads complete within declared machine, memory, and time budgets; report which problem classes actually reach each scale |
| GPU where beneficial | Real CUDA SpMV verified, but slower end-to-end | Correct CPU/GPU agreement and a repeated, statistically defensible total-solve speedup on at least one representative workload; automatic routing uses measured crossover points |
| Recognised benchmarks | AFIRO and five MIPLIB cases attempted | Pre-registered Netlib, MIPLIB, Mittelmann, and applicable convex QPLIB suite run with provenance, reference objective/bounds, repeatable budgets, and failures shown |
| Industrial relevance | Repository examples are synthetic | At least one fully documented open-literature case each for blending, refinery/production, power dispatch, and logistics/supply chain; no proprietary MRPL claims without permission/data |
| CLI/API and trustworthy output | CLI and online API work | Primal, objective, dual/bound/gap, status, machine/device and verification are consistent; unsupported or interrupted runs never masquerade as OPTIMAL |
| Online delivery requested by user | Local preview and Render Blueprint work in source | Real Render HTTPS deployment, persistent data, remote CPU and CUDA connectors, recovery and access controls pass deployment smoke |

MIQP, NLP and MINLP are **future extension points**, not an initial completion
gate in this SIH statement. Their interfaces should not force a rewrite of the
LP/MILP/QP core, but no unsupported class should be advertised as solved.

## Work packages and order

### 0. Freeze an honest benchmark baseline [in progress]

- [x] Preserve full HTTP job responses, verifier output, reference values,
  machine/GPU identity, and CUDA operation counts for current tests.
- [ ] Version a corpus manifest: official download URL, license, checksum,
  original format, dimensions, integer count, known objective/bounds, and
  expected status for each instance. Keep generated models and seeds separate.
- [ ] Fix budgets before comparing implementations: CPU thread count, CUDA
  model, memory ceiling, per-class wall-clock cap, repeated cold/warm trials,
  and tolerance. Record median and spread, not only the best run.
- [ ] Build an automated result table with status, primal validity, objective
  error, dual/best bound, MIP gap, nodes, peak memory, wall time and failure
  reason. Publish all attempted rows, including timeouts.
- Exit: one command reruns the baseline and reproduces a machine-readable
  report. A result without a saved input hash or verifier status is incomplete.

### 1. Make status claims independently checkable [first correctness gate]

- [ ] Extend solver results with LP/QP dual vectors and reduced costs, MILP
  incumbent and global bound provenance, and explicit feasible/optimal flags.
- [ ] Check primal constraints, bounds, integrality, objective and dual
  feasibility with scaled tolerances; recompute gap from independently
  supplied primal/dual data where possible.
- [ ] Require a Farkas certificate for INFEASIBLE and a valid improving ray
  for UNBOUNDED. The current verifier accepts those labels without proof.
- [ ] Check presolve/postsolve mappings and cut validity against original
  models. Wrong reconstruction must downgrade the result.
- [ ] Add adversarial regressions: inconsistent bounds, rank deficiency,
  near-zero pivots, duplicate rows, large coefficient ratios, cancellation,
  and misleading solver statuses.
- Exit: no false OPTIMAL/INFEASIBLE/UNBOUNDED result in unit, differential,
  and randomized tests. Any unproved claim becomes FEASIBLE/UNKNOWN/ERROR
  with a recorded reason. The coordinator retains failed verification.

### 2. Repair numerical LP/QP foundations [blocking scale and MILP]

- [ ] Capture and minimize failing node LPs from flugpl, gt2, pk1,
  gen-ip016, industrial logistics, and the refinery simplex case. Create one
  regression per distinct root cause before changing algorithms.
- [ ] Replace dense basis extraction/LU and dense normal-equation/KKT
  factorization with sparse symbolic/numeric factorization or a justified
  sparse iterative method implemented in the core. Measure fill and memory.
- [ ] Add scaling/equilibration, rank-revealing handling, regularization,
  iterative refinement and residual checks. Keep original-model objective
  and feasibility checks authoritative after unscaling.
- [ ] Reuse factorization and warm starts across simplex pivots and MILP child
  relaxations. Avoid rebuilding full models for bound-only node changes.
- [ ] Treat convexity explicitly for QP. Validate Hessian symmetry/PSD and
  return UNSUPPORTED for nonconvex QP until that class is implemented.
- Exit: all current small LP/QP canaries retain correct answers; previously
  failing minimized node LPs produce a trustworthy answer or a precise,
  reproducible non-optimal status. A first wider Netlib/QPLIB slice improves
  without regressions.

### 3. Make MILP search reliable before making it faster

- [ ] Never discard a failed node LP and then certify the tree. Repair
  relaxation failures through the numerical work above; preserve a node for
  fallback/retry when feasible.
- [ ] Audit global best-bound bookkeeping, open-node accounting, integrality
  tolerances, incumbent validation and MIP gap. Separate FEASIBLE from
  certified OPTIMAL in all API and UI outputs.
- [ ] Validate every cut mathematically and by randomized feasible-point
  checks; benchmark branch-and-bound with cuts disabled as a correctness
  reference. Add root separation only after validity is established.
- [ ] Improve primal heuristics, reliability/pseudocost branching, node
  selection and warm starts; then add bounded multi-core node processing
  with deterministic one-thread replay for debugging.
- Exit: selected MIPLIB and industrial MILP models produce verified
  incumbents and trustworthy bounds; no incorrect optimum claim. Record
  objective gap and time against an established reference solver.

### 4. Demonstrate actual large sparse scale

- [ ] Move model storage to indexed, compact sparse arrays with 64-bit
  counts where required. Bound memory per nonzero and avoid copying a whole
  model into every MILP node.
- [ ] Add streaming MPS/model upload and external object storage or chunked
  transfer for the online service. The current 5 MB request cap cannot carry
  million-scale models. Worker downloads must check hashes and support
  interruption/retry.
- [ ] Add cancel/deadline checks inside numerical iterations and search,
  peak-memory instrumentation and controlled out-of-memory results.
- [ ] Run a scale ladder: 10k, 100k, then 1m variables and a separate
  1m-constraint sparse instance. Include LP, a structured convex QP, and
  MILP at sizes each class can meaningfully solve; never infer MILP capacity
  from a transport LP.
- Exit: each claimed scale has a full model hash, worker hardware/memory,
  status, verifier/certificate, reference or bound, wall time and peak memory.
  A failed million-scale run remains a documented gap, not a completed gate.

### 5. Turn CUDA execution into measured acceleration

- [x] Build a CUDA worker and demonstrate actual kernel calls with matching
  verified CPU/GPU answers through the online job path.
- [ ] Profile total solve time and kernel/copy/host-factorization time on
  multiple representative sparse sizes. Current SpMV moves and converts the
  matrix for every call and is slower than CPU.
- [ ] Reuse device buffers and sparse structure; benchmark stable CSR/CSC
  layouts and batched operations. Only port additional linear algebra when
  profiling shows it dominates and the CPU fallback remains correct.
- [ ] Repeat paired CPU/GPU runs with identical input, tolerances, threads and
  machine. Test numerical agreement and GPU memory exhaustion/fallback.
- [ ] Calibrate automatic routing from measured total-solve crossover, with
  explicit CUDA choice still available. Never label a CUDA-capable worker as
  GPU-used when `gpu_operations=0`.
- Exit: at least one named industrial or recognised benchmark workload has
  repeated median total-solve speedup of at least 1.2x with equal verified
  solution quality. If that cannot be demonstrated, report partial CUDA
  support and keep automatic execution on CPU for the measured cases.

### 6. Complete SIH benchmark and industrial evidence

- [ ] Add a preselected, varied Netlib LP slice (degenerate, ill-conditioned,
  sparse and large), MIPLIB instances beyond the five currently attempted,
  suitable Mittelmann cases and convex QPLIB cases. Record exclusions and
  why each is outside current LP/MILP/convex-QP scope.
- [ ] Compare against HiGHS or another established solver under declared
  settings. HiGHS stays in the benchmark harness; no production path calls it.
- [ ] Include published/open-literature refinery scheduling, crude blending,
  production planning, power dispatch, transport and logistics formulations.
  Check equations, units, objective and constraints independently; mark
  repository-generated data as synthetic.
- [ ] Run degeneracy, weak LP relaxation and ill-conditioned stress cases at
  non-toy sizes, including coefficient perturbations and repeated runs.
- Exit: the selected corpus and industrial cases have a public manifest,
  objective/gap correctness, comparable runtime/memory and honest failure
  table. Do not call the solver "industrial-scale" based only on synthetic
  transport models or an unverified incumbent.

### 7. Deliver the online product without weakening the solver evidence

- [ ] Deploy `render.yaml` to the user's Render account; smoke-check HTTPS,
  durable database, worker pairing, job/result persistence after restart,
  key rotation, revocation, timeout and lost-lease recovery.
- [ ] Run at least one non-loopback CPU worker and one non-loopback CUDA worker
  on a company/server-class machine. Save request ID, machine identity,
  verification and actual GPU operation count.
- [ ] Add database backup/restore, model/result retention, upload size limits
  that match the scale target, request rate limits and a clear trust boundary
  for worker access to model data.
- [ ] If more than one customer/team will use it, add per-user/workspace
  isolation before accepting their data. The present service is one private
  workspace, not a multi-tenant platform.
- Exit: an independent operator can deploy from written instructions, submit
  an acceptance-corpus model from a browser, receive a verified result, and
  reproduce the deployment smoke. A local preview does not satisfy this gate.

## Final acceptance and claims

Before a final SIH demonstration, freeze the selected suite and run it on a
documented machine with fixed budgets. Target **zero false proof claims** on
every test. For LP/convex QP, target at least 95% verified optimum results on
the selected applicable corpus; for MILP, target at least 90% verified feasible
solutions with a reported bound/gap and at least 80% at a 1% gap or better
within the declared budget. These are proposed engineering gates, not current
scores or guarantees; revise them before the final run if the evaluators
publish stricter criteria. The title "GPU-accelerated" requires the measured
total-solve speedup gate above, not merely a nonzero CUDA kernel count.

Evidence bundle: source revision; corpus manifest/hashes; compiler and CUDA
versions; exact commands; hardware and thread settings; raw solver and
reference responses; verifier/certificate output; timing/memory CSV; failures;
and a concise comparison report. Keep unsuccessful attempts in that bundle.

The critical path starts with work packages 0-2. The first implementation
batch should minimize one MIPLIB failing node LP, add its regression, correct
that numerical failure, and rerun that model plus the existing LP/QP canaries.
Only then expand MILP search, scale, and GPU optimization.

Progress, 2026-09-27: captured and regression-tested one infeasible `flugpl`
node LP. Node-level presolve now classifies it before IPM/simplex; the focused
100-node run has no node-LP failure warning, while the full MILP remains
unsolved at that limit. The verifier now rejects `INFEASIBLE`/`UNBOUNDED`
without certificates, removing a false verification claim while certificate
generation remains open. See `benchmarks/reports/flugpl-node-presolve.md`.

GPU measurement progress: five paired measured CPU/CUDA trials plus a warmup
on each of the 10k, 22.5k and 40k-variable synthetic transport LPs all passed
verification and objective parity. Median whole-job CUDA speedup was 0.829x,
0.866x and 0.946x respectively, so automatic routing now defaults to CPU;
explicit CUDA remains available. Full timings and responses are in
`benchmarks/reports/gpu-paired.md` and the adjacent JSON/gzip files. This does
not satisfy the acceleration exit gate. An exact-content, per-thread CUDA
matrix/buffer cache was implemented and retested; median ratios remain below
1x. Kernel/copy profiling and larger representative workloads remain next.

Corpus evidence progress: `benchmarks/reports/corpus-manifest.json` now fixes
20 current inputs by SHA-256 and records dimensions, origin and the measured
HiGHS reference status/value where available. The `--check` command verifies
input integrity. Licensing and official known-bound metadata still need
review before a release corpus is frozen.

Explicit CUDA usability progress: the website no longer labels an Automatic
CPU-routed job as a missed GPU run. A CUDA-requested job that presolve finishes
without kernels gets one bounded retry with presolve off; zero-kernel results
after that are failed with a clear reason and the original answer retained.
The real coordinator/worker/solver round trip, including a 14-operation retry,
is recorded in `benchmarks/reports/cuda-selection-roundtrip.md`.
