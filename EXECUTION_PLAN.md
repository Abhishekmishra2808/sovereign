# Sovereign online implementation plan

Updated 2026-09-27. This is the plan of record for the requested online website.
The earlier local/offline website plans in this repository are superseded. A compute
worker may run on a laptop or company server; the website itself runs on Render.

For the complete SIH 26119 requirement matrix, engineering work packages, and
measurable acceptance gates, see [SIH_26119_COMPLETION_PLAN.md](SIH_26119_COMPLETION_PLAN.md).

## Product contract

- Render serves one private workspace, the dashboard, the job API and a durable
  queue. It does not execute optimization jobs.
- A machine with the Sovereign C++ executable runs an outbound-only Python worker
  over HTTPS. Workers receive model data, run the solver and return results.
- Jobs support LP, MILP and QP; JSON and MPS; explicit CPU/CUDA and automatic
  routing; algorithm, presolve and search controls.
- An OPTIMAL/FEASIBLE result must include a passing verification report before
  the website marks the job complete. GPU use is reported from actual kernel
  calls, not inferred from an NVIDIA device being present.
- HiGHS appears only in the benchmark harness as a reference solver.

## Implementation status

1. [x] Render coordinator and dashboard: Dockerfile, Blueprint, same-origin API,
   login, SQLite persistence, worker pairing/revocation and job queue.
2. [x] Outbound worker: HTTPS-only remote connections, separate worker token,
   capability report, leases, retries, cancellation heartbeat, time limit,
   subprocess isolation and result upload.
3. [x] Solver choices: LP simplex/interior point, QP interior point/Frank-Wolfe,
   MILP branch-and-cut/bound, branching rule, presolve and node limit exposed in
   the website and forwarded to the C++ executable.
4. [x] CPU/GPU routing: automatic jobs currently use CPU because repeated
   whole-solver trials have not shown a CUDA speedup on tested workloads.
   Explicit CUDA jobs wait for CUDA capability, and the previous size-based
   policy is opt-in for experiments. Actual GPU operation count is returned.
5. [x] C++ CUDA sparse matrix-vector kernel integrated into sparse multiplication
   when built with SOVEREIGN_USE_CUDA=ON. This is partial acceleration, not a
   full GPU LP/MILP/QP engine.
6. [x] HTTP end-to-end smoke for LP, MILP, QP, MPS and the dashboard example.
   Unit and coordinator contract tests pass on the current CPU build.
7. [x] SIH benchmark corpus run through the actual coordinator/worker. Records
   official Netlib/MIPLIB examples, QP, industrial-style synthetic cases,
   algorithm profiles, HiGHS reference, failures and timeouts. Current CPU
   run: 43 comparisons, 23 verified optimum matches, 10 timeouts, 9 numerical
   errors and 1 verified feasible result. The solver does not yet pass every instance.
8. [x] NVIDIA CUDA Toolkit 13.4.2 installed on the RTX 5050 PC. A CUDA build
   and outbound GPU worker are running. Live HTTP jobs with 10,000, 22,500,
   and 40,000 variables produced verified CPU/GPU-matching optima and reported
   26-29 actual GPU operations. See hard-pc-roundtrip.md for full responses.
   These single runs showed no GPU speedup; dense CPU factorization remains.
9. [ ] Deploy the Blueprint to the user's Render account and pair a real
   worker. Requires account/repository access and the resulting Render URL;
   repository files alone are not a live website.
10. [ ] Improve solver reliability and scale where benchmarks show numerical
    errors, timeouts or objective mismatches. Re-run identical corpus and
    publish honest before/after measurements.
11. [~] Optimize and validate GPU performance: five paired runs per tested
    transport size are captured in `benchmarks/reports/gpu-paired.md`.
    A per-thread device matrix cache avoids repeated structure upload but
    still does not improve total solve time. Kernel/transfer profiling and
    dominant CPU factorization work remain open.
12. [ ] Production hardening: database backup/restore test, rate limits and
    model/result retention, stronger per-user access if multiple users need
    isolation, and deployment smoke against the real Render URL.

## Current verification

- `npm run build` passes.
- `npm run lint` passes.
- `ctest --test-dir build-cloud -C Release --output-on-failure` passes.
- `python -m unittest discover -s tests/scripts -p test_*.py` passes.
- `python tests/scripts/smoke_cloud.py --engine build-cloud/solver/Release/sovereign.exe`
  passes LP, MILP, QP, MPS, example and static asset checks.
- The SIH corpus report is `benchmarks/reports/sih-online.md` (CPU-only).
  The live PC/GPU report is `benchmarks/reports/hard-pc-roundtrip.md`; full
  captured job responses are in the adjacent JSON file.

## Deployment sequence

Push the repository, create the Render Blueprint from `render.yaml`, check
`/api/health`, sign in with the generated workspace key, build the C++ engine
on each compute machine, pair it in Machines, start `worker/runner.py`, submit
the example job, then run a representative customer model. Exact steps are in
`DEPLOYMENT.md`.
