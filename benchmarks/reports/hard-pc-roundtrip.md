# Hard PC worker round trip

Initial run: 2026-09-26T23:28:28.052666+00:00

Updated with CUDA results: 2026-09-26T23:45:20.885493+00:00

Jobs were submitted to the preview website and claimed by separate CPU and CUDA connectors on this PC. The complete API responses, including primal vectors, verification and routing, are in `hard-pc-roundtrip.json`.

| Model | Device | Job ID | State | Solver status | Objective | Verified | GPU operations | Solver seconds |
|---|---|---|---|---|---:|---|---:|---:|
| transport_100x100 | CPU | `253e5bdf8086fb8d8f70ebce` | COMPLETED | OPTIMAL | 1009.0000000400709 | True | 0 | 0.2100697 |
| flugpl | CPU | `d502084be51e3e271ef901da` | FAILED | none | none | False | 0 | time limit |
| gt2 | CPU | `10bf1c2962a0908dc18d5d61` | FAILED | none | none | False | 0 | time limit |
| industrial_logistics_milp | CPU | `f0aca306b0572543d7683206` | COMPLETED | FEASIBLE | 550.0000000003491 | True | 0 | 0.000901 |
| transport_100x100 | CUDA | `faf4eff50398584be556f42a` | COMPLETED | OPTIMAL | 1009.0000000400709 | True | 26 | 0.394584 |
| transport_150x150 | CPU | `708509613595780ab8a6b5da` | COMPLETED | OPTIMAL | 1513.4000004519796 | True | 0 | 0.7496997 |
| transport_150x150 | CUDA | `c928c5dea4e69246e6fd1d7e` | COMPLETED | OPTIMAL | 1513.4000004519796 | True | 26 | 0.8026649 |
| transport_200x200 | CPU | `cac3e538821fcc22b4d4be75` | COMPLETED | OPTIMAL | 2017.80000000996 | True | 0 | 2.2164813 |
| transport_200x200 | CUDA | `53f9fbc768dc1fed80f5a62d` | COMPLETED | OPTIMAL | 2017.80000000996 | True | 29 | 2.243216 |

| industrial_logistics_milp | CUDA | `2de6a4fd647b5f5f2aa61003` | COMPLETED | FEASIBLE | 550.0000000003488 | True | 348 | 0.2696061 |

## Findings

The 10,000-, 22,500- and 40,000-variable transport LPs returned identical CPU and CUDA objectives and passed solution verification. HiGHS independently found optima of 1513.4 and 2017.8 for the two larger models, matching Sovereign within numerical tolerance. The CUDA responses report 26, 26 and 29 GPU sparse matrix-vector operations respectively.

On this single-run comparison, CUDA solver time was 0.395 vs 0.210 seconds at 10,000 variables; 0.803 vs 0.750 at 22,500; and 2.243 vs 2.216 at 40,000. These single runs do not establish a GPU speedup. HiGHS reference times for the two larger cases were 0.097 and 0.298 seconds with one CPU thread; it remained faster than Sovereign. Dense CPU factorization remains a major part of the solve, and the CUDA kernel currently converts and copies its sparse matrix on each call.

A standalone CUDA SpMV check compared CPU and GPU output on random sparse matrices (10,000, 100,000 and 1,250,000 nonzeros); maximum differences were below 1e-14. It also measured the current GPU path at 16-262 times slower than CPU because it converts the matrix and allocates/copies device buffers on every call. Raw timings are in `gpu-spmv.txt`. These are single-process timings, not repeated statistical benchmarks.

The initial CUDA queue check happened before CUDA Toolkit installation. That job correctly remained queued with only the CPU connector and was cancelled. After installing CUDA Toolkit 13.4.2 and rebuilding the solver, the CUDA connector joined and completed actual GPU jobs.

Two official MIPLIB jobs reached the 20-second limit. The logistics MILP ran 348 GPU operations and returned a verified feasible point, but its 0.11 gap and failed node relaxations mean optimality is not proven. Full failure messages and result payloads are retained in the JSON report.
