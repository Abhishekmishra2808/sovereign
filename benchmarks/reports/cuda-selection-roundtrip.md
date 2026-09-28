# CUDA detection, routing and job acceptance

The engine now reaches the GPU through the NVIDIA driver loaded at run time
(`solver/gpu/src/cuda_driver.cpp`), so the ordinary build detects CUDA on any
machine with an NVIDIA driver; no CUDA Toolkit is needed. The probe loads the
driver, compiles the PTX kernels for the installed card and checks a dense LU
and both sparse products against the CPU before reporting
`cuda_available: true`. Otherwise `cuda_reason` says why.

On this PC (NVIDIA GeForce RTX 2050, compute 8.6, 4 GB, driver CUDA 12.5,
32-bit MinGW engine) `sovereign capabilities` reports:

```json
{"cuda_available":true,"cuda_backend":"driver-api","cuda_driver_version":"12.5",
 "gpu_compute_capability":"8.6","gpu_memory_mb":4095,"gpu_name":"NVIDIA GeForce RTX 2050"}
```

## What runs on the GPU

Each interior-point iteration factors a dense system (LP: `A D A^T`; QP: the
KKT matrix) twice. `DenseLU::factorize` runs that factorization on the GPU with
the same partial-pivoting rule as the CPU code, so results agree to rounding.
Sparse products `A x` and `A^T x` also run on the GPU for CUDA jobs.

Interior point, CPU against RTX 2050, same executable (generated
production-planning LPs, `build/gen_lp.py`):

| Rows x columns | CPU | GPU | Speed-up | Objectives agree |
|---|---:|---:|---:|---|
| 300 x 500 | 0.46 s | 0.82 s | CPU faster | yes |
| 600 x 900 | 3.63 s | 1.50 s | 2.4x | yes |
| 1000 x 1500 | 32.6 s | 3.80 s | 8.6x | yes |
| 1500 x 2200 | 314.8 s | 9.91 s | 32x | yes |
| transport 200x200 (400 rows) | 1.88 s | 1.29 s | 1.5x | yes |
| portfolio QP, 600 assets | 3.43 s | 1.08 s | 3.2x | yes |

Automatic routing therefore prefers a CUDA worker once the factorized system
has 400 rows (`SOVEREIGN_GPU_MIN_ROWS`). Simplex, Frank-Wolfe and
branch-and-bound (whose node LPs use the dual simplex) stay on CPU, and
explicit CUDA requests for them are rejected at submission.

## Coordinator/worker round trip

Using the real coordinator claim, worker execution and `/api/worker/complete`
on this PC:

| Case | Requested | Assigned | State | GPU factorizations |
|---|---|---|---|---:|
| small LP | Automatic | CPU | COMPLETED | 0 |
| 600-row LP | Automatic | CUDA | COMPLETED | 33 |
| MILP | Automatic | CPU | COMPLETED | 0 |
| small LP | CUDA | CUDA | COMPLETED | 11 |
| LP fixed by presolve | CUDA | CUDA | COMPLETED (retried without presolve) | 9 |
| unconstrained LP | CUDA | CUDA | FAILED (no GPU work possible) | 0 |

Every returned answer passed verification. The
[captured responses](cuda-selection-roundtrip.json) include routing reason,
status, objective, GPU counts and failure message. Reproduce with
`python tests/scripts/run_cuda_selection_roundtrip.py` from the repository root.
