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

Each interior-point iteration factors one system (LP: `A D A^T`; QP: the KKT
matrix); the predictor and corrector reuse it. `DenseLU::factorize` runs a
dense factorization on the GPU with the same partial-pivoting rule as the CPU
code, so results agree to rounding. Sparse products `A x` and `A^T x` also run
on the GPU for CUDA jobs.

LP interior point can instead factor `A D A^T` with the sparse LDL^T
(`solver/numerical/src/sparse_ldlt.cpp`, minimum-degree ordering, symbolic
analysis once per solve). Interior point on the same executable, RTX 2050,
generated models (`benchmarks/tools/generate_gpu_showcase.py`,
`benchmarks/tools/generate_sparse_scale.py`):

| Model (rows) | CPU dense LU | GPU dense LU | CPU sparse LDL^T | Objectives agree |
|---|---:|---:|---:|---|
| plan 600 x 900 (600) | 2.02 s | 0.91 s | 0.45 s | yes |
| plan 1000 x 1500 (1000) | 19.8 s | 2.03 s | 1.79 s | yes |
| plan 1500 x 2200 (1500) | 113 s | 5.44 s | 8.33 s | yes |
| staircase 20 x 100 (2,100) | 294 s | not run | 0.22 s | yes |
| staircase 50 x 200 (10,200) | does not fit | does not fit | 0.90 s | yes, and matches HiGHS |

The random planning LPs fill in to about 60% of the dense triangle, so the
GPU's dense factorization wins once each factorization is large enough; the
staircase LPs stay sparse, where the CPU sparse factorization is hundreds of
times faster. `SOVEREIGN_DEVICE=auto` therefore picks per model: sparse
LDL^T unless one factorization would exceed `SOVEREIGN_IPM_GPU_MIN_FLOPS`
(2e8 multiply-adds), then the GPU. Explicit CUDA jobs keep the dense GPU path.

Automatic routing sends LP and QP jobs whose factorized system has 400 rows
(`SOVEREIGN_GPU_MIN_ROWS`) to a CUDA worker, which then runs the engine with
`SOVEREIGN_DEVICE=auto`. Simplex, Frank-Wolfe and branch-and-bound (whose node
LPs use the dual simplex) stay on CPU, and explicit CUDA requests for them are
rejected at submission.

## Coordinator/worker round trip

Using the real coordinator claim, worker execution and `/api/worker/complete`
on this PC:

| Case | Requested | Assigned | State | GPU factorizations |
|---|---|---|---|---:|
| small LP | Automatic | CPU | COMPLETED | 0 |
| 1500-row planning LP | Automatic | CUDA | COMPLETED | 19 |
| 2100-row staircase LP | Automatic | CUDA | COMPLETED (engine chose sparse LDL^T) | 0 |
| MILP | Automatic | CPU | COMPLETED | 0 |
| small LP | CUDA | CUDA | COMPLETED | 6 |
| LP fixed by presolve | CUDA | CUDA | COMPLETED (retried without presolve) | 5 |
| unconstrained LP | CUDA | CUDA | FAILED (no GPU work possible) | 0 |

Every returned answer passed verification. The
[captured responses](cuda-selection-roundtrip.json) include routing reason,
status, objective, GPU counts and failure message. Reproduce with
`python tests/scripts/run_cuda_selection_roundtrip.py` from the repository root.
