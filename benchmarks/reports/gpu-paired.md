# Paired CPU/CUDA whole-solver measurements

Five measured pairs per model, plus one warmup pair, on the same CUDA-enabled
Sovereign executable and machine. CPU/CUDA execution order alternated. All
measured runs returned `OPTIMAL`, passed primal verification, and matched
objectives within 1e-7 relative/absolute tolerance. CUDA runs performed real
kernels; CPU runs performed none. The models are **synthetic transport LPs**,
not industrial benchmark instances.

Hardware: Windows, 16 logical CPU threads, NVIDIA GeForce RTX 5050 Laptop GPU
(8,151 MiB, driver 610.74). Algorithm: interior point; presolve enabled.

| Variables | CPU median wall | CUDA median wall | CPU/CUDA speedup | CUDA operations/run |
|---:|---:|---:|---:|---:|
| 10,000 | 0.358 s | 0.432 s | 0.829x | 26 |
| 22,500 | 0.910 s | 1.051 s | 0.866x | 26 |
| 40,000 | 2.552 s | 2.698 s | 0.946x | 29 |

The GPU is slower on all three tested workloads. Automatic routing therefore
defaults to CPU until a repeatable whole-solver crossover is demonstrated.
Users can still select CUDA explicitly; the former structural auto heuristic
is opt-in with `SOVEREIGN_GPU_AUTO_ENABLED=1` for experiments.

After adding an exact-content, per-thread CUDA matrix/buffer cache, the same
five-pair protocol returned CPU/CUDA median wall ratios of **0.869x, 0.927x,
and 0.942x** at 10k, 22.5k and 40k variables. Every run still passed
verification and used actual CUDA kernels. The cache reduced repeated device
allocation and CSC-to-CSR conversion, but did not deliver a whole-solver
speedup. Both CPU and GPU absolute times rose in the second session, so the
within-session ratios are the useful comparison. The second run is in
`gpu-paired-cached.json` and `gpu-paired-cached-responses.jsonl.gz`.
The standalone SpMV check also passes after mutating the same host matrix and
input vectors between CUDA calls, guarding against stale cache reuse; raw
output is in `gpu-spmv-cached.txt`.

Reproduce with `python benchmarks/runners/run_gpu_paired.py --trials 5 --warmups 1`.
The machine-readable [summary](gpu-paired.json) includes individual timings,
medians, median absolute deviations, input/executable SHA-256 hashes, status,
verification, objective parity and GPU operation counts. Compressed
`gpu-paired-responses.jsonl.gz` contains every full solver response and
verification object. This experiment does not measure kernel/copy breakdown
or peak GPU memory; those remain open profiling tasks.
