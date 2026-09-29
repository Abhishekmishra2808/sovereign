# Paired CPU/CUDA interior point, engine 1.2.0

Engine 1.2.0 (`build64/solver/sovereign.exe`, the same binary bundled in
`sovereign-compute` 0.3.0 and later). On CUDA the interior point factors a dense
normal-equation or KKT system on the GPU; on the CPU it factors a sparse LDL^T
with minimum-degree ordering. Five measured CPU/CUDA pairs per model after one
warm-up pair, alternating the order. Every measured run returned `OPTIMAL`,
passed verification and matched the other device's objective within 1e-7. CUDA
runs executed real kernels; CPU runs executed none. The models are synthetic.

Hardware: Windows, 12 logical CPU threads, NVIDIA GeForce RTX 2050 (4,096 MiB,
driver 555.99). Algorithm: interior point; presolve on.

| Model | Rows x columns | CPU median wall | CUDA median wall | CPU/CUDA | CUDA ops per run |
|---|---|---:|---:|---:|---:|
| plan_1500x2200 (dense planning LP) | 1,500 x 2,200 | 8.53 s | 5.29 s | 1.61x | 131 |
| plan_1000x1500 (dense planning LP) | 1,000 x 1,500 | 1.45 s | 2.00 s | 0.72x | 131 |
| transport_100x100 (sparse LP) | 200 x 10,000 | 0.16 s | 0.37 s | 0.42x | 61 |
| portfolio_qp_600 (QP) | 41 x 600 | 0.08 s | 0.67 s | 0.12x | 48 |

The GPU is faster only on the largest dense model. On sparse models, and on the
smaller dense one, the sparse CPU factorization wins. With `SOVEREIGN_DEVICE=auto`
the engine chooses per model: it used CUDA for plan_1500x2200 and the sparse CPU
path for the transport LP and the QP. This is one GPU and one machine, single
runs of five pairs each; it is not a general speed-up claim.

Reproduce with:

```
python benchmarks/runners/run_gpu_paired.py --engine build64/solver/sovereign.exe \
  --models benchmarks/datasets/gpu/plan_1000x1500.json benchmarks/datasets/gpu/plan_1500x2200.json \
           benchmarks/datasets/gpu/portfolio_qp_600.json benchmarks/datasets/scale/transport_100x100.json \
  --trials 5 --warmups 1 --output benchmarks/reports/gpu-dense-ipm.json
```

Individual timings, hashes and verification are in [gpu-dense-ipm.json](gpu-dense-ipm.json);
full solver responses are in `gpu-dense-ipm-responses.jsonl.gz`. The earlier
engine 1.1.0 measurement, which offloaded only sparse matrix-vector products and
was slower on every model, is kept in [gpu-paired.md](gpu-paired.md).
