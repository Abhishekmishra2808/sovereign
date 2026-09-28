# SIH 26119: online worker evidence

Generated: 2026-09-26T23:10:47.560145+00:00

Measured through a local HTTP coordinator and outbound worker using the same protocol as Render. All runs in this report use CPU. No live Render deployment or GPU speedup is claimed.

43 comparisons; 23 verified optimum matches; 9 numerical errors; 10 timeouts.

Machine: Windows-10-10.0.26200-SP0 | 16 logical CPUs | NVIDIA GeForce RTX 5050 Laptop GPU

Per-job wall-clock limit: 3 seconds. HiGHS: one CPU thread.

Sovereign timings below are solver-reported; timeouts use the enforced wall-clock limit. Full HTTP turnaround is recorded separately in JSON. Single runs, not statistical performance claims.

| Dataset | Source | Algorithm | Sovereign status | Objective | Seconds | Verified | HiGHS status | HiGHS objective | HiGHS seconds | Verified optimum match |
|---|---|---|---|---:|---:|---|---|---:|---:|---|
| afiro | Netlib LP | Revised simplex | OPTIMAL | -464.75314285714285 | 0.0007 | True | OPTIMAL | -464.75314285714285 | 0.0177 | True |
| afiro | Netlib LP | Interior point | OPTIMAL | -464.7531428511401 | 0.0003 | True | OPTIMAL | -464.75314285714285 | 0.0177 | True |
| flugpl | MIPLIB official | Branch & cut / strong | NUMERICAL_ERROR | None | 0.0008 | False | OPTIMAL | 1201500.0 | 0.1227 | False |
| flugpl | MIPLIB official | Branch & bound / fractional | NUMERICAL_ERROR | None | 0.0005 | False | OPTIMAL | 1201500.0 | 0.1227 | False |
| flugpl | MIPLIB official | Branch & cut / pseudocost | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | 1201500.0 | 0.1227 | False |
| gt2 | MIPLIB official | Branch & cut / strong | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | 21166.0 | 0.1170 | False |
| gt2 | MIPLIB official | Branch & bound / fractional | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | 21166.0 | 0.1170 | False |
| gt2 | MIPLIB official | Branch & cut / pseudocost | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | 21166.0 | 0.1170 | False |
| b-ball | MIPLIB official | Branch & cut / strong | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | -1.500001 | 0.2460 | False |
| b-ball | MIPLIB official | Branch & bound / fractional | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | -1.500001 | 0.2460 | False |
| b-ball | MIPLIB official | Branch & cut / pseudocost | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | -1.500001 | 0.2460 | False |
| pk1 | MIPLIB official | Branch & cut / strong | NUMERICAL_ERROR | None | 1.6348 | False | TIME_LIMIT_REACHED | 18.0 | 3.0021 | False |
| pk1 | MIPLIB official | Branch & bound / fractional | NUMERICAL_ERROR | None | 1.8540 | False | TIME_LIMIT_REACHED | 18.0 | 3.0021 | False |
| pk1 | MIPLIB official | Branch & cut / pseudocost | TIME_LIMIT | None | 3.0000 | False | TIME_LIMIT_REACHED | 18.0 | 3.0021 | False |
| gen-ip016 | MIPLIB official | Branch & cut / strong | NUMERICAL_ERROR | None | 0.0007 | False | TIME_LIMIT_REACHED | -9405.178445272 | 3.0036 | False |
| gen-ip016 | MIPLIB official | Branch & bound / fractional | NUMERICAL_ERROR | None | 0.0006 | False | TIME_LIMIT_REACHED | -9405.178445272 | 3.0036 | False |
| gen-ip016 | MIPLIB official | Branch & cut / pseudocost | TIME_LIMIT | None | 3.0000 | False | TIME_LIMIT_REACHED | -9405.178445272 | 3.0036 | False |
| kuhn_degeneracy | Synthetic robustness | Revised simplex | OPTIMAL | -1.2500000000000002 | 0.0001 | True | OPTIMAL | -1.25 | 0.0008 | True |
| kuhn_degeneracy | Synthetic robustness | Interior point | OPTIMAL | -1.2499999999982918 | 0.0001 | True | OPTIMAL | -1.25 | 0.0008 | True |
| illconditioned | Synthetic robustness | Revised simplex | OPTIMAL | 30000000.0 | 0.0001 | True | OPTIMAL | 30000000.0 | 0.0006 | True |
| illconditioned | Synthetic robustness | Interior point | OPTIMAL | 29999999.99967302 | 0.0001 | True | OPTIMAL | 30000000.0 | 0.0006 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / strong | OPTIMAL | 100.0 | 0.0005 | True | OPTIMAL | 100.0 | 0.0009 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & bound / fractional | OPTIMAL | 100.0 | 0.0001 | True | OPTIMAL | 100.0 | 0.0009 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / pseudocost | OPTIMAL | 100.0 | 0.0002 | True | OPTIMAL | 100.0 | 0.0009 | True |
| transport_20x20 | Synthetic scale | Revised simplex | OPTIMAL | 201.99999999999994 | 0.0074 | True | OPTIMAL | 201.99999999999994 | 0.0020 | True |
| transport_20x20 | Synthetic scale | Interior point | OPTIMAL | 202.0000000008087 | 0.0016 | True | OPTIMAL | 201.99999999999994 | 0.0020 | True |
| transport_50x50 | Synthetic scale | Revised simplex | OPTIMAL | 504.6000000000004 | 0.9790 | True | OPTIMAL | 504.6000000000004 | 0.0095 | True |
| transport_50x50 | Synthetic scale | Interior point | OPTIMAL | 504.60000000368524 | 0.0213 | True | OPTIMAL | 504.6000000000004 | 0.0095 | True |
| transport_100x100 | Synthetic scale | Revised simplex | TIME_LIMIT | None | 3.0000 | False | OPTIMAL | 1009.0000000000015 | 0.0342 | False |
| transport_100x100 | Synthetic scale | Interior point | OPTIMAL | 1009.0000000400709 | 0.3639 | True | OPTIMAL | 1009.0000000000015 | 0.0342 | True |
| industrial_refinery_lp | Industrial example | Revised simplex | NUMERICAL_ERROR | None | 0.0002 | False | OPTIMAL | 675000.0 | 0.0010 | False |
| industrial_refinery_lp | Industrial example | Interior point | OPTIMAL | 674999.999966755 | 0.0001 | True | OPTIMAL | 675000.0 | 0.0010 | True |
| industrial_blending_lp | Industrial example | Revised simplex | OPTIMAL | 7200.000000000002 | 0.0010 | True | OPTIMAL | 7200.0 | 0.0008 | True |
| industrial_blending_lp | Industrial example | Interior point | OPTIMAL | 7200.000000026874 | 0.0001 | True | OPTIMAL | 7200.0 | 0.0008 | True |
| industrial_power_dispatch_lp | Industrial example | Revised simplex | OPTIMAL | 2700.0 | 0.0001 | True | OPTIMAL | 2700.0 | 0.0005 | True |
| industrial_power_dispatch_lp | Industrial example | Interior point | OPTIMAL | 2700.000000099279 | 0.0002 | True | OPTIMAL | 2700.0 | 0.0005 | True |
| industrial_logistics_milp | Industrial example | Branch & cut / strong | NUMERICAL_ERROR | None | 0.0002 | False | OPTIMAL | 550.0 | 0.0162 | False |
| industrial_logistics_milp | Industrial example | Branch & bound / fractional | NUMERICAL_ERROR | None | 0.0002 | False | OPTIMAL | 550.0 | 0.0162 | False |
| industrial_logistics_milp | Industrial example | Branch & cut / pseudocost | FEASIBLE | 550.0000000003491 | 0.0016 | True | OPTIMAL | 550.0 | 0.0162 | False |
| sample_qp | QP example | QP interior point | OPTIMAL | -2.0 | 0.0001 | True | OPTIMAL | -1.999999999999995 | 0.0039 | True |
| sample_qp | QP example | Frank-Wolfe | OPTIMAL | -2.0 | 0.0005 | True | OPTIMAL | -1.999999999999995 | 0.0039 | True |
| qp_ge | QP example | QP interior point | OPTIMAL | 1.9999999996220788 | 0.0001 | True | OPTIMAL | 1.9999999999999998 | 0.0010 | True |
| qp_ge | QP example | Frank-Wolfe | OPTIMAL | 2.0 | 0.0001 | True | OPTIMAL | 1.9999999999999998 | 0.0010 | True |

## Coverage and limits

- Netlib: AFIRO. Official MIPLIB: flugpl, gt2, b-ball, pk1, gen-ip016 (original MPS passed independently to each solver).
- Synthetic robustness: degeneracy, ill-conditioning, weak relaxation. Synthetic transport: 400, 2,500, and 10,000 variables.
- Repository industrial examples: refinery, blending, power dispatch, logistics. These are not claimed as published or proprietary industrial data.
- QP examples compared with HiGHS using the quadratic Hessian, not an LP relaxation.
- Mittelmann and QPLIB instances are not bundled or tested. Million-variable scale is shown only on synthetic structured LP/QP (scale-ladder.md); full GPU acceleration and GPU speedups are not established.
- GPU runs are not claimed when no CUDA-enabled engine is available. Auto-routing policy tests are separate from GPU performance evidence.
- Only OPTIMAL + independent verification + reference OPTIMAL + objective tolerance agreement counts as a match. Failed verification and timeouts remain visible.
- HiGHS is used only in this benchmark harness, never to solve production jobs.
