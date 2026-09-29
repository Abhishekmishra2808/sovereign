# SIH 26119: online worker evidence

Generated: 2026-09-29T06:32:41.844514+00:00

Measured through a local HTTP coordinator and outbound worker using the same protocol as Render. All runs in this report use CPU. No live Render deployment or GPU speedup is claimed.

61 comparisons; 49 verified optimum matches; 0 numerical errors; 1 timeouts.

Machine: Windows-10-10.0.26200-SP0 | 12 logical CPUs | NVIDIA GeForce RTX 2050

Per-job wall-clock limit: 30 seconds. HiGHS: one CPU thread.

Sovereign timings below are solver-reported; timeouts use the enforced wall-clock limit. Full HTTP turnaround is recorded separately in JSON. Single runs, not statistical performance claims.

| Dataset | Source | Algorithm | Sovereign status | Objective | Seconds | Verified | HiGHS status | HiGHS objective | HiGHS seconds | Verified optimum match |
|---|---|---|---|---:|---:|---|---|---:|---:|---|
| afiro | Netlib LP | Dual simplex | OPTIMAL | -464.7531428571429 | 0.0003 | True | OPTIMAL | -464.75314285714285 | 0.0016 | True |
| afiro | Netlib LP | Interior point | OPTIMAL | -464.7531428511402 | 0.0006 | True | OPTIMAL | -464.75314285714285 | 0.0016 | True |
| flugpl | MIPLIB official | Branch & cut / strong | OPTIMAL | 1201500.0 | 0.4580 | True | OPTIMAL | 1201500.0 | 0.2225 | True |
| flugpl | MIPLIB official | Branch & bound / fractional | OPTIMAL | 1201500.0 | 0.8531 | True | OPTIMAL | 1201500.0 | 0.2225 | True |
| flugpl | MIPLIB official | Branch & cut / pseudocost | OPTIMAL | 1201500.0 | 0.7409 | True | OPTIMAL | 1201500.0 | 0.2225 | True |
| gt2 | MIPLIB official | Branch & cut / strong | OPTIMAL | 21166.0 | 22.6656 | True | OPTIMAL | 21166.0 | 0.0836 | True |
| gt2 | MIPLIB official | Branch & bound / fractional | FEASIBLE | 53344.0 | 27.0493 | True | OPTIMAL | 21166.0 | 0.0836 | False |
| gt2 | MIPLIB official | Branch & cut / pseudocost | OPTIMAL | 21166.0 | 4.9207 | True | OPTIMAL | 21166.0 | 0.0836 | True |
| b-ball | MIPLIB official | Branch & cut / strong | FEASIBLE | -1.5 | 26.5905 | True | OPTIMAL | -1.500001 | 0.2938 | False |
| b-ball | MIPLIB official | Branch & bound / fractional | FEASIBLE | -1.5 | 23.3558 | True | OPTIMAL | -1.500001 | 0.2938 | False |
| b-ball | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | -1.5 | 23.3302 | True | OPTIMAL | -1.500001 | 0.2938 | False |
| pk1 | MIPLIB official | Branch & cut / strong | FEASIBLE | 11.999999999999998 | 27.0328 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0093 | False |
| pk1 | MIPLIB official | Branch & bound / fractional | FEASIBLE | 11.999999999999998 | 27.0288 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0093 | False |
| pk1 | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | 12.0 | 27.0362 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0093 | False |
| gen-ip016 | MIPLIB official | Branch & cut / strong | FEASIBLE | -9425.864845696002 | 27.0456 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0148 | False |
| gen-ip016 | MIPLIB official | Branch & bound / fractional | FEASIBLE | -9441.184307408 | 24.3204 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0148 | False |
| gen-ip016 | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | -9432.954064407999 | 27.0359 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0148 | False |
| kuhn_degeneracy | Synthetic robustness | Dual simplex | OPTIMAL | -1.25 | 0.0001 | True | OPTIMAL | -1.25 | 0.0012 | True |
| kuhn_degeneracy | Synthetic robustness | Interior point | OPTIMAL | -1.2499999999982918 | 0.0002 | True | OPTIMAL | -1.25 | 0.0012 | True |
| illconditioned | Synthetic robustness | Dual simplex | OPTIMAL | 30000000.0 | 0.0001 | True | OPTIMAL | 30000000.0 | 0.0006 | True |
| illconditioned | Synthetic robustness | Interior point | OPTIMAL | 29999999.99967302 | 0.0002 | True | OPTIMAL | 30000000.0 | 0.0006 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / strong | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0009 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & bound / fractional | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0009 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / pseudocost | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0009 | True |
| transport_20x20 | Synthetic scale | Dual simplex | OPTIMAL | 201.99999999999991 | 0.0007 | True | OPTIMAL | 201.99999999999994 | 0.0019 | True |
| transport_20x20 | Synthetic scale | Interior point | OPTIMAL | 202.00000000080883 | 0.0010 | True | OPTIMAL | 201.99999999999994 | 0.0019 | True |
| transport_50x50 | Synthetic scale | Dual simplex | OPTIMAL | 504.6000000000004 | 0.0094 | True | OPTIMAL | 504.6000000000004 | 0.0103 | True |
| transport_50x50 | Synthetic scale | Interior point | OPTIMAL | 504.6000000036868 | 0.0086 | True | OPTIMAL | 504.6000000000004 | 0.0103 | True |
| transport_100x100 | Synthetic scale | Dual simplex | OPTIMAL | 1009.0000000000014 | 0.0828 | True | OPTIMAL | 1009.0000000000015 | 0.0454 | True |
| transport_100x100 | Synthetic scale | Interior point | OPTIMAL | 1009.0000000400751 | 0.0460 | True | OPTIMAL | 1009.0000000000015 | 0.0454 | True |
| transport_150x150 | Synthetic scale | Dual simplex | OPTIMAL | 1513.3999999999976 | 0.3673 | True | OPTIMAL | 1513.3999999999976 | 0.1588 | True |
| transport_150x150 | Synthetic scale | Interior point | OPTIMAL | 1513.4000004520094 | 0.1265 | True | OPTIMAL | 1513.3999999999976 | 0.1588 | True |
| transport_200x200 | Synthetic scale | Dual simplex | OPTIMAL | 2017.7999999999934 | 1.2637 | True | OPTIMAL | 2017.7999999999936 | 0.7049 | True |
| transport_200x200 | Synthetic scale | Interior point | OPTIMAL | 2017.800000009963 | 0.2744 | True | OPTIMAL | 2017.7999999999936 | 0.7049 | True |
| industrial_refinery_lp | Industrial example | Dual simplex | OPTIMAL | 675000.0 | 0.0001 | True | OPTIMAL | 675000.0 | 0.0010 | True |
| industrial_refinery_lp | Industrial example | Interior point | OPTIMAL | 674999.9999667553 | 0.0004 | True | OPTIMAL | 675000.0 | 0.0010 | True |
| industrial_blending_lp | Industrial example | Dual simplex | OPTIMAL | 7200.0 | 0.0001 | True | OPTIMAL | 7200.0 | 0.0011 | True |
| industrial_blending_lp | Industrial example | Interior point | OPTIMAL | 7200.000000026874 | 0.0004 | True | OPTIMAL | 7200.0 | 0.0011 | True |
| industrial_power_dispatch_lp | Industrial example | Dual simplex | OPTIMAL | 2700.0 | 0.0001 | True | OPTIMAL | 2700.0 | 0.0006 | True |
| industrial_power_dispatch_lp | Industrial example | Interior point | OPTIMAL | 2700.000000099279 | 0.0004 | True | OPTIMAL | 2700.0 | 0.0006 | True |
| industrial_logistics_milp | Industrial example | Branch & cut / strong | OPTIMAL | 550.0 | 0.0004 | True | OPTIMAL | 550.0 | 0.0148 | True |
| industrial_logistics_milp | Industrial example | Branch & bound / fractional | OPTIMAL | 550.0 | 0.0006 | True | OPTIMAL | 550.0 | 0.0148 | True |
| industrial_logistics_milp | Industrial example | Branch & cut / pseudocost | OPTIMAL | 550.0 | 0.0004 | True | OPTIMAL | 550.0 | 0.0148 | True |
| sample_qp | QP example | QP interior point | OPTIMAL | -2.0 | 0.0002 | True | OPTIMAL | -1.999999999999995 | 0.0008 | True |
| sample_qp | QP example | Frank-Wolfe | OPTIMAL | -2.0 | 0.0006 | True | OPTIMAL | -1.999999999999995 | 0.0008 | True |
| qp_ge | QP example | QP interior point | OPTIMAL | 1.9999999996220788 | 0.0002 | True | OPTIMAL | 1.9999999999999998 | 0.0011 | True |
| qp_ge | QP example | Frank-Wolfe | OPTIMAL | 2.0 | 0.0002 | True | OPTIMAL | 1.9999999999999998 | 0.0011 | True |
| plan_600x900 | GPU showcase | Dual simplex | OPTIMAL | 37175.761943396574 | 0.2356 | True | OPTIMAL | 37175.761943396436 | 0.0847 | True |
| plan_600x900 | GPU showcase | Interior point | OPTIMAL | 37175.761931149886 | 0.2576 | True | OPTIMAL | 37175.761943396436 | 0.0847 | True |
| plan_1000x1500 | GPU showcase | Dual simplex | OPTIMAL | 59174.29816757092 | 1.1418 | True | OPTIMAL | 59174.298167570734 | 0.3308 | True |
| plan_1000x1500 | GPU showcase | Interior point | OPTIMAL | 59174.298167497785 | 1.1890 | True | OPTIMAL | 59174.298167570734 | 0.3308 | True |
| plan_1500x2200 | GPU showcase | Dual simplex | OPTIMAL | 91500.1974783319 | 3.9157 | True | OPTIMAL | 91500.19747833154 | 0.9812 | True |
| plan_1500x2200 | GPU showcase | Interior point | OPTIMAL | 91500.19742444395 | 7.7305 | True | OPTIMAL | 91500.19747833154 | 0.9812 | True |
| portfolio_qp_600 | GPU showcase | QP interior point | OPTIMAL | -0.1849816431569199 | 0.0467 | True | OPTIMAL | -0.1849816443799799 | 0.0152 | True |
| portfolio_qp_600 | GPU showcase | Frank-Wolfe | OPTIMAL | -0.1844093054520936 | 0.4498 | True | OPTIMAL | -0.1849816443799799 | 0.0152 | False |
| staircase_20x100 | Sparse scale | Dual simplex | OPTIMAL | 21329.7060606061 | 0.1311 | True | OPTIMAL | 21329.70606060608 | 0.0311 | True |
| staircase_20x100 | Sparse scale | Interior point | OPTIMAL | 21329.70606259925 | 0.0251 | True | OPTIMAL | 21329.70606060608 | 0.0311 | True |
| staircase_50x200 | Sparse scale | Dual simplex | OPTIMAL | 106185.7357575748 | 3.4624 | True | OPTIMAL | 106185.73575757477 | 0.3768 | True |
| staircase_50x200 | Sparse scale | Interior point | OPTIMAL | 106185.73575779606 | 0.1832 | True | OPTIMAL | 106185.73575757477 | 0.3768 | True |
| portfolio_qp_5000 | Sparse scale | QP interior point | OPTIMAL | -0.1944651286560086 | 0.0629 | True | OPTIMAL | -0.19446512875558752 | 0.4261 | True |
| portfolio_qp_5000 | Sparse scale | Frank-Wolfe | TIME_LIMIT | None | 30.0000 | False | OPTIMAL | -0.19446512875558752 | 0.4261 | False |

## Coverage and limits

- Netlib: AFIRO. Official MIPLIB: flugpl, gt2, b-ball, pk1, gen-ip016 (original MPS passed independently to each solver).
- Synthetic robustness: degeneracy, ill-conditioning, weak relaxation. Synthetic scale: transport LPs to 40,000 variables, planning LPs to 1,500 rows x 2,200 variables, staircase LPs to 10,200 rows x 20,000 variables and portfolio QPs to 5,000 assets.
- Repository industrial examples: refinery, blending, power dispatch, logistics. These are not claimed as published or proprietary industrial data.
- QP examples compared with HiGHS using the quadratic Hessian, not an LP relaxation.
- Mittelmann and QPLIB instances are not bundled or tested. Million-variable scale is shown only on synthetic structured LP/QP (scale-ladder.md).
- GPU: measured separately in gpu-dense-ipm.md. CUDA interior point is 1.6x faster only on the 1,500-row dense planning LP; the sparse CPU path wins on the other models tested.
- Only OPTIMAL + independent verification + reference OPTIMAL + objective tolerance agreement counts as a match. Failed verification and timeouts remain visible.
- HiGHS is used only in this benchmark harness, never to solve production jobs.
