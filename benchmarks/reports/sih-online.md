# SIH 26119: online worker evidence

Generated: 2026-09-29T09:19:47.698277+00:00

Measured through a local HTTP coordinator and outbound worker using the same protocol as Render. All runs in this report use CPU. No live Render deployment or GPU speedup is claimed.

61 comparisons; 49 verified optimum matches; 0 numerical errors; 1 timeouts.

Machine: Windows-10-10.0.26200-SP0 | 12 logical CPUs | NVIDIA GeForce RTX 2050

Per-job wall-clock limit: 30 seconds. HiGHS: one CPU thread.

Sovereign timings below are solver-reported; timeouts use the enforced wall-clock limit. Full HTTP turnaround is recorded separately in JSON. Single runs, not statistical performance claims.

| Dataset | Source | Algorithm | Sovereign status | Objective | Seconds | Verified | HiGHS status | HiGHS objective | HiGHS seconds | Verified optimum match |
|---|---|---|---|---:|---:|---|---|---:|---:|---|
| afiro | Netlib LP | Dual simplex | OPTIMAL | -464.7531428571429 | 0.0003 | True | OPTIMAL | -464.75314285714285 | 0.0111 | True |
| afiro | Netlib LP | Interior point | OPTIMAL | -464.7531428511402 | 0.0009 | True | OPTIMAL | -464.75314285714285 | 0.0111 | True |
| flugpl | MIPLIB official | Branch & cut / strong | OPTIMAL | 1201500.0 | 0.1932 | True | OPTIMAL | 1201500.0 | 0.3444 | True |
| flugpl | MIPLIB official | Branch & bound / fractional | OPTIMAL | 1201500.0 | 1.4704 | True | OPTIMAL | 1201500.0 | 0.3444 | True |
| flugpl | MIPLIB official | Branch & cut / pseudocost | OPTIMAL | 1201500.0 | 1.0055 | True | OPTIMAL | 1201500.0 | 0.3444 | True |
| gt2 | MIPLIB official | Branch & cut / strong | OPTIMAL | 21166.0 | 0.7043 | True | OPTIMAL | 21166.0 | 0.1755 | True |
| gt2 | MIPLIB official | Branch & bound / fractional | FEASIBLE | 53344.0 | 27.0428 | True | OPTIMAL | 21166.0 | 0.1755 | False |
| gt2 | MIPLIB official | Branch & cut / pseudocost | OPTIMAL | 21166.0 | 7.7892 | True | OPTIMAL | 21166.0 | 0.1755 | True |
| b-ball | MIPLIB official | Branch & cut / strong | FEASIBLE | -1.5 | 22.4254 | True | OPTIMAL | -1.500001 | 0.5643 | False |
| b-ball | MIPLIB official | Branch & bound / fractional | FEASIBLE | -1.5 | 19.9000 | True | OPTIMAL | -1.500001 | 0.5643 | False |
| b-ball | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | -1.5 | 19.8093 | True | OPTIMAL | -1.500001 | 0.5643 | False |
| pk1 | MIPLIB official | Branch & cut / strong | FEASIBLE | 11.0 | 27.0204 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0021 | False |
| pk1 | MIPLIB official | Branch & bound / fractional | FEASIBLE | 11.0 | 27.0387 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0021 | False |
| pk1 | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | 12.0 | 27.0150 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0021 | False |
| gen-ip016 | MIPLIB official | Branch & cut / strong | FEASIBLE | -9430.336616408 | 27.0383 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0030 | False |
| gen-ip016 | MIPLIB official | Branch & bound / fractional | FEASIBLE | -9441.184307408 | 21.7442 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0030 | False |
| gen-ip016 | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | -9432.954064407999 | 27.0352 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0030 | False |
| kuhn_degeneracy | Synthetic robustness | Dual simplex | OPTIMAL | -1.25 | 0.0001 | True | OPTIMAL | -1.25 | 0.0007 | True |
| kuhn_degeneracy | Synthetic robustness | Interior point | OPTIMAL | -1.2499999999982918 | 0.0003 | True | OPTIMAL | -1.25 | 0.0007 | True |
| illconditioned | Synthetic robustness | Dual simplex | OPTIMAL | 30000000.0 | 0.0004 | True | OPTIMAL | 30000000.0 | 0.0005 | True |
| illconditioned | Synthetic robustness | Interior point | OPTIMAL | 29999999.99967302 | 0.0002 | True | OPTIMAL | 30000000.0 | 0.0005 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / strong | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0008 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & bound / fractional | OPTIMAL | 100.0 | 0.0009 | True | OPTIMAL | 100.0 | 0.0008 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / pseudocost | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0008 | True |
| transport_20x20 | Synthetic scale | Dual simplex | OPTIMAL | 201.99999999999991 | 0.0008 | True | OPTIMAL | 201.99999999999994 | 0.0017 | True |
| transport_20x20 | Synthetic scale | Interior point | OPTIMAL | 202.00000000080883 | 0.0010 | True | OPTIMAL | 201.99999999999994 | 0.0017 | True |
| transport_50x50 | Synthetic scale | Dual simplex | OPTIMAL | 504.6000000000004 | 0.0070 | True | OPTIMAL | 504.6000000000004 | 0.0075 | True |
| transport_50x50 | Synthetic scale | Interior point | OPTIMAL | 504.6000000036868 | 0.0068 | True | OPTIMAL | 504.6000000000004 | 0.0075 | True |
| transport_100x100 | Synthetic scale | Dual simplex | OPTIMAL | 1009.0000000000014 | 0.0647 | True | OPTIMAL | 1009.0000000000015 | 0.0362 | True |
| transport_100x100 | Synthetic scale | Interior point | OPTIMAL | 1009.0000000400751 | 0.0359 | True | OPTIMAL | 1009.0000000000015 | 0.0362 | True |
| transport_150x150 | Synthetic scale | Dual simplex | OPTIMAL | 1513.3999999999976 | 0.3248 | True | OPTIMAL | 1513.3999999999976 | 0.1276 | True |
| transport_150x150 | Synthetic scale | Interior point | OPTIMAL | 1513.4000004520094 | 0.1313 | True | OPTIMAL | 1513.3999999999976 | 0.1276 | True |
| transport_200x200 | Synthetic scale | Dual simplex | OPTIMAL | 2017.7999999999934 | 1.6364 | True | OPTIMAL | 2017.7999999999936 | 0.5782 | True |
| transport_200x200 | Synthetic scale | Interior point | OPTIMAL | 2017.800000009963 | 0.2339 | True | OPTIMAL | 2017.7999999999936 | 0.5782 | True |
| industrial_refinery_lp | Industrial example | Dual simplex | OPTIMAL | 675000.0 | 0.0001 | True | OPTIMAL | 675000.0 | 0.0008 | True |
| industrial_refinery_lp | Industrial example | Interior point | OPTIMAL | 674999.9999667553 | 0.0005 | True | OPTIMAL | 675000.0 | 0.0008 | True |
| industrial_blending_lp | Industrial example | Dual simplex | OPTIMAL | 7200.0 | 0.0004 | True | OPTIMAL | 7200.0 | 0.0026 | True |
| industrial_blending_lp | Industrial example | Interior point | OPTIMAL | 7200.000000026874 | 0.0004 | True | OPTIMAL | 7200.0 | 0.0026 | True |
| industrial_power_dispatch_lp | Industrial example | Dual simplex | OPTIMAL | 2700.0 | 0.0002 | True | OPTIMAL | 2700.0 | 0.0006 | True |
| industrial_power_dispatch_lp | Industrial example | Interior point | OPTIMAL | 2700.000000099279 | 0.0004 | True | OPTIMAL | 2700.0 | 0.0006 | True |
| industrial_logistics_milp | Industrial example | Branch & cut / strong | OPTIMAL | 550.0 | 0.0009 | True | OPTIMAL | 550.0 | 0.0299 | True |
| industrial_logistics_milp | Industrial example | Branch & bound / fractional | OPTIMAL | 550.0 | 0.0007 | True | OPTIMAL | 550.0 | 0.0299 | True |
| industrial_logistics_milp | Industrial example | Branch & cut / pseudocost | OPTIMAL | 550.0 | 0.0022 | True | OPTIMAL | 550.0 | 0.0299 | True |
| sample_qp | QP example | QP interior point | OPTIMAL | -2.0 | 0.0002 | True | OPTIMAL | -1.999999999999995 | 0.0021 | True |
| sample_qp | QP example | Frank-Wolfe | OPTIMAL | -2.0 | 0.0008 | True | OPTIMAL | -1.999999999999995 | 0.0021 | True |
| qp_ge | QP example | QP interior point | OPTIMAL | 1.9999999996220788 | 0.0005 | True | OPTIMAL | 1.9999999999999998 | 0.0046 | True |
| qp_ge | QP example | Frank-Wolfe | OPTIMAL | 2.0 | 0.0002 | True | OPTIMAL | 1.9999999999999998 | 0.0046 | True |
| plan_600x900 | GPU showcase | Dual simplex | OPTIMAL | 37175.761943396574 | 0.2672 | True | OPTIMAL | 37175.761943396436 | 0.0959 | True |
| plan_600x900 | GPU showcase | Interior point | OPTIMAL | 37175.761931149886 | 0.2848 | True | OPTIMAL | 37175.761943396436 | 0.0959 | True |
| plan_1000x1500 | GPU showcase | Dual simplex | OPTIMAL | 59174.29816757092 | 1.2531 | True | OPTIMAL | 59174.298167570734 | 0.5262 | True |
| plan_1000x1500 | GPU showcase | Interior point | OPTIMAL | 59174.298167497785 | 1.3272 | True | OPTIMAL | 59174.298167570734 | 0.5262 | True |
| plan_1500x2200 | GPU showcase | Dual simplex | OPTIMAL | 91500.1974783319 | 4.5626 | True | OPTIMAL | 91500.19747833154 | 1.1399 | True |
| plan_1500x2200 | GPU showcase | Interior point | OPTIMAL | 91500.19742444395 | 13.9809 | True | OPTIMAL | 91500.19747833154 | 1.1399 | True |
| portfolio_qp_600 | GPU showcase | QP interior point | OPTIMAL | -0.1849816431569199 | 0.0723 | True | OPTIMAL | -0.1849816443799799 | 0.0238 | True |
| portfolio_qp_600 | GPU showcase | Frank-Wolfe | FEASIBLE | -0.1844093054520936 | 1.9233 | True | OPTIMAL | -0.1849816443799799 | 0.0238 | False |
| staircase_20x100 | Sparse scale | Dual simplex | OPTIMAL | 21329.7060606061 | 0.1125 | True | OPTIMAL | 21329.70606060608 | 0.0367 | True |
| staircase_20x100 | Sparse scale | Interior point | OPTIMAL | 21329.70606259925 | 0.0435 | True | OPTIMAL | 21329.70606060608 | 0.0367 | True |
| staircase_50x200 | Sparse scale | Dual simplex | OPTIMAL | 106185.7357575748 | 2.7662 | True | OPTIMAL | 106185.73575757477 | 0.3497 | True |
| staircase_50x200 | Sparse scale | Interior point | OPTIMAL | 106185.73575779606 | 0.1449 | True | OPTIMAL | 106185.73575757477 | 0.3497 | True |
| portfolio_qp_5000 | Sparse scale | QP interior point | OPTIMAL | -0.1944651286560086 | 0.0513 | True | OPTIMAL | -0.19446512875558752 | 0.3250 | True |
| portfolio_qp_5000 | Sparse scale | Frank-Wolfe | TIME_LIMIT | None | 30.0000 | False | OPTIMAL | -0.19446512875558752 | 0.3250 | False |

## Coverage and limits

- Netlib: AFIRO. Official MIPLIB: flugpl, gt2, b-ball, pk1, gen-ip016 (original MPS passed independently to each solver).
- Synthetic robustness: degeneracy, ill-conditioning, weak relaxation. Synthetic scale: transport LPs to 40,000 variables, planning LPs to 1,500 rows x 2,200 variables, staircase LPs to 10,200 rows x 20,000 variables and portfolio QPs to 5,000 assets.
- Repository industrial examples: refinery, blending, power dispatch, logistics. These are not claimed as published or proprietary industrial data.
- QP examples compared with HiGHS using the quadratic Hessian, not an LP relaxation.
- Mittelmann and QPLIB instances are not bundled or tested. Million-variable scale is shown only on synthetic structured LP/QP (scale-ladder.md).
- GPU: measured separately in gpu-dense-ipm.md. CUDA interior point is 1.6x faster only on the 1,500-row dense planning LP; the sparse CPU path wins on the other models tested.
- Only OPTIMAL + independent verification + reference OPTIMAL + objective tolerance agreement counts as a match. Failed verification and timeouts remain visible.
- HiGHS is used only in this benchmark harness, never to solve production jobs.
