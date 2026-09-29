# SIH 26119: online worker evidence

Generated: 2026-09-29T07:29:50.085320+00:00

Measured through a local HTTP coordinator and outbound worker using the same protocol as Render. All runs in this report use CPU. No live Render deployment or GPU speedup is claimed.

61 comparisons; 49 verified optimum matches; 0 numerical errors; 1 timeouts.

Machine: Windows-10-10.0.26200-SP0 | 12 logical CPUs | NVIDIA GeForce RTX 2050

Per-job wall-clock limit: 30 seconds. HiGHS: one CPU thread.

Sovereign timings below are solver-reported; timeouts use the enforced wall-clock limit. Full HTTP turnaround is recorded separately in JSON. Single runs, not statistical performance claims.

| Dataset | Source | Algorithm | Sovereign status | Objective | Seconds | Verified | HiGHS status | HiGHS objective | HiGHS seconds | Verified optimum match |
|---|---|---|---|---:|---:|---|---|---:|---:|---|
| afiro | Netlib LP | Dual simplex | OPTIMAL | -464.7531428571429 | 0.0006 | True | OPTIMAL | -464.75314285714285 | 0.0065 | True |
| afiro | Netlib LP | Interior point | OPTIMAL | -464.7531428511402 | 0.0006 | True | OPTIMAL | -464.75314285714285 | 0.0065 | True |
| flugpl | MIPLIB official | Branch & cut / strong | OPTIMAL | 1201500.0 | 0.3480 | True | OPTIMAL | 1201500.0 | 0.1310 | True |
| flugpl | MIPLIB official | Branch & bound / fractional | OPTIMAL | 1201500.0 | 0.7183 | True | OPTIMAL | 1201500.0 | 0.1310 | True |
| flugpl | MIPLIB official | Branch & cut / pseudocost | OPTIMAL | 1201500.0 | 0.6823 | True | OPTIMAL | 1201500.0 | 0.1310 | True |
| gt2 | MIPLIB official | Branch & cut / strong | OPTIMAL | 21166.0 | 20.7104 | True | OPTIMAL | 21166.0 | 0.0736 | True |
| gt2 | MIPLIB official | Branch & bound / fractional | FEASIBLE | 53344.0 | 27.0381 | True | OPTIMAL | 21166.0 | 0.0736 | False |
| gt2 | MIPLIB official | Branch & cut / pseudocost | OPTIMAL | 21166.0 | 2.8687 | True | OPTIMAL | 21166.0 | 0.0736 | True |
| b-ball | MIPLIB official | Branch & cut / strong | FEASIBLE | -1.5 | 19.0228 | True | OPTIMAL | -1.500001 | 0.2005 | False |
| b-ball | MIPLIB official | Branch & bound / fractional | FEASIBLE | -1.5 | 18.8068 | True | OPTIMAL | -1.500001 | 0.2005 | False |
| b-ball | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | -1.5 | 18.8277 | True | OPTIMAL | -1.500001 | 0.2005 | False |
| pk1 | MIPLIB official | Branch & cut / strong | FEASIBLE | 11.999999999999998 | 27.0350 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0043 | False |
| pk1 | MIPLIB official | Branch & bound / fractional | FEASIBLE | 11.999999999999998 | 27.0385 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0043 | False |
| pk1 | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | 12.0 | 27.0223 | True | TIME_LIMIT_REACHED | 13.999999999999915 | 30.0043 | False |
| gen-ip016 | MIPLIB official | Branch & cut / strong | FEASIBLE | -9430.336616408 | 27.0340 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0044 | False |
| gen-ip016 | MIPLIB official | Branch & bound / fractional | FEASIBLE | -9441.184307408 | 22.7165 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0044 | False |
| gen-ip016 | MIPLIB official | Branch & cut / pseudocost | FEASIBLE | -9432.954064407999 | 27.0356 | True | TIME_LIMIT_REACHED | -9421.387448424 | 30.0044 | False |
| kuhn_degeneracy | Synthetic robustness | Dual simplex | OPTIMAL | -1.25 | 0.0001 | True | OPTIMAL | -1.25 | 0.0009 | True |
| kuhn_degeneracy | Synthetic robustness | Interior point | OPTIMAL | -1.2499999999982918 | 0.0003 | True | OPTIMAL | -1.25 | 0.0009 | True |
| illconditioned | Synthetic robustness | Dual simplex | OPTIMAL | 30000000.0 | 0.0001 | True | OPTIMAL | 30000000.0 | 0.0009 | True |
| illconditioned | Synthetic robustness | Interior point | OPTIMAL | 29999999.99967302 | 0.0003 | True | OPTIMAL | 30000000.0 | 0.0009 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / strong | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0006 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & bound / fractional | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0006 | True |
| weak_lp_relaxation | Synthetic robustness | Branch & cut / pseudocost | OPTIMAL | 100.0 | 0.0003 | True | OPTIMAL | 100.0 | 0.0006 | True |
| transport_20x20 | Synthetic scale | Dual simplex | OPTIMAL | 201.99999999999991 | 0.0008 | True | OPTIMAL | 201.99999999999994 | 0.0021 | True |
| transport_20x20 | Synthetic scale | Interior point | OPTIMAL | 202.00000000080883 | 0.0010 | True | OPTIMAL | 201.99999999999994 | 0.0021 | True |
| transport_50x50 | Synthetic scale | Dual simplex | OPTIMAL | 504.6000000000004 | 0.0067 | True | OPTIMAL | 504.6000000000004 | 0.0071 | True |
| transport_50x50 | Synthetic scale | Interior point | OPTIMAL | 504.6000000036868 | 0.0070 | True | OPTIMAL | 504.6000000000004 | 0.0071 | True |
| transport_100x100 | Synthetic scale | Dual simplex | OPTIMAL | 1009.0000000000014 | 0.0654 | True | OPTIMAL | 1009.0000000000015 | 0.0353 | True |
| transport_100x100 | Synthetic scale | Interior point | OPTIMAL | 1009.0000000400751 | 0.0345 | True | OPTIMAL | 1009.0000000000015 | 0.0353 | True |
| transport_150x150 | Synthetic scale | Dual simplex | OPTIMAL | 1513.3999999999976 | 0.3068 | True | OPTIMAL | 1513.3999999999976 | 0.1184 | True |
| transport_150x150 | Synthetic scale | Interior point | OPTIMAL | 1513.4000004520094 | 0.1014 | True | OPTIMAL | 1513.3999999999976 | 0.1184 | True |
| transport_200x200 | Synthetic scale | Dual simplex | OPTIMAL | 2017.7999999999934 | 0.9443 | True | OPTIMAL | 2017.7999999999936 | 0.5274 | True |
| transport_200x200 | Synthetic scale | Interior point | OPTIMAL | 2017.800000009963 | 0.2115 | True | OPTIMAL | 2017.7999999999936 | 0.5274 | True |
| industrial_refinery_lp | Industrial example | Dual simplex | OPTIMAL | 675000.0 | 0.0001 | True | OPTIMAL | 675000.0 | 0.0009 | True |
| industrial_refinery_lp | Industrial example | Interior point | OPTIMAL | 674999.9999667553 | 0.0003 | True | OPTIMAL | 675000.0 | 0.0009 | True |
| industrial_blending_lp | Industrial example | Dual simplex | OPTIMAL | 7200.0 | 0.0001 | True | OPTIMAL | 7200.0 | 0.0010 | True |
| industrial_blending_lp | Industrial example | Interior point | OPTIMAL | 7200.000000026874 | 0.0004 | True | OPTIMAL | 7200.0 | 0.0010 | True |
| industrial_power_dispatch_lp | Industrial example | Dual simplex | OPTIMAL | 2700.0 | 0.0001 | True | OPTIMAL | 2700.0 | 0.0005 | True |
| industrial_power_dispatch_lp | Industrial example | Interior point | OPTIMAL | 2700.000000099279 | 0.0004 | True | OPTIMAL | 2700.0 | 0.0005 | True |
| industrial_logistics_milp | Industrial example | Branch & cut / strong | OPTIMAL | 550.0 | 0.0004 | True | OPTIMAL | 550.0 | 0.0128 | True |
| industrial_logistics_milp | Industrial example | Branch & bound / fractional | OPTIMAL | 550.0 | 0.0003 | True | OPTIMAL | 550.0 | 0.0128 | True |
| industrial_logistics_milp | Industrial example | Branch & cut / pseudocost | OPTIMAL | 550.0 | 0.0003 | True | OPTIMAL | 550.0 | 0.0128 | True |
| sample_qp | QP example | QP interior point | OPTIMAL | -2.0 | 0.0002 | True | OPTIMAL | -1.999999999999995 | 0.0019 | True |
| sample_qp | QP example | Frank-Wolfe | OPTIMAL | -2.0 | 0.0005 | True | OPTIMAL | -1.999999999999995 | 0.0019 | True |
| qp_ge | QP example | QP interior point | OPTIMAL | 1.9999999996220788 | 0.0001 | True | OPTIMAL | 1.9999999999999998 | 0.0008 | True |
| qp_ge | QP example | Frank-Wolfe | OPTIMAL | 2.0 | 0.0001 | True | OPTIMAL | 1.9999999999999998 | 0.0008 | True |
| plan_600x900 | GPU showcase | Dual simplex | OPTIMAL | 37175.761943396574 | 0.1944 | True | OPTIMAL | 37175.761943396436 | 0.0671 | True |
| plan_600x900 | GPU showcase | Interior point | OPTIMAL | 37175.761931149886 | 0.2270 | True | OPTIMAL | 37175.761943396436 | 0.0671 | True |
| plan_1000x1500 | GPU showcase | Dual simplex | OPTIMAL | 59174.29816757092 | 0.9312 | True | OPTIMAL | 59174.298167570734 | 0.2681 | True |
| plan_1000x1500 | GPU showcase | Interior point | OPTIMAL | 59174.298167497785 | 1.0785 | True | OPTIMAL | 59174.298167570734 | 0.2681 | True |
| plan_1500x2200 | GPU showcase | Dual simplex | OPTIMAL | 91500.1974783319 | 2.9467 | True | OPTIMAL | 91500.19747833154 | 0.7100 | True |
| plan_1500x2200 | GPU showcase | Interior point | OPTIMAL | 91500.19742444395 | 6.2427 | True | OPTIMAL | 91500.19747833154 | 0.7100 | True |
| portfolio_qp_600 | GPU showcase | QP interior point | OPTIMAL | -0.1849816431569199 | 0.0421 | True | OPTIMAL | -0.1849816443799799 | 0.0119 | True |
| portfolio_qp_600 | GPU showcase | Frank-Wolfe | FEASIBLE | -0.1844093054520936 | 0.4047 | True | OPTIMAL | -0.1849816443799799 | 0.0119 | False |
| staircase_20x100 | Sparse scale | Dual simplex | OPTIMAL | 21329.7060606061 | 0.1006 | True | OPTIMAL | 21329.70606060608 | 0.0240 | True |
| staircase_20x100 | Sparse scale | Interior point | OPTIMAL | 21329.70606259925 | 0.0202 | True | OPTIMAL | 21329.70606060608 | 0.0240 | True |
| staircase_50x200 | Sparse scale | Dual simplex | OPTIMAL | 106185.7357575748 | 2.8525 | True | OPTIMAL | 106185.73575757477 | 0.1742 | True |
| staircase_50x200 | Sparse scale | Interior point | OPTIMAL | 106185.73575779606 | 0.1503 | True | OPTIMAL | 106185.73575757477 | 0.1742 | True |
| portfolio_qp_5000 | Sparse scale | QP interior point | OPTIMAL | -0.1944651286560086 | 0.0516 | True | OPTIMAL | -0.19446512875558752 | 0.3140 | True |
| portfolio_qp_5000 | Sparse scale | Frank-Wolfe | TIME_LIMIT | None | 30.0000 | False | OPTIMAL | -0.19446512875558752 | 0.3140 | False |

## Coverage and limits

- Netlib: AFIRO. Official MIPLIB: flugpl, gt2, b-ball, pk1, gen-ip016 (original MPS passed independently to each solver).
- Synthetic robustness: degeneracy, ill-conditioning, weak relaxation. Synthetic scale: transport LPs to 40,000 variables, planning LPs to 1,500 rows x 2,200 variables, staircase LPs to 10,200 rows x 20,000 variables and portfolio QPs to 5,000 assets.
- Repository industrial examples: refinery, blending, power dispatch, logistics. These are not claimed as published or proprietary industrial data.
- QP examples compared with HiGHS using the quadratic Hessian, not an LP relaxation.
- Mittelmann and QPLIB instances are not bundled or tested. Million-variable scale is shown only on synthetic structured LP/QP (scale-ladder.md).
- GPU: measured separately in gpu-dense-ipm.md. CUDA interior point is 1.6x faster only on the 1,500-row dense planning LP; the sparse CPU path wins on the other models tested.
- Only OPTIMAL + independent verification + reference OPTIMAL + objective tolerance agreement counts as a match. Failed verification and timeouts remain visible.
- HiGHS is used only in this benchmark harness, never to solve production jobs.
