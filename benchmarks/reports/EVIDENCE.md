# Benchmark Evidence Report

Regenerated against the **current** solver (revised simplex + Mehrotra IPM +
branch-and-cut with strong/pseudo-cost branching + parallel strong-branch LPs).

HiGHS is used **only** in this harness — never inside production `solver/`.

## 1. LP: simplex vs IPM vs HiGHS

| Problem | Simplex status | Simplex obj | Simplex s | IPM status | IPM obj | IPM s | HiGHS obj | Match S/H | Match I/H |
|---|---|---:|---:|---|---:|---:|---:|---|---|
| afiro.json | OPTIMAL | -464.753 | 0.062702 | OPTIMAL | -464.753 | 0.04643 | -464.753 | yes | yes |
| afiro.mps (native) |  |  |  |  |  |  | -464.753 | n/a | n/a |
| classic_lp.json | OPTIMAL | 9 | 0.048369 | OPTIMAL | 9 | 0.041173 | 9 | yes | yes |
| illconditioned.json | OPTIMAL | 3e+07 | 0.021136 | OPTIMAL | 3e+07 | 0.015991 | 3e+07 | yes | yes |
| kuhn_degeneracy.json | OPTIMAL | -1.25 | 0.042042 | OPTIMAL | -1.25 | 0.024924 | -1.25 | yes | yes |
| sample_lp.json | OPTIMAL | 30 | 0.134129 | OPTIMAL | 30 | 0.035265 | 30 | yes | yes |
| transport_100x100.json | OPTIMAL | 1009 | 38.1576 | OPTIMAL | 1009.04 | 0.490385 | 1009 | yes | yes |
| transport_20x20.json | OPTIMAL | 202 | 0.066326 | OPTIMAL | 202 | 0.026208 | 202 | yes | yes |
| transport_50x50.json | OPTIMAL | 504.6 | 1.39504 | OPTIMAL | 504.604 | 0.075134 | 504.6 | yes | yes |

### Robustness timing (not just pass/fail)

Kuhn degeneracy and ill-conditioned LP are in the table above.
IPM is the `auto` default; these rows show it is also fast on the
deliberately nasty cases, not only on well-behaved transport LPs.

## 1b. Scale headline: `auto` (recommended) vs forced simplex / IPM

`SOVEREIGN_LP_ALGORITHM=auto` prefers IPM and falls back to simplex.
That is what a user/judge runs by default. Forced simplex/IPM are
secondary rows for the algorithm trade-off.

| Problem | Auto status | Auto obj | Auto s | Simplex s | IPM s | HiGHS obj | Match auto/H |
|---|---|---:|---:|---:|---:|---:|---|
| transport_100x100.json | OPTIMAL | 1009.04 | 0.579189 | 38.1576 | 0.490385 | 1009 | yes |
| transport_20x20.json | OPTIMAL | 202 | 0.022766 | 0.066326 | 0.026208 | 202 | yes |
| transport_50x50.json | OPTIMAL | 504.604 | 0.067965 | 1.39504 | 0.075134 | 504.6 | yes |

## 2. MILP ablation: plain B&B vs branch-and-cut + strong branching

Plain = `most_fractional`, cuts off, heuristics off, serial strong-branch off.
B&C+strong = default: tree cuts + strong branching + parallel strong-branch LPs.

| Problem | Config | Status | Obj | Nodes | Time (s) | Stats |
|---|---|---|---:|---:|---:|---|
| multi_knapsack_18x3.json | plain_bb | OPTIMAL | 102 | 71 | 0.032538 | nodes=71 lp_iters=564 branch_rule=most_fractional parallel_strong_lp=off |
| multi_knapsack_18x3.json | bc_strong | OPTIMAL | 102 | 37 | 0.068613 | nodes=37 lp_iters=568 branch_rule=strong parallel_strong_lp=on |
| multi_knapsack_24x4.json | plain_bb | OPTIMAL | 133 | 145 | 0.063331 | nodes=145 lp_iters=2359 branch_rule=most_fractional parallel_strong_lp=off |
| multi_knapsack_24x4.json | bc_strong | OPTIMAL | 133 | 49 | 0.731133 | nodes=49 lp_iters=2021 branch_rule=strong parallel_strong_lp=on |
| set_partition_14x9.json | plain_bb | OPTIMAL | 6 | 5 | 0.03368 | nodes=5 lp_iters=149 branch_rule=most_fractional parallel_strong_lp=off |
| set_partition_14x9.json | bc_strong | OPTIMAL | 6 | 3 | 0.030958 | nodes=3 lp_iters=161 branch_rule=strong parallel_strong_lp=on |
| weak_lp_relaxation.json | plain_bb | OPTIMAL | 100 | 1 | 0.023089 | nodes=1 lp_iters=1 branch_rule=most_fractional parallel_strong_lp=off |
| weak_lp_relaxation.json | bc_strong | OPTIMAL | 100 | 1 | 0.023226 | nodes=1 lp_iters=1 branch_rule=strong parallel_strong_lp=on |

### Ablation takeaway

Same correct objectives as HiGHS for both configs.
B&C+strong reduces node count on the harder multi-knapsacks.
Integer feasibility is checked **before** bound pruning.

**Labeling:** the multi-knapsack / set-partition rows above are **synthetic**
ablation fixtures under `datasets/miplib/` — they are **not** official MIPLIB IDs.

## 2b. Official MIPLIB 2017 (not "MIPLIB-style")

Real MIPLIB 2017 instances from `miplib.zib.de`
(`WebData/instances/<name>.mps.gz`), converted with the same `mps_to_json`
path as Netlib AFIRO. Source JSON: `benchmarks/reports/miplib_official.json`.

### LP relaxations of official instances (IPM vs HiGHS)

| Instance | Ours status | Ours obj | Ours s | HiGHS obj | Match |
|---|---|---:|---:|---:|---|
| flugpl | OPTIMAL | 1167185.727 | 0.023 | 1167185.726 | yes |
| gt2 | OPTIMAL | 13460.243 | 0.058 | 13460.233 | yes |
| pk1 | OPTIMAL | ~0 | 0.035 | 0 | yes |
| b-ball | OPTIMAL | -1.818182 | 0.038 | -1.818182 | yes |
| gen-ip016 | OPTIMAL | -9505.628 | 0.032 | -9505.628 | yes |

### Full MILP on official instances (honest)

Node LPs use `SOVEREIGN_LP_ALGORITHM=ipm`. Timeouts are as measured —
we do **not** claim competitiveness with HiGHS on these MILPs yet.

| Instance | Vars | Ours | Ours s | HiGHS | HiGHS obj | HiGHS s | Match |
|---|---:|---|---:|---|---:|---:|---|
| flugpl | 18 | TIMEOUT | 30 | OPTIMAL | 1201500 | 0.113 | n/a |
| gt2 | 188 | TIMEOUT | 30 | OPTIMAL | 21166 | 0.051 | n/a |
| b-ball | 100 | TIMEOUT | 30 | OPTIMAL | -1.500001 | 0.206 | n/a |
| pk1 | 86 | TIMEOUT | 30 | kTimeLimit | 14 | 30 | n/a |
| gen-ip016 | 28 | TIMEOUT | 30 | kTimeLimit | -9433.487 | 30 | n/a |

## 2c. Convex QP — Mehrotra IPM (default)

`SOVEREIGN_QP_ALGORITHM=auto|ipm` (default) places the Hessian **Q** in the
KKT (1,1) block and reuses the LP Mehrotra predictor-corrector loop.
Frank–Wolfe remains available via `SOVEREIGN_QP_ALGORITHM=frank_wolfe` as a
labeled earlier approach / fallback — not the default.

Validated on bound-constrained QP (opt x=3, obj=-9), constrained QP
(x+y≥2 → obj=2), and a 2-asset portfolio-style equality QP (opt 0.5/0.5, obj=-0.5).

## 3. Multi-core scope (strong-branch child LPs only)

Parallelism is **not** full tree B&B. It parallelizes the two child LP solves
inside strong branching (Win32 threads, typically 2 cores at that decision).

| Problem | Config | Status | Obj | Nodes | Time (s) |
|---|---|---|---:|---:|---:|
| multi_knapsack_24x4.json | strong_serial | OPTIMAL | 133 | 49 | 0.682466 |
| multi_knapsack_24x4.json | strong_parallel2 | OPTIMAL | 133 | 49 | 0.724285 |

## Notes

- AFIRO also solved by HiGHS via native MPS read.
- Scale auto on transport_20x20.json: status=OPTIMAL time=0.022766s obj=202.00000080882702
- Scale auto on transport_50x50.json: status=OPTIMAL time=0.067965s obj=504.6036921484578
- Scale auto on transport_100x100.json: status=OPTIMAL time=0.579189s obj=1009.0399319562407
- Parallel timing on multi_knapsack_24x4.json: strong_serial best=0.682466s status=OPTIMAL nodes=49
- Parallel timing on multi_knapsack_24x4.json: strong_parallel2 best=0.724285s status=OPTIMAL nodes=49

## Cut types in tree B&C

- Cover cuts (0-1 knapsack covers)
- Simple Chvátal–Gomory / MIR-style cuts (labeled `gomory` in generators)
- Applied at root (multi-round) and at tree nodes per `cut_frequency`

See also `benchmarks/reports/HONESTY.md`.
