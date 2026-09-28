# Scale ladder

Engine `build64\solver\sovereign.exe` (64-bit, single-threaded), Windows-10-10.0.26200-SP0, 15.7 GB RAM. Generated 2026-09-29T00:59:31+0530.

Wall time includes reading the JSON model; peak memory is the process peak working set. HiGHS runs single-threaded as a reference objective only.

| Model | Type | Variables | Rows | Nonzeros | Status | Objective | Verified | Wall s | Solver s | Load s | Peak MB | IPM iters | HiGHS status | HiGHS s | Rel. gap |
|---|---|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---|---:|---:|
| lp_staircase_100k | LP | 100,000 | 50,500 | 199,900 | OPTIMAL | 529526.1335 | yes | 3.2 | 2.0 | 0.6 | 152 | 16 | OPTIMAL | 2.5 | 1.6e-12 |
| qp_portfolio_100k | QP | 100,000 | 2,001 | 870,266 | OPTIMAL | -0.1986944505 | yes | 5.8 | 3.1 | 1.7 | 254 | 15 | OPTIMAL | 203.0 | 1.8e-11 |
| lp_staircase_1m | LP | 1,000,000 | 501,000 | 1,999,500 | OPTIMAL | 5297439.673 | yes | 93.3 | 76.5 | 8.4 | 1452 | 20 | OPTIMAL | 70.9 | 1.3e-12 |
| qp_portfolio_1m | QP | 1,000,000 | 20,001 | 8,702,560 | OPTIMAL | -0.1995718961 | yes | 75.6 | 46.2 | 18.4 | 2474 | 14 | not run | - | - |

Model files (regenerate with `python benchmarks/tools/generate_scale_ladder.py <name>`):

- `lp_staircase_100k`: 16.3 MB, sha256 `8cfff79dd6dbc2c9cdf7caaa8cea75e5dcbffb40591d83cad11ae12d55bbb287`
- `qp_portfolio_100k`: 23.7 MB, sha256 `1adde9ba8728554fa4deaa5fd017e9c93623ead89494bf0e9b7c2a3624bee22b`
- `lp_staircase_1m`: 167.0 MB, sha256 `c07184bcd0707cfecb469f1a4602ddd95d51b5dbe9c3d212b767adb9aa45ee78`
- `qp_portfolio_1m`: 248.8 MB, sha256 `c9902682c6c1399d430fa773f6eb9ebe59ca286b6b6ce1acd132e4a061214502`

Solver messages:

- `lp_staircase_100k`: Optimal solution found by primal-dual interior-point (Mehrotra). Normal equations: sparse LDL^T with minimum-degree ordering (nnz(L) = 515159, dense triangle 1270105200).
- `qp_portfolio_100k`: Optimal convex QP found by Mehrotra predictor-corrector IPM. KKT system: sparse quasi-definite LDL^T with minimum-degree ordering (order 104001, nnz(L) = 989944, dense triangle 5408156001).
- `lp_staircase_1m`: Optimal solution found by primal-dual interior-point (Mehrotra). Normal equations: sparse LDL^T with minimum-degree ordering (nnz(L) = 5911616, dense triangle 125250375250).
- `qp_portfolio_1m`: Optimal convex QP found by Mehrotra predictor-corrector IPM. KKT system: sparse quasi-definite LDL^T with minimum-degree ordering (order 1040001, nnz(L) = 9896877, dense triangle 540801560001).
