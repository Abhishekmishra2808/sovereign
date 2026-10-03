# LISWET structured projection

Sovereign recognizes the identity-Hessian convex-sequence projection

`min 0.5 * ||x-c||²` subject to `x_i - 2x_(i+1) + x_(i+2) >= 0`.

## Primary algorithm

The structured path uses knot-space Goldfarb–Idnani in the original `x`
variables. It starts at `x=c`, with an empty active set and zero multipliers.
The active set changes one constraint at a time. Between active constraints,
the solution is represented by implicit piecewise-linear hat functions, so no
dense `A`, `A'A`, or hat matrix is formed. The knot projection uses a
long-double tridiagonal Cholesky solve. Dual directions are recovered by the
second-difference recurrence.

The direction-support guard is numerical-invariant protection, not an
optimality test. Its scaled threshold is

`16 * eps_ld * n² * max(1, ||r||inf, ||lambda||inf)`.

The `n²` effective count covers the recurrence's discrete double-cumulative
sum. A guard trip gets one full state refresh from the active set and one
same-iteration retry. Persistent trips remain failures.

Set `SOVEREIGN_QP_GI_HIGH_PRECISION=1` to evaluate the GI dual and multiplier
recurrences in `__float128`. Tridiagonal solves remain in long double. The flag
is off by default.

## Certification

The independent structured certificate recovers multipliers from `x-c` and
checks:

- primal feasibility: `A x >= 0`;
- dual feasibility: `lambda >= 0`;
- stationarity: `x-c-A'lambda = 0`;
- complementarity: `lambda_i (A x)_i = 0`;
- final-row recurrence consistency and finite values.

The final certificate uses its existing scale
`max(1, ||c||inf, ||A'lambda||inf, ||x||inf)` and tolerance
`1e-8 * scale`. These tolerances are not changed by the GI guard.

## Known limits and evidence

The structured IPM is retained as an independent cross-check, but its
synthetic ladder currently certifies through `n=1000` and fails certification
at `n>=2000`. GI certifies the official SIF instance (`N=2000`, `K=2`,
2002 variables) and the separate synthetic-scalability cases with `n=10000`
and `n=20000` constraints.
When the IPM reaches its early complementarity stop, it runs
`polish_liswet_knots`; the `n=1000` audit reports `polished=1`. Therefore the
published IPM-vs-GI `n=1000` agreement is not an independent algorithmic
cross-check.
Use `benchmarks/tools/run_liswet1.py` with an explicit c-file plus its
`--expected-length`, or with a SIF. It refuses to invent or silently generate
synthetic data. Use
`benchmarks/tools/verify_liswet_independent.py` to check saved `c.txt` and
`x.txt` independently with 50-digit mpmath arithmetic.

The runner also accepts `--sif` for the official LISWET1 SIF data formula:
`c_i=sqrt((i-1)/(N+K-1))+0.1*sin(i)`, with the SIF objective convention
`0.5*sum(x_i^2)-sum(c_i*x_i)`. It honors the active `IE N` and `IE K` lines
(the published file's active default is `N=2000,K=2`, hence 2002 variables;
the 10000 cases in that file are commented alternatives), writes the decoded
`c.txt`, and attempts a PyCUTEst gradient-at-zero cross-check when PyCUTEst
is installed. The downloaded official file was run with `N=2000,K=2`;
PyCUTEst evaluated Sovereign's output as
`p.obj(x)=7.221894982161224` with constraint range
`[-2.22e-16, 2.22e-16]`. The independent 50-digit distance objective was
`7.221894986561684`, a `4.4e-9` difference from double-precision
evaluation; `||grad_SIF(0)+c.txt||inf` was `4.99e-11`. The nonzero objective
gradient is expected at this constrained solution and is not used as an
unconstrained stationarity test.

The mid-size cross-solver script uses CVXOPT's sparse primal QP interior-point
method by default (Clarabel and OSQP are selectable diagnostics). On the
synthetic-scalability `n=1000` and `n=5000` cases, CVXOPT's primal/objective comparison
gave respectively `||x_GI-x_ref||inf = 4.86e-8, 5.63e-8` and relative
objective differences `6.39e-11, 8.48e-12`. CVXOPT labels these solves
`unknown` because its recovered dual residual is poorly scaled at this
second-difference condition number; the primal residuals and the direct
solution/objective comparisons are reported rather than promoted to a
certificate.

## Build and performance

The canonical Windows build is a Release Ninja build with GCC C++17,
`-O3 -DNDEBUG` (the C source also shows `-O2` after the project defaults).
With `SOVEREIGN_QP_GI_PROFILE=1`, one `n=10000` synthetic-scalability run measured about
49.9 seconds on the test machine: 12.18 seconds in direction knot
projections, 11.59 seconds in active-set/state recomputations (including
2.94 seconds in their projections), and 2.77 seconds in the dual recurrence.
The remaining 23.38 seconds is an allocation/vector-scan/other proxy, not a
direct allocator measurement. The profile adds timing instrumentation and is
off by default.

The first optimization candidates are (1) reuse the direction and
recomputation workspaces to remove per-iteration vector allocations, and
(2) fuse or reduce repeated full-length scans during drops and periodic
recomputations. Neither optimization is implemented in this verification
step.
