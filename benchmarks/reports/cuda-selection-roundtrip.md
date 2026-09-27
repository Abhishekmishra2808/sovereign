# Explicit CUDA job acceptance

The dashboard previously displayed "This solve did not execute GPU kernels"
for any job whose requested device was not CPU, including **Automatic** jobs
that the coordinator correctly routed to CPU. It also allowed an explicit
CUDA job to complete with zero kernels when presolve finished the model.

The worker now keeps the user's presolve setting for its first solve. If an
explicit CUDA job returns a feasible/optimal answer with zero CUDA operations,
it retries once with presolve disabled under the **same job time limit**. It
records the effective setting and a warning. An explicit CUDA job with zero
kernels after the retry is marked failed, while its solver answer and
verification remain visible for inspection. CPU-only simplex/Frank-Wolfe
choices are rejected at submission for CUDA jobs. The result panel only shows
a zero-kernel CUDA warning when CUDA was requested or assigned.

Using the real coordinator claim, CUDA-enabled worker executable and
`/api/worker/complete` endpoint on this PC:

| Model | State | Solver status | CUDA operations | Effective presolve | Verification |
|---|---|---|---:|---|---|
| ordinary LP, Automatic | COMPLETED | OPTIMAL | 0, routed to CPU | on | passed |
| ordinary LP | COMPLETED | OPTIMAL | 17 | on | passed |
| LP fixed by presolve | COMPLETED | OPTIMAL | 14 | off after retry | passed |
| unconstrained LP | FAILED | OPTIMAL answer retained | 0 | on; retry also zero | passed primal check |

The [captured responses](cuda-selection-roundtrip.json) include routing,
status, objective, GPU count, verification and failure reason. The final row
is intentionally failed: an unconstrained LP has no sparse matrix operation
for this CUDA implementation to perform, so accepting it as a GPU solve would
be misleading. The Automatic row is correctly routed to CPU and does not show
a CUDA failure warning in the dashboard. These checks validate GPU *use*, not
a GPU performance benefit. Reproduce them with
`python tests/scripts/run_cuda_selection_roundtrip.py` from the repository root
using a CUDA-enabled build at `build-gpu-real/solver/Release/sovereign.exe`.
