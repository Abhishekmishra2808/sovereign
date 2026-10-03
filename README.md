> **Online workspace:** the website now lives in the separate sibling `sovereign-frontend/` folder. This Git repository contains the backend coordinator and solver. Deploy the two projects using [Vercel with Postgres](VERCEL_DEPLOYMENT.md) or the backend on [Render](DEPLOYMENT.md). Install the worker package on your laptop or GPU server with `python -m pip install .`; the worker connects outward over HTTPS and runs the solver locally. The hosted site needs no GPU or solver binary.
# Sovereign AI Mathematical Optimization Platform

From-scratch **LP / MILP / QP** solver with an interactive agent CLI (Claude Code / Cursor-style) and optional Azure OpenAI GPT-5.6.

Docs: [`solution.md`](solution.md) Â· [`agent.md`](agent.md) Â· [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

## Interactive agent CLI

```powershell
# from repo root
$env:PYTHONPATH = (Get-Location).Path
.\sov.cmd
```

Or:

```powershell
python -m agent.cli
```

Feels like an agent REPL:

```text
  sovereign
  Industrial optimization agent CLI

> /examples
> /load examples/models/sample_milp.json
> /solve
> /verify
> /help
```

Natural-language mode (optional LLM):

```powershell
$env:AZURE_OPENAI_ENDPOINT="https://YOUR.openai.azure.com"
$env:AZURE_OPENAI_API_KEY="..."
$env:AZURE_OPENAI_DEPLOYMENT="gpt-5.6"
.\sov.cmd
> Maximize profit for a binary knapsack with capacity 7 ...
```

One-shot (non-interactive):

```powershell
.\sov.cmd examples\models\sample_lp.json --verify
```

## Build the C++ solver

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
.\build\solver\sovereign.exe solve examples\models\sample_milp.json --verify
python examples\run_industrial_demos.py
```

## LISWET1 evidence

The official `LISWET1.SIF` in `benchmarks/data/` has active parameters
`N=2000`, `K=2`, so the real instance has 2000 constraints and 2002
variables. Run it with:

```powershell
python benchmarks/tools/run_liswet1.py `
  --sif benchmarks/data/LISWET1.SIF `
  --binary build64/solver/sovereign.exe `
  --out-dir benchmarks/reports/liswet_real
python benchmarks/tools/verify_liswet_independent.py `
  --c-file benchmarks/reports/liswet_real/c.txt `
  --x-file benchmarks/reports/liswet_real/x.txt
```

The official run is separate from the `n=10000` and `n=20000`
**synthetic-scalability** experiments. Those larger runs are not official
LISWET1 dimensions; their reports live under
`benchmarks/reports/liswet_synthetic_scalability/` when regenerated.

## Non-negotiables

1. Solver algorithms are implemented in this repo from mathematical foundations.
2. No external optimization solver in production code.
3. GPT-5.6 never replaces the numerical engine; verification is independent.

Headline evidence (regenerate anytime):

```powershell
python benchmarks/runners/run_evidence_pack.py
```

Reports: [`benchmarks/reports/EVIDENCE.md`](benchmarks/reports/EVIDENCE.md) Â· [`benchmarks/reports/HONESTY.md`](benchmarks/reports/HONESTY.md)

## LP algorithms

```powershell
$env:SOVEREIGN_LP_ALGORITHM = "auto"     # IPM then simplex fallback (default)
$env:SOVEREIGN_LP_ALGORITHM = "ipm"      # Mehrotra primal-dual interior-point
$env:SOVEREIGN_LP_ALGORITHM = "simplex"  # revised simplex only
```

MILP uses branch-and-cut with **strong branching** (default), tree-node cuts, and **parallel strong-branch LP solves**.

```powershell
pip install highspy
python benchmarks/runners/run_benchmarks.py --suite all --timeout 180
```

Reports: [`benchmarks/reports/EVIDENCE.md`](benchmarks/reports/EVIDENCE.md) Â· [`benchmarks/reports/HONESTY.md`](benchmarks/reports/HONESTY.md) Â· [`benchmarks/reports/latest.csv`](benchmarks/reports/latest.csv)

Regen: `python benchmarks/runners/run_evidence_pack.py`

Headline result: **Netlib AFIRO** optimal objective **-464.753â€¦** matches HiGHS exactly. Named degeneracy / ill-conditioned / weak-relaxation cases also match. Scale: **2500-var transport ~0.3s** and **10k-var ~3s**, both OPTIMAL (product-form basis updates).
