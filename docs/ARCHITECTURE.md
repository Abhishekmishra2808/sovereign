# Architecture and Build Guide

## Status (v1.0)

Sovereign AI Mathematical Optimization Platform — from-scratch LP / MILP / QP core.

| Phase | Status |
|-------|--------|
| 0 Architecture | Done |
| 1 Sparse + revised simplex LP | Done |
| 2 Presolve | Done |
| 3 MILP branch-and-bound | Done |
| 4 Cuts + heuristics | Done |
| 5 Convex QP | Done |
| 6 Strengthened verification | Done |
| 7 Benchmark harness | **Evidence pack live** — Netlib AFIRO + HiGHS compare; see `benchmarks/reports/EVIDENCE.md` |
| 8 Azure GPT-5.6 agent | Done (env-configured; pitch as differentiator, not scored core) |
| 9 Industrial demos | Done (`examples/models/industrial_*`) |
| 10 GPU | Hooks + CPU SpMV (CUDA optional, evidence-gated) |

### Known gaps vs problem statement (honest)

| Requirement | Status |
|-------------|--------|
| Interior-point methods | **Done** — Mehrotra primal-dual IPM (`SOVEREIGN_LP_ALGORITHM=ipm\|auto\|simplex`) |
| Multi-core parallelization | **Done** — parallel strong-branch child LP solves (Win32 threads) |
| Tree-wide branch-and-cut | **Done** — cuts at nodes by `cut_frequency` (not root-only) |
| Advanced branching (strong / pseudo-cost) | **Done** — strong (default) + pseudo-cost + most-fractional |
| Million-variable scale | Not claimed; simplex **and IPM** solve 10k-var transport (IPM ~0.9s, simplex ~3s) |

| Layer | Role |
|-------|------|
| `solver/` | From-scratch C++ optimization core |
| `agent/` | Python schemas, tools, GPT-5.6 orchestration |
| `api/` | FastAPI tool/HTTP boundary |
| `benchmarks/` | External comparison harness only |
| `tests/` | Unit / LP / MILP / QP suites |
| `docs/` | Architecture and phase plan |

**Hard rule:** production solver never calls HiGHS/SCIP/CBC/GLPK/Gurobi/CPLEX/OR-Tools/SciPy optimize.

**Hard rule:** GPT-5.6 is orchestration only; the verifier is the authority on numerical validity.

## Build

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run

```powershell
.\build\solver\sovereign.exe solve examples\models\sample_lp.json --verify
.\build\solver\sovereign.exe solve examples\models\sample_milp.json --verify
.\build\solver\sovereign.exe solve examples\models\sample_qp.json --verify
python examples\run_industrial_demos.py
python benchmarks\runners\run_benchmarks.py
```

## Agent CLI (interactive)

```powershell
$env:PYTHONPATH = (Get-Location).Path
.\sov.cmd
```

Slash commands: `/help` `/load` `/solve` `/verify` `/math` `/examples` `/status` `/exit`

One-shot: `.\sov.cmd examples\models\sample_lp.json --verify`

## Agent / API


```powershell
$env:PYTHONPATH = (Get-Location).Path
# Optional LLM:
# $env:AZURE_OPENAI_ENDPOINT="https://YOUR.openai.azure.com"
# $env:AZURE_OPENAI_API_KEY="..."
# $env:AZURE_OPENAI_DEPLOYMENT="gpt-5.6"
uvicorn api.main:app --port 8000
```

## Separation of concerns

```text
User / GPT-5.6 agent
        │
        ▼
Python ToolClient / FastAPI
        │  structured JSON OptimizationModel
        ▼
sovereign CLI  →  Presolve → LP / MILP / QP
        │
        └── SolutionVerifier
```
