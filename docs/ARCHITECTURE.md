# Architecture and Build Guide

> This document describes the legacy local release. The current hosted workspace
> uses this repository for the backend and the sibling `sovereign-frontend/`
> folder for the website. See `VERCEL_DEPLOYMENT.md` for the split deployment.

## Status

Sovereign — from-scratch LP / MILP / QP core, packaged as a local-first
application.

| Area | Status |
|------|--------|
| Sparse CSC/CSR, revised simplex LP | Done |
| Presolve + postsolve | Done |
| MILP branch-and-bound + branch-and-cut | Done (small scale only) |
| Convex QP (Mehrotra IPM) | Done |
| Independent solution verification | Done |
| Benchmark harness + evidence pack | Done |
| Optional Azure OpenAI orchestration | Done (optional, never on the solve path) |
| Industrial demos | Done |
| GPU | **No CUDA kernels exist.** SpMV hook only, and it honestly declines to run |
| **Offline asset bundling** | **Done** — enforced by `check_offline_assets.py` |
| **Local C++ server, launcher, job manager, auth** | **Not started.** See `ROADMAP.md` §2 and `docs/auth.md` |

### Honest gaps

| Requirement | Reality |
|-------------|---------|
| Industrial scale (10⁵–10⁶ variables) | **Not met.** Dense O(m³) basis factorization is the hard blocker. See `ROADMAP.md` §3 |
| Full parallel branch-and-bound | **Partial.** Only the two strong-branching child LPs run concurrently, and it measured *slower* than serial |
| GPU acceleration | **Not implemented** |
| MIQP / NLP / MINLP | **Not implemented.** No claim is made |
| MPS parser | **Not implemented.** JSON reader only. Required by the offline product |
| MILP optimality proving at scale | **Not met.** Official MIPLIB instances hit the time limit |

For reference, NVIDIA cuOpt's own README states its MIP solver excels at finding
good feasible solutions but that "proving feasible solutions optimal remains
under active development." Matching a mature GPU-native effort is not a
realistic near-term goal, and the specification's success criteria do not
require it.

### Layers

| Layer | Role | Ships to user? |
|-------|------|----------------|
| `solver/` | From-scratch C++ optimization core. **No auth, no HTTP, no web dependency** | Yes |
| `solver/server/` | Local C++ HTTP server (planned) | Yes |
| `solver/launcher/` | Process lifecycle, browser open (planned) | Yes |
| `web/` | React local UI, bundled offline | Yes |
| `api/server.py` | Python bridge — **development only**, superseded by `solver/server` | No |
| `api/main.py` | FastAPI tool boundary for the optional LLM agent | No |
| `agent/` | Python schemas, tools, optional OpenAI orchestration | No |
| `benchmarks/` | External comparison harness only. The only place HiGHS may appear | No |
| `tests/` | Unit / LP / MILP / QP / correctness-regression suites | No |
| `docs/` | Architecture, mathematics, benchmarks, auth | No |

**Hard rule:** the production solver never calls HiGHS / SCIP / CBC / GLPK /
Gurobi / CPLEX / OR-Tools / SciPy optimize. They appear only in `benchmarks/`.

**Hard rule:** the LLM is orchestration only. The verifier is the authority on
numerical validity.

**Hard rule:** `sovereign_core` never links authentication, HTTP, or web code.
`sovereign solve model.mps` must keep working with no credential, no server and
no network. A test enforces this.

## Build

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run

```powershell
# CLI — always works, no server, no network, no auth
.\build\solver\sovereign.exe solve examples\models\sample_lp.json --verify
.\build\solver\sovereign.exe solve examples\models\sample_milp.json --verify
.\build\solver\sovereign.exe solve examples\models\sample_qp.json --verify

python examples\run_industrial_demos.py
python benchmarks\runners\run_benchmarks.py
```

## Web UI

```powershell
cd web
npm install
npm run dev            # http://localhost:5173, proxies /api to a running server
npm run build          # emits to ../api/static
```

### Offline guarantee

The UI must render and operate with the network disconnected. This is enforced,
not assumed:

```powershell
cd web; npm run build
python benchmarks\tools\check_offline_assets.py    # fails on any remote reference
```

Fonts are self-hosted in `web/public/fonts` (latin + latin-ext only, 664 KB).
The hero video is optional and degrades to a static gradient when no local copy
exists — see `web/src/lib/heroMedia.ts`. **Do not reintroduce a Google Fonts
`@import`, a CDN `<script>`, or a remote image; the audit will fail the build.**

## Local-first architecture (planned — see `ROADMAP.md` §2)

```
        sovereign-launcher.exe
                 │
                 ├─ detect a running instance (lockfile + port probe)
                 ├─ spawn sovereign-server.exe --port 0
                 ├─ read chosen port
                 └─ open http://127.0.0.1:<port> in the default browser
                            │
                            ▼
                 Sovereign Web UI  (bundled, offline)
                            │  REST + SSE, loopback only
                            ▼
                 sovereign-server.exe
                    ├─ AuthManager ──► SecureCredentialStore ──► Windows DPAPI
                    ├─ JobManager (thread pool, async solve, progress events)
                    └─ routes: /api/auth, /api/jobs, /api/system, /api/problems
                            │
                            ▼
                 sovereign_core        ◄─── the same core the CLI uses
                    LP / MILP / QP
                            │
                            ▼
                 CPU backend  |  CUDA backend (not yet implemented)
```

Two processes, no runtime dependencies. The web UI is *an interface to the
engine, not the engine*. Every objective displayed in the dashboard is produced
by `sovereign_core`, which has no knowledge that HTTP exists.

This is a **local application plus a localhost server plus an automatically
opened browser**. It is deliberately *not* a captive portal: no Wi-Fi control,
no DNS interception, no traffic redirection, no network gateway.

## Separation of concerns

```text
CLI ─────────────┐
                 │
Local Web UI ────┼──►  sovereign_core  ──►  Presolve ──► LP / MILP / QP
                 │            │
Future API ──────┘            └─► SolutionVerifier (independent of the algorithm)
```
