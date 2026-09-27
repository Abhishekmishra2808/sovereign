# Sovereign — archived local-first plan

The user changed the product to an online Render-hosted website with outbound
compute workers. This local-first plan is historical. The current plan and
status are in [EXECUTION_PLAN.md](EXECUTION_PLAN.md); deployment steps are in
[DEPLOYMENT.md](DEPLOYMENT.md).

Working plan for the local-first application, benchmark programme and release
pipeline. This file exists so progress survives across sessions; `ROADMAP.md`
records *status*, this records *the plan of record*.

Last updated: see git log. Status vocabulary: `[ ]` todo · `[~]` in progress ·
`[x]` done · `[!]` blocked.

---

## Guiding constraints

These come from the problem statement and the product spec, and they are not
negotiable. Every task below is checked against them.

1. **No wrapper.** The solver implements its own algorithms. HiGHS/SCIP/CBC/
   GLPK/Gurobi/CPLEX/OR-Tools/SciPy may appear **only** in `benchmarks/`.
2. **No fabricated results.** If a benchmark did not run, it is not reported. If
   we lose, we report that we lost.
3. **No unproven OPTIMAL.** A status asserts a proof; the verifier is the
   authority. See `SolverResult::duality_gap`.
4. **Offline.** The release must run with the network disconnected. Enforced by
   `benchmarks/tools/check_offline_assets.py`.
5. **Sovereign core is independent.** It must not link auth, HTTP or web code.
   `sovereign solve model.mps` works with no credential, server or network.
6. **MRPL framing without fabrication.** Refinery cases are synthetic and
   literature-derived. We never claim MRPL proprietary data.

---

## Decision: one milestone at a time

Four large tracks were queued simultaneously (local app, auth, refinery demos,
engine performance). Running them in parallel produced partial work in each.
The plan is now strictly sequential, with one milestone in flight at a time.

**Current milestone: M1 — Release executable and one-click local launch.**

Rationale: it is the largest visible gap, it is required by the product spec,
it needs no credentials, and it unblocks the dashboard work. It also gives us
the process architecture (launcher / server / core) that everything else plugs
into.

### Milestone order

| # | Milestone | Depends on | Why this order |
|---|-----------|-----------|----------------|
| **M1** | Release exe + one-click local launch | — | In flight. Unblocks the UI. |
| M2 | C++ MPS parser | — | Required by PS; unblocks real benchmarking |
| M3 | Benchmark datasets + recorded results | M2 | Needs to read the files |
| M4 | MRPL refinery case studies | M2 | Needs the parser to load them |
| M5 | Benchmark visualisation | M3 | Needs data to show |
| M6 | SecureCredentialStore (DPAPI) | M1 | No credentials needed to build it |
| M7 | Offline credential + local session | M6 | Needs the server from M1 |
| M8 | Auth service (separate deploy) | M6 + **user infra** | BLOCKED on credentials |
| M9 | Warm-started node LPs | — | Biggest MILP performance win |
| M10 | Sparse basis factorization | M9 | Multi-week; the scale blocker |
| M11 | CPU/GPU device router | M10 | Only meaningful after profiling |

---

## M1 — Release executable and one-click local launch

**Goal:** install once, then double-click. A local server starts, the default
browser opens to `http://127.0.0.1:<port>`, and the dashboard works. No Python,
Node, CMake or compiler on the user's machine.

### Tasks

- [x] **M1.1** Offline asset bundling. Self-host Inter/Outfit, hero video
      optional. `check_offline_assets.py` passes: 22 files, zero remote refs.
- [x] **M1.2** Vendor `cpp-httplib` (MIT) and `Monocypher` (CC0), licence-reviewed.
- [ ] **M1.3** `solver/server/` — C++ HTTP server, **loopback only**.
      - [ ] M1.3a Route table + static file serving from disk
      - [ ] M1.3b `Host` header validation (blocks DNS rebinding)
      - [ ] M1.3c `Origin` validation on state-changing methods
      - [ ] M1.3d Request body / header / URL caps
      - [ ] M1.3e No `Access-Control-Allow-Origin: *` anywhere
      - [ ] M1.3f Graceful shutdown
- [ ] **M1.4** `JobManager` — async solve, never block the HTTP request.
      - [ ] M1.4a Job states `QUEUED → PRESOLVING → SOLVING → {terminal}`
      - [ ] M1.4b Thread pool sized to cores
      - [ ] M1.4c SSE progress stream
      - [ ] M1.4d Cooperative cancellation
      - [ ] M1.4e Progress payload: runtime, LP iterations, B&B nodes, incumbent,
            best bound, MIP gap, presolve reductions, device, memory
- [ ] **M1.5** `solver/launcher/` — process lifecycle
      - [ ] M1.5a Pick a free ephemeral port, bind loopback
      - [ ] M1.5b Single-instance detection (lockfile + port probe); if already
            running, open the browser at the existing port and exit
      - [ ] M1.5c Launch default browser
      - [ ] M1.5d Surface the chosen port to the launcher
- [ ] **M1.6** `solver/io/mps_io` — CLI accepts `.mps` (blocked on M2, but the
      CLI must not leak a JSON parse error in the meantime)
- [ ] **M1.7** Packaging
      - [ ] M1.7a Windows x64 zip: `sovereign.exe`, `sovereign-server.exe`,
            `sovereign-launcher.exe`, `static/`, `examples/`, `LICENSE`s
      - [ ] M1.7b Inno Setup script → single installer, Start-menu shortcut
      - [ ] M1.7c GitHub Actions release workflow
      - [ ] M1.7d Verify the extracted release runs with networking disabled

### Definition of done for M1

A clean Windows VM, no toolchain, extract the zip, double-click, dashboard loads,
solve an included example, export a solution, and `check_offline_assets.py`
reports zero remote references against the shipped bundle.

---

## M2 — C++ MPS parser

**Goal:** the engine reads the format the entire optimization world speaks.

**Why first among the engine work:** the PS names MPS-format benchmarks, and
right now 20 `.mps` files sit in the repo unreadable by the engine. The CLI
currently throws a raw `nlohmann::json` exception at them.

### Design decisions (from reading cuOpt's `mps_parser.cpp`)

Taken from the reference implementation, verified against their tests:

- **Section detection:** a record starting in column 1 is a section header; data
  records start with a space. `*` and `$` in column 1 are comments and must be
  stripped *before* this test.
- **Fixed vs free format:** one runtime flag. Fixed uses `substr` at documented
  offsets; free uses a `string_view` cursor that allocates nothing per field.
- **Markers are recorded, not applied inline.** Collect `(type, after_col)`
  during the scan, then one sorted pass assigns integer types. This removes a
  whole class of off-by-one bugs and makes the parser trivially chunkable later.
- **Column contiguity is an invariant.** All entries for a column must be
  adjacent; jumping to an already-seen name is an error.
- **Integer default bounds.** An integer variable with no explicit bounds gets
  `[0, 1]`, per CPLEX/SCIP convention.
- **Second `N` row is ignored**, not an error — matches CPLEX/SCIP.
- **`UP` with a negative value and no other bound ⇒ `lb = -inf`.** Needs a
  separate "bounds were defined" bitset, not just lo/hi vectors.
- **`RANGES`** turns a row into an interval: `G`→`[b, b+|R|]`, `L`→`[b-|R|, b]`,
  `E`→ depends on sign of `R`.
- **Build CSC directly.** MPS is column-oriented, so it maps almost 1:1 onto our
  existing `SparseMatrixCSC`.
- **BOUNDS codes:** `UP LO FX FR MI PL BV LI UI SC`. Unknown ⇒ error.

### Tasks

- [ ] M2.1 `string_view` field cursor (no per-field allocation)
- [ ] M2.2 Section state machine + free/fixed field readers
- [ ] M2.3 ROWS (with `N`-row handling) and name→index maps
- [ ] M2.4 COLUMNS with MARKER/INTORG/INTEND position recording
- [ ] M2.5 RHS + objective offset (RHS on the objective row becomes `-offset`)
- [ ] M2.6 BOUNDS, all codes, with the negative-`UP` and `PL` edge cases
- [ ] M2.7 RANGES
- [ ] M2.8 Post-parse: integer default bounds, crossing-bound warning, contiguity
- [ ] M2.9 CLI dispatch on file extension with a friendly error, not a JSON
      exception
- [ ] M2.10 Tests, including a **cross-check against the existing Python parser**
      (`benchmarks/tools/mps_to_json.py`) over all 20 repo `.mps` files. Two
      independent implementations agreeing is a real test; a self-consistent
      parser is not.
- [ ] M2.11 Port cuOpt's edge-case list: CRLF, comments mid-section, no NAME,
      spaces in fixed-format names, two objectives, OBJNAME, unknown bound type,
      variables appearing only in BOUNDS.

### Definition of done
`sovereign solve benchmarks/datasets/**/*.mps` works on all 20 repo files and
produces objectives matching the JSON path. Both formats (fixed and free) parse
identically.

---

## M3 — Benchmark datasets and recorded results

### Datasets to obtain (all public, no registration)

| Dataset | Source | Notes |
|---|---|---|
| **Netlib** | `netlib.org/lp/data.html` | 100+ LPs. The canonical LP suite |
| **MIPLIB 2017** | already have 61 official instances | Verify licence/attribution in the report |
| **Mittelmann** | `cohga.univie.ac.at/Mittelmann/` | LP/QP/MIP stress sets built to expose numerical weakness. **Missing today** |
| **QPLIB** | `qplib.wordpress.com` | Only ~20 active QPs. Scarce — report the small n honestly rather than padding |
| **Robustness set** | construct | Degenerate, ill-conditioned, weak-relaxation, wide-sparse |

A `benchmarks/datasets/FETCH.md` records provenance, licence and a checksum per
file. **No dataset is committed without a recorded source.**

### Tasks
- [ ] M3.1 `fetch_datasets.py` with recorded provenance + checksums
- [ ] M3.2 Netlib full set
- [ ] M3.3 Mittelmann sets
- [ ] M3.4 QPLIB set
- [ ] M3.5 Per-class comparison tolerances (LP 1e-7, QP 1e-6, MILP 1e-4) —
      already in `run_benchmarks.py`
- [ ] M3.6 Record **both** the boolean verdict and the observed relative error.
      `obj_close` returning only yes/no is how the IPM bug stayed hidden.
- [ ] M3.7 Record status, objective, bound, gap, runtime, iterations, nodes,
      peak RSS, device, thread count
- [ ] M3.8 Regenerate `EVIDENCE.md` **including the losses**
- [ ] M3.9 Numerical-robustness section: degeneracy, ill-conditioning, weak
      relaxation, wide sparse — with what breaks and how

### Definition of done
A single command reproduces the whole report, every number traces to a run, and
problems we fail on are listed with the same prominence as wins.

---

## M4 — MRPL refinery case studies

**Framing:** Mangalore Refinery and Petrochemicals Limited, `SIH PS 26119`.
Refinery scheduling, crude blending, production planning, supply chain.

**Hard rule:** synthetic and literature-derived, clearly labelled. We never
claim MRPL data. Models state their source in the file.

### Tasks
- [ ] M4.1 Crude blending LP — assay-based, with property curves and
      nonlinearity linearised by segments. Source: standard refinery
      optimisation literature
- [ ] M4.2 Refinery LP (`industrial_refinery_lp.json` exists — expand and
      document its provenance and assumptions)
- [ ] M4.3 Production planning MILP with unit on/off binaries
- [ ] M4.4 Logistics / supply chain MILP
- [ ] M4.5 **Scheduling MILP** — this is the hard one and the most
      MRPL-relevant. Continuous time, sequence-dependent changeovers,
      tank-capacity state. Representable as MILP via time-indexed binary
      formulation; document the formulation and its size
- [ ] M4.6 Power dispatch LP for co-generation
- [ ] M4.7 Each model: documented units, ranges, cost basis, citation, and an
      `assumptions` block in the JSON
- [ ] M4.8 Verification: brute-force / independent check where the size permits

### Definition of done
A refinery case solves, verifies, and its optimality claim is independently
checked. A reviewer can read the model and see where every number came from.

---

## M5 — Benchmark visualisation

**Decision: horizontal bar chart. Not 3D.**

Bar length is the only length encoding humans read accurately. 3D perspective
and depth cues distort precisely the quantity being compared, which makes a 3D
chart actively misleading for benchmark data. This is not a stylistic
preference; it is an accuracy argument.

Design:
- Horizontal bars, **log scale** for runtime (our times span 0.001 s to timeout)
- One row per problem, Sovereign next to the reference solver
- Colour by outcome: win / tie / loss, with the loss colour clearly distinct
- Sort by problem, not by time, so rows are comparable across runs
- Annotate TIMEOUT and no-solve as their own category, never as "slow"
- **The chart must render from `latest.csv` with no network and no JS charting
  library** — inline SVG, consistent with the offline requirement

---

## M6 — SecureCredentialStore (DPAPI)

- [ ] M6.1 Interface: `save / load / exists / remove`
- [ ] M6.2 `WindowsSecureCredentialStore` via `CryptProtectData`, per-user scope,
      `CRYPTPROTECT_UI_FORBIDDEN`, **not** `CRYPTPROTECT_LOCAL_MACHINE`
- [ ] M6.3 Non-Windows stub so the core stays cross-platform
- [ ] M6.4 Guard: refuse to use DPAPI roaming profiles (blob can be exported to
      the domain and decrypted elsewhere, defeating device binding)
- [ ] M6.5 Tests: round-trip, tampered blob, wrong user, removed store

---

## M7 — Offline credential and local session

- [ ] M7.1 Credential envelope encode/decode (canonical JSON, sorted keys)
- [ ] M7.2 Ed25519 verify — **done**, see `ed25519_verify.cpp`
- [ ] M7.3 Claim validation: device binding, expiry, credential version
- [ ] M7.4 Clock-rollback guard via a monotonic timestamp in the DPAPI blob
- [ ] M7.5 Local session cookie: `HttpOnly`, `SameSite=Strict`, no `Domain`,
      regenerated on login
- [ ] M7.6 CSRF token in a custom header
- [ ] M7.7 `GET /api/auth/me`, `POST /api/auth/logout` (works offline)
- [ ] M7.8 Distinguish **network unavailable** from **authentication failure** in
      both API and UI
- [ ] M7.9 The frontend never receives the long-lived credential

---

## M8 — Auth service · BLOCKED

Blocked on user-provisioned infrastructure. See `docs/auth.md` §Provisioning
checklist. Client-side work (M6, M7) proceeds independently.

---

## M9 — Warm-started node LPs

Currently every B&B node re-solves from a cold logical basis. Real solvers
inherit the parent basis and re-optimize. Biggest available MILP win.

- [ ] M9.1 Thread the parent basis into the node LP
- [ ] M9.2 Dual simplex phase-1 warm start from an infeasible basis
- [ ] M9.3 Bound-change representation so a node is not a full model copy
- [ ] M9.4 Before/after node-count and time comparison on the existing suites

Reference: cuOpt `cpp/src/branch_and_bound/branch_and_bound.hpp`, and
`dual_simplex/basis_solves.cpp` for warm-started phase 1.

---

## M10 — Sparse basis factorization

**The scale blocker.** Dense LU is O(m³) time, O(m²) memory, which is why
`transport_100x100` takes 27 s under forced simplex.

Approach, from cuOpt `cpp/src/dual_simplex/right_looking_lu.{hpp,cpp}`:
- CSC throughout, **signed** indices (`-1` as the pivot marker → int32 indices)
- Trailing matrix `A − LUᵀ` held in **dual uncompressed** form: values indexed
  by column, row indices indexed by row. Values are only ever needed
  column-wise, and the pivot row is traversed via indices only — so a
  second value array is not worth its synchronisation cost
- Three `start/end/max` triples per row and column (fill-in appends, cancellation
  swap-removes), 2× allocation, garbage-collect above 90% unused
- Markowitz pivoting with a **bucket-indexed degree table** for O(1) min-degree
  lookup
- Relocation with doubling on overflow, with hit/miss instrumentation

- [ ] M10.1 Sparse LU prototype with correctness tests against dense LU
- [ ] M10.2 Integrate behind the existing linear-system abstraction
- [ ] M10.3 Hypersparse path (basis permutation only, no factor)
- [ ] M10.4 Benchmark: rows solvable, and time, versus the dense path
- [ ] M10.5 Remove the dense path once the sparse one is proven

---

## M11 — CPU/GPU device router

**Deferred, deliberately.** No CUDA hardware is available on the build machine
and no bottleneck has been measured. Writing kernels now would be exactly the
"meaningless GPU kernels" the spec forbids.

Current state, stated plainly: **zero CUDA source files**; CMake never defines
`SOVEREIGN_USE_CUDA`; `spmv_csc_auto` is never called by the solver. CPU does
100% of the work. There is no dispatcher.

Before writing any kernel:
1. Profile. Find the actual hot spot.
2. Only then consider the candidates that are genuinely GPU-shaped:
   batched strong-branching LPs, SpMV in a first-order LP inner loop, KKT
   assembly.
3. Implement a `DeviceBackend` abstraction with CPU and CUDA implementations.
4. **Benchmark both, and report the cases where the GPU is slower.** A GPU path
   with no measured win stays off by default.

Sequential pivoting is inherently serial; the honest GPU wins are batched or
first-order, not "make the simplex parallel".

---

## Session protocol

1. Read this file. Work the **current milestone only**.
2. Build, `ctest`, and run the specific tests for what changed.
3. Verify the solver still solves AFIRO to `-464.75314285714285` and
   `transport_50x50` to `504.6`. These are the regression canaries.
4. Update the checkboxes here and the status table in `ROADMAP.md`.
5. Report honestly: what passed, what is still missing, what is blocked.
6. Ask before starting a new milestone.
