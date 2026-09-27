# Sovereign — Roadmap

Living document. Updated as work lands. This file was missing from the original
repository and is item 7 of the founding specification; it now exists and is
maintained.

Status legend: `[x]` done and verified · `[~]` partial or scaffolded ·
`[ ]` not started · `[!]` known-broken or known-honest-gap

---

## 0. Where the project actually is

Verified by audit, not by assumption. Build + full test suite green; 54 tests.

| Claim | Reality |
|---|---|
| LP correct | **Yes.** Netlib AFIRO matches reference to 1e-14. Revised simplex with product-form updates. |
| MILP correct | **Yes but small.** Correct on modest instances; official MIPLIB instances exceed the time limit. |
| QP | **Beta.** Mehrotra IPM with Q in the KKT (1,1) block. |
| Scales to industrial size | **No.** Dense O(m²)/O(m³) basis factorization is a hard wall. |
| GPU accelerated | **No.** No CUDA kernels exist. The SpMV hook honestly declines to run. |
| "Proving optimality" on hard MILP | **No.** Strong branching exists; no dual bound tightening, no restarts, no RINS/RENS. |

Reference point: NVIDIA cuOpt's own README states its MIP solver "excels at
finding high-quality feasible solutions quickly... **Proving feasible solutions
optimal remains under active development**." cuOpt is a mature, GPU-native,
multi-year effort. Matching it is not a realistic near-term goal, and the
specification's own success criteria do not ask for it.

---

## 1. Correctness work (highest priority — in progress)

These are the items that produce *wrong answers*, not slow answers. All landed.

- [x] **IPM termination test was wrong.** The gap test divided the complementarity
      *sum* by `n`, making the test n times too lenient. On `transport_100x100`
      (10 200 columns) the solver reported OPTIMAL at 1009.0399 when the true
      optimum is 1009.0000. Now `1009.00000004`, relative error 4e-11.
      Derivation in the source: for `Ax=b, x>=0`, `c'x - b'y = x'rd + x's`.
- [x] **Simplex claimed OPTIMAL without a certificate.** Now recomputes primal
      residual, worst reduced cost and duality gap on the *final refactorized
      basis against the true RHS*, and downgrades to `NUMERICAL_ERROR` if the
      basis does not certify.
- [x] **Invalid cut generator.** `generate_mir_cuts` skipped columns with a
      negative lower bound instead of abandoning the cut, emitting inequalities
      that can delete integer-feasible points. Cuts are inherited by all
      descendants, so one bad cut poisons a subtree.
- [x] **Independent cut validity gate.** `check_cut_validity()` runs before any
      cut enters a node, written separately from the generators so a generator
      bug degrades to "cut rejected", never to a wrong answer. Rejections are
      reported, never silent.
- [x] **Status honesty.** Added `TIME_LIMIT`, `ITERATION_LIMIT`,
      `NUMERICAL_ERROR`; added `is_conclusive()`. Iteration limit and numerical
      breakdown no longer masquerade as `ERROR` or `OPTIMAL`.
- [x] **Verifier rejects unproven OPTIMAL.** An `OPTIMAL` status whose reported
      duality gap is open is now a verification failure.
- [x] **Benchmark tolerance was hiding the IPM bug.** `obj_close` used
      `rtol=1e-3`, which reported "match" for a 4e-5 relative error. Now
      1e-7 for LP, with per-class tolerances and the observed relative error
      reported alongside the boolean.
- [x] **14 regression tests** covering all of the above; each fails on the
      pre-fix code.

- [ ] **Exhaustive-enumeration validation harness.** For small MILPs, brute-force
      the integer optimum and compare. Not started. This is the single best
      remaining correctness investment and is cheap to build.

## 2. Local-first application architecture (new — NOT STARTED)

Goal: `sovereign.exe` install once, then run entirely offline with a
browser-based interface. No captive portal, no network interception — only the
frictionless-launch *experience*.

Design is complete and written up (`docs/architecture.md`, `docs/auth.md`).
**Implementation has not begun.** The items below are the build order, not
achievements.

- [x] Architecture designed and documented (`docs/architecture.md`, `docs/auth.md`)
- [x] Third-party dependencies vendored and licence-reviewed:
      cpp-httplib (MIT) for transport, Monocypher (CC0) for Ed25519/SHA-256
- [x] **Phase A — CDN dependency removed.** Inter + Outfit self-hosted
      (`web/public/fonts`, latin + latin-ext only, 664 KB, committed).
      Hero video made optional with a static-gradient fallback. Zero remote
      preconnects. Enforced by `benchmarks/tools/check_offline_assets.py`,
      which fails the build on any remote reference in the built output.
      **Verified: 22 files scanned, zero remote asset references.**
- [x] **Phase F groundwork — Ed25519 verification, cross-language proven.**
      `solver/numerical/src/ed25519_verify.cpp` wraps vendored Monocypher
      (CC0). Node signs → Monocypher verifies, proven by committed test vectors
      (`tests/data/`, generated by `tests/scripts/make_test_vectors.mjs` using
      the same code path as the production keygen). 12 tests, including a
      bit-flip sweep over all 64 signature bytes and an appended-byte
      length-extension attempt.
      **Found and fixed a real weakness while testing:** Monocypher's
      `crypto_ed25519_check` accepts a forged all-zero signature when the public
      key is all zeroes — the Ed25519 small-order-point malleability issue that
      libsodium guards against and Monocypher does not. Our wrapper now rejects
      degenerate keys. Residual risk (the other seven low-order points) is
      documented in the source rather than glossed over.
      Verification was sanity-checked by sabotage: with the verifier stubbed to
      `return true`, 7 tests fail. A crypto test that cannot fail is worthless.
- [x] **Phase B groundwork — deps vendored**, licence-reviewed, wired into CMake
      as a separate `monocypher` target.
- [ ] **Phase B — C++ local server** (`sovereign-server`), replacing the
      Python/FastAPI bridge so a release needs no Python. Loopback-only bind.
      Prereq: cpp-httplib vendored (done).
- [ ] **Phase C — `JobManager`**: async solve, `QUEUED → PRESOLVING → SOLVING →
      {OPTIMAL|FEASIBLE|INFEASIBLE|UNBOUNDED|CANCELLED|FAILED}`, SSE progress.
- [ ] **Phase D — `sovereign-launcher`**: free-port selection, single-instance
      detection, automatic browser open.
- [ ] **Phase E — `SecureCredentialStore`** + `WindowsSecureCredentialStore`
      on DPAPI (`CryptProtectData`, per-user scope).
- [ ] **Phase F — Ed25519 offline-credential verification** (device binding,
      expiry, credential version). **Signature verification is done and tested
      above; the credential format, claim validation and device binding are not.**
- [ ] **Phase G — local browser session**: `SameSite=Strict` + `HttpOnly`
      cookie, `GET /api/auth/me`, logout that works fully offline.
- [ ] **Phase H — security hardening**: `Host` validation, `Origin` validation,
      CSRF token, path-traversal rejection, request body caps, no `*` CORS on
      authenticated endpoints.
- [ ] **Phase I — 15 auth/offline tests** plus the offline end-to-end scenario
      (online login → disconnect → launch with no prompt → logout → launch →
      login required).
- [ ] **Phase J — packaging**: Windows x64 zip + Inno Setup script, zero
      runtime dependencies.
- [ ] **Phase K — `react-router` site** with a `/download` page.

- [!] **Device revocation push.** A revoked device will keep working offline
      until its credential expires. Inherent to offline-first; will be
      documented in the threat model, not "fixed".
- [ ] **Credential expiry policy** — needs a product decision (see open questions).

## 3. Performance — the honest gap

These are what stand between "a solver" and "an industrial solver". Ordered by
impact, not by ease.

- [ ] **Sparse basis factorization.** Replace dense `DenseLU` (O(m³) time,
      O(m²) memory) with a product-form / Forrest-Tan sparse LU. **This is the
      single largest blocker to industrial scale.** Everything else is secondary
      until this lands.
- [ ] **Warm-started node LPs.** Every B&B node currently re-solves from a cold
      logical basis. Real solvers inherit the parent basis and re-optimize.
      This is the biggest MILP win available and is a well-understood change.
- [ ] **Stop copying the model per node.** `SearchNode` holds a full deep copy of
      the name-keyed `OptimizationModel`; two children means two more copies.
      Move to bound-change/delta representation, and off name-keyed maps onto
      indexed column storage.
- [ ] **Column-oriented model representation.** `unordered_map<string,double>`
      throughout the model and solution is the wrong structure for a solver and
      blocks every other optimization on this list.
- [ ] **Partial pricing / dual simplex pricing.** Current pricing is a full
      O(n) scan every iteration.
- [ ] **Bound flipping in the ratio test.** Finite upper bounds are currently
      encoded as extra rows, which inflates `m` and is why simplex is slow on
      bounded LPs.
- [ ] **Parallel B&B over the node queue.** Today only the two strong-branching
      child LPs run concurrently (Win32 threads), and it is *slower* than serial
      on the measured case. A real worker pool over the queue is required.
- [ ] **MIP features:** objective cutoff/incumbent constraint, reduced-cost
      fixings, bound tightening, RINS/RENS sub-MIPs, restarts.
- [ ] **Cut library.** Currently cover + a restricted CG. Wanted: MIR, cMIR,
      GMI, flow cover, clique, zero-half. Exact rational arithmetic for Gomory.
- [ ] **MPS parser.** Required by the spec and by the offline product (the
      dashboard's "open local MPS file"). Only a JSON reader exists today.
- [ ] **Crossover** from IPM to a vertex solution, so IPM results are
      directly comparable to simplex and can seed B&B.

## 4. GPU / CUDA (deliberately last, per specification)

The spec says: profile first, do not jump to GPU because it sounds impressive.
Agreed. There is no CUDA device on the current build machine and no measured
bottleneck that GPU would fix.

- [x] `spmv_csc_cpu` reference implementation and a backend-selection hook that
      **honestly declines to run without measured benefit**
- [ ] Profile. Find the actual hot spot. (Candidates: IPM KKT assembly and
      solve, batched strong-branching LPs, SpMV inside the IPM inner loop.)
- [ ] `CUDABackend` behind the existing abstraction
- [ ] Measured CPU-vs-GPU comparison, **including cases where GPU is slower**
- [ ] Multi-GPU PDLP-style distributed LP, if profiling justifies it

Reference points worth reading before writing any kernel: NVIDIA cuOpt
(`cpp/src/pdlp/`, `cpp/src/barrier/csr_kkt_build.cuh`, `cpp/src/dual_simplex/right_looking_lu.hpp`)
and cuPDLPx (`src/solver.cu`, `src/active_set_boost.cu`). cuOpt's `CUOPT_METHOD`
races PDLP / barrier / dual-simplex concurrently and takes the first to converge
— a portfolio by hardware, which is the honest answer to "IPM or first-order on
a GPU". Its MIP optimality proving is also unfinished, which bounds what is
achievable.

## 5. Not started, low priority

- [ ] MIQP, NLP, MINLP (the specification says architecture should allow it; no
      claim of support is made)
- [ ] Python bindings
- [ ] Checkpoint / resume for long MILP jobs
- [ ] Learned branching (specification says experimental only, with a
      deterministic fallback always retained)
- [ ] Marketing site routing, download page polish, CI/CD

---

## Open questions blocking further progress

1. **Auth infrastructure.** No domain, no database, no signing key provisioning.
   See `docs/auth.md` §"Provisioning checklist" for the exact list.
2. **Offline credential expiry policy.** Unlimited is simplest UX and weakest
   revocation. 90-day with silent online refresh is the recommendation.
3. **Registration policy.** Open registration on a public site invites spam.
   Invite-only is the likely requirement.
4. **Windows code signing.** Unsigned releases trigger SmartScreen. OV
   certificate is roughly $200–400/year. Not required for a demo.
5. **Is Python acceptable for the hosted auth service only?** It would
   substantially accelerate the server side, and it is never shipped to users.
