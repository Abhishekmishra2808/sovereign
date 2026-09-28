# Honesty / gaps (current)

Keep this file aligned with measured evidence — not aspirational pitch language.

## What is proven

- From-scratch LP (revised simplex + Mehrotra IPM), MILP (B&C), convex QP (Mehrotra IPM; Frank–Wolfe labeled fallback)
- Netlib AFIRO objective match vs HiGHS (simplex and IPM)
- Official MIPLIB 2017 instances via the same MPS→JSON path as AFIRO: **LP relaxations match HiGHS** on flugpl/gt2/pk1/b-ball/gen-ip016; full MILP not yet competitive (TIMEOUT @30s where HiGHS often finishes in <1s)
- Named robustness cases (Kuhn degeneracy, ill-conditioned) run under **both** simplex and IPM
- Scale through 10k-var transport LPs; **headline path is `auto` (IPM-first)**
- Branch-and-cut with tree cuts + strong/pseudo-cost branching, with ablation table
- Plain B&B and B&C+strong return the same HiGHS-matching optima (102/133/6) on **synthetic** multi-knapsacks

## Algorithm trade-off (engineering decision, not a buried bug)

Exact anti-cycling (lex slack-row perturbation + sticky Bland + refactor under
Bland) makes simplex **correct but slower** on large degenerate LPs.
IPM is fast on the same instances. `SOVEREIGN_LP_ALGORITHM=auto` selects IPM
by default and falls back to simplex — that is the recommended user path.

- Simplex with anti-cycling: correct, but ~tens of seconds at 10k vars on transport
- IPM: ~sub-second on the same 10k-var instance, matching HiGHS
- Auto: prefers IPM (headline scale number)

- Auto transport_100x100.json: status=OPTIMAL time=0.579189s obj=1009.0399319562407
- Auto transport_20x20.json: status=OPTIMAL time=0.022766s obj=202.00000080882702
- Auto transport_50x50.json: status=OPTIMAL time=0.067965s obj=504.6036921484578
- IPM transport_100x100.json: status=OPTIMAL time=0.490385s obj=1009.0399319562407 iters=7
- IPM transport_20x20.json: status=OPTIMAL time=0.026208s obj=202.00000080882702 iters=8
- IPM transport_50x50.json: status=OPTIMAL time=0.075134s obj=504.6036921484578 iters=7
- Simplex transport_100x100.json: status=OPTIMAL time=38.157606s obj=1009.0000000000014 iters=12253
- Simplex transport_20x20.json: status=OPTIMAL time=0.066326s obj=201.99999999999994 iters=566
- Simplex transport_50x50.json: status=OPTIMAL time=1.395043s obj=504.6000000000004 iters=3938

## Robustness: IPM is fast there too (not only on 'nice' LPs)

- illconditioned.json: simplex 0.021136s / IPM 0.015991s (both OPTIMAL/OPTIMAL, objs 29999999.999999996 / 29999999.67302191)
- kuhn_degeneracy.json: simplex 0.042042s / IPM 0.024924s (both OPTIMAL/OPTIMAL, objs -1.2500000000000002 / -1.249999998293319)

## Anti-cycling (revised simplex)

- Lex RHS perturbation on **slack** rows only `delta=(1e-10+1e-12*|b|)*(1+i%1021)` (skip artificial logical-basis rows)
- Sticky Bland per phase (local stack flag — does **not** leak across MILP nodes);
  un-sticks after 64 consecutive improving pivots
- While Bland is active: refactor before pricing (clears product-form etas)

## Multi-core — precise claim

**We parallelize strong-branching child LP evaluation via Win32 threads (2-way).**
This is **not** general parallel branch-and-bound, not parallel simplex pivots,
and not OpenMP across the search tree. Typical concurrency = 2 cores during a
strong-branch decision. Measured times:

- strong_serial: 0.682466s, nodes=49, status=OPTIMAL
- strong_parallel2: 0.724285s, nodes=49, status=OPTIMAL

## Cuts actually implemented

- Cover cuts
- Simple CG/MIR (`generate_mir_cuts`, often labeled gomory)
- Tree application: root multi-round + node rounds by `SOVEREIGN_CUT_FREQUENCY`

## Still open / narrow

- Full tree-level parallel B&B (beyond strong-branch children)
- Iterative (Krylov) KKT solvers for IPM. Both IPMs factor sparse systems directly (LP: `A D Aᵀ`; QP: quasi-definite augmented KKT) with a single-threaded, non-supernodal LDLᵀ and exact minimum-degree ordering; dense LU is used for small or dense systems
- Official MIPLIB **full MILP** competitiveness (converter + LP-relax are solid; B&B search is not yet)
- Synthetic multi-knapsack / set-partition under `datasets/miplib/` are ablation fixtures, **not** official MIPLIB IDs
- Million-variable industrial scale not claimed

## Pitch discipline

- Lead with solver evidence tables, not the LLM agent
- Scale headline = **auto** (IPM-first), with simplex/IPM forced as secondary
- Say “2-way parallel strong-branch LP solves,” not “multi-core MILP solver” generically
- If IPM timeouts or mismatches on a scale/robustness row, that row stays in the table — do not bury it
