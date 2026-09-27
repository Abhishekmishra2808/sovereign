# MIPLIB flugpl: first node-LP reliability fix

The official `flugpl` model exposed a branch-node LP that neither LP method
could classify. With fractional branching, cuts disabled, and a 100-node cap,
the earlier build returned `NUMERICAL_ERROR`: interior point reported a
singular Newton system and simplex hit its 100,000-iteration Phase I limit.
The captured 18-variable, 18-row relaxation is in
`tests/data/flugpl_infeasible_node_lp.json`. HiGHS, used only as an independent
benchmark reference, classified it as infeasible.

Greedy row reduction left the single equality `ANM5 + 0.9 STM5 = STM6`.
At this node, `ANM5 >= 11`, `STM5 >= 67`, and `STM6 = 71`, so the left side
is at least 71.3. The reduced model is in
`tests/data/flugpl_infeasible_node_minimal.json`; HiGHS also classifies it
infeasible. Both the full node and reduced contradiction are regression inputs.

Node LPs now run presolve after branch bounds are applied. The captured node
is classified infeasible without entering either LP algorithm. A unit
regression exercises it through branch-and-bound. The failing-node dump now
uses the normal JSON serializer, preserving numeric precision and objective
terms for future investigations.

After the change, the same 100-node `flugpl` search returned `ITERATION_LIMIT`
with 819 LP iterations, **zero node-LP failure warnings**, and no incumbent.
The full response, settings, input hashes, and timing are in
`flugpl-node-presolve.json`. This is a narrower improvement: it does not prove
the full MIPLIB instance solved or establish benchmark performance.

Validation: Release C++ tests passed; the CPU HTTP worker smoke passed LP,
MILP, QP, MPS, dashboard example and static assets. Infeasibility certificate
verification remains a separate open item in the completion plan. The verifier
now rejects proof-free `INFEASIBLE` and `UNBOUNDED` statuses rather than
marking them valid; those jobs cannot be marked verified by the website until
checkable certificates are implemented. The Python API tests also pass.
The CUDA build also succeeded with `CUDA_PATH` set to the installed Toolkit
`v13.4`. A CUDA transport LP returned `OPTIMAL`, passed primal verification,
and reported 26 actual GPU operations. That canary checks build/runtime parity;
it is not a GPU speedup measurement.
