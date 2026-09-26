# Development Agent Instructions

## Mission

Build a sovereign mathematical optimization platform consisting of:

1. A from-scratch LP/QP/MILP solver.
2. An AI optimization agent powered by Azure OpenAI GPT-5.6.
3. A deterministic verification layer.
4. A benchmark framework comparing the solver against established optimization engines.

The mathematical solver is the primary deliverable.

---

## 1. Non-Negotiable Rule: Solver Must Be From Scratch

The optimization core MUST NOT use an existing optimization solver library.

Do NOT use:

- HiGHS as a backend
- SCIP as a backend
- CBC as a backend
- GLPK as a backend
- Gurobi
- CPLEX
- Xpress
- OR-Tools optimization
- SciPy optimization routines
- Any other library that performs the actual LP/QP/MILP solve

Existing solvers may ONLY be used by the separate benchmark harness.

Never hide an external solver behind an internal interface.

**Bad:**

```python
def solve(model):
    return highs.solve(model)
```

**Good:**

```text
Optimization Model
       |
       v
Our Presolver
       |
       v
Our Sparse Matrix Engine
       |
       v
Our LP/MILP/QP Algorithms
       |
       v
Our Solution
```

---

## 2. LLM Usage

Use:

**Azure OpenAI GPT-5.6**

The LLM is NOT the numerical optimization engine.

The LLM is responsible for:

- Understanding natural language
- Extracting variables
- Extracting constraints
- Constructing optimization models
- Selecting tools
- Calling tools
- Inspecting results
- Iterating
- Explaining results

Never ask the LLM to independently calculate large optimization problems when a deterministic solver tool exists.

---

## 3. Agent Architecture

Implement:

```text
User
 |
 v
GPT-5.6
 |
 +--> formulate_model
 |
 +--> select_solver
 |
 +--> call_solver
 |
 +--> verify_solution
 |
 +--> inspect_result
 |
 +--> modify_model
 |
 +--> call_solver again
 |
 v
Final Answer
```

The agent must be capable of multiple solver iterations.

---

## 4. Tool Contracts

Every tool must have a strict schema.

Initial tools:

- `math_calculator`
- `lp_solver`
- `qp_solver`
- `milp_solver`
- `verify_solution`

Tool inputs must use structured JSON.

Do not pass arbitrary natural-language prompts to the solver.

Example:

```json
{
  "problem_type": "LP",
  "sense": "minimize",
  "variables": [],
  "objective": {},
  "constraints": []
}
```

---

## 5. Solver Development Order

Implement in this order:

| Stage | Work |
|-------|------|
| 1 | Sparse vectors and matrices |
| 2 | Numerical utilities |
| 3 | Revised simplex LP solver |
| 4 | Presolve |
| 5 | LP verification and regression suite |
| 6 | Branch and Bound |
| 7 | MILP support |
| 8 | Cutting planes |
| 9 | MILP heuristics |
| 10 | QP |
| 11 | Multi-core execution |
| 12 | GPU experiments |

Never skip correctness testing to move faster.

---

## 6. Numerical Engineering Rules

Every numerical algorithm must explicitly consider:

- Floating-point error
- Feasibility tolerances
- Optimality tolerances
- Scaling
- Degeneracy
- Ill-conditioned matrices
- Near-zero values
- Overflow/underflow
- Residual errors

Do not use naive equality checks such as:

```cpp
if (x == 0)
```

Use appropriate numerical tolerances.

---

## 7. Sparse Matrix Rules

Industrial problems can contain millions of variables and constraints.

Do not default to dense matrices.

Support sparse representations.

Initially evaluate:

- CSR
- CSC

Choose representation based on algorithmic requirements.

Avoid unnecessary matrix copies.

Profile memory usage.

---

## 8. LP Solver

Initial LP support:

```text
min/max cᵀx

subject to

Ax <= b
Ax >= b
Ax = b

lower <= x <= upper
```

Implement:

- Revised simplex
- Basis representation
- Reduced costs
- Primal feasibility
- Dual feasibility
- Pivot selection
- Phase I
- Phase II
- Infeasibility detection
- Unboundedness detection

Build a strong LP test suite before implementing MILP.

---

## 9. MILP Solver

Support:

- Continuous variables
- Integer variables
- Binary variables

Initial implementation:

```text
LP relaxation
      ↓
Branch
      ↓
Node LP
      ↓
Bound
      ↓
Prune
      ↓
Incumbent
```

Then implement:

- Best-bound node selection
- Depth-first fallback
- Variable branching
- Integrality checks
- Bound propagation
- Incumbent management

Later:

- Strong branching
- Pseudocost branching
- Gomory cuts
- Cover cuts
- Heuristics

---

## 10. QP Solver

Support a useful initial subset of convex QP.

Example:

```text
minimize

1/2 xᵀQx + cᵀx

subject to

Ax <= b
```

Ensure positive-semidefinite handling is correct.

Do not claim general non-convex QP support unless implemented and tested.

---

## 11. Presolve

Implement independent presolve passes.

Every transformation must be reversible.

The solver must be able to map the reduced solution back to the original variables.

Each presolve pass requires tests.

---

## 12. Verification

Every solver result must pass a verifier.

The verifier must independently calculate:

- Constraint violations
- Bound violations
- Integrality violations
- Objective value
- Residuals

The LLM must never be the authority for whether a numerical solution is valid.

---

## 13. Agent Safety

If the LLM produces an invalid or incomplete model:

Do not silently solve it.

Instead:

1. Validate the model
2. Identify the issue
3. Ask the LLM to repair it
4. Revalidate
5. Solve

For ambiguous user requirements, ask for clarification instead of inventing constraints.

---

## 14. Agent Iteration

The agent should be able to perform:

```text
formulate
→ validate
→ solve
→ inspect
→ modify
→ solve again
→ verify
→ explain
```

Set a maximum iteration/tool-call limit.

Do not create infinite solver loops.

Every iteration must record:

- Model version
- Tool call
- Solver result
- Reason for modification

---

## 15. Benchmark Harness

The benchmark harness is separate from the production solver.

It may execute:

- OurSolver
- HiGHS
- SCIP
- CBC
- Gurobi

The production code must never import or depend on these solvers.

Benchmark metrics:

- Runtime
- Objective
- Status
- Optimality gap
- Iterations
- Nodes
- Memory
- Numerical failures

Use identical:

- Input
- Hardware
- Timeout
- Precision assumptions where applicable

---

## 16. Regression Tests

Every bug becomes a regression test.

Required test categories:

- Basic LP
- Infeasible LP
- Unbounded LP
- Degenerate LP
- Ill-conditioned LP
- Large sparse LP
- Basic MILP
- Binary MILP
- Integer MILP
- Weak relaxation MILP
- Large MILP
- Convex QP
- Presolve tests
- Numerical stability tests

---

## 17. Performance

Do not optimize based on assumptions.

Profile first.

Priority:

```text
Correctness
    ↓
Numerical stability
    ↓
Memory efficiency
    ↓
Algorithmic efficiency
    ↓
CPU parallelism
    ↓
GPU acceleration
```

GPU work must be justified by benchmark results.

---

## 18. GPU

Do not implement "GPU acceleration" merely as a marketing feature.

Profile the solver and identify expensive numerical kernels.

Candidates:

- SpMV
- Vector operations
- Reductions
- Batched operations
- Suitable sparse numerical kernels

Compare:

```text
CPU
vs
GPU
vs
CPU + GPU
```

including data-transfer overhead.

Only retain GPU execution paths where they provide measurable benefits.

---

## 19. Code Quality

Prefer:

- Small modules
- Clear interfaces
- Deterministic behavior
- Strong typing
- Unit tests
- Documentation
- Reproducible benchmarks

Avoid:

- Giant classes
- Hidden global state
- Magic numerical constants
- Unverified optimizations
- Solver dependencies hidden behind wrappers

---

## 20. Suggested Repository Structure

```text
/
├── solver/
│   ├── core/
│   ├── sparse/
│   ├── numerical/
│   ├── lp/
│   ├── qp/
│   ├── milp/
│   ├── presolve/
│   ├── cuts/
│   ├── heuristics/
│   ├── parallel/
│   └── gpu/
│
├── agent/
│   ├── prompts/
│   ├── tools/
│   ├── schemas/
│   ├── orchestration/
│   └── verification/
│
├── api/
│
├── benchmarks/
│   ├── datasets/
│   ├── runners/
│   ├── external/
│   └── reports/
│
├── tests/
│   ├── unit/
│   ├── numerical/
│   ├── lp/
│   ├── qp/
│   └── milp/
│
├── docs/
│
├── CMakeLists.txt
├── README.md
├── solution.md
└── agent.md
```

---

## 21. Development Behavior

Before implementing a major component:

1. Understand the mathematical algorithm
2. Define its input/output contract
3. Write tests
4. Implement the simplest correct version
5. Validate against known solutions
6. Profile it
7. Optimize only after correctness is established

Do not copy implementation code from existing optimization solvers.

Use mathematical literature, papers, textbooks, and public documentation to understand algorithms, but implement the algorithms independently.

---

## 22. Definition of Done

The project is not considered complete because an LLM can call a solver.

The minimum successful system must demonstrate:

- From-scratch LP solver
- From-scratch MILP capability
- QP capability
- Sparse numerical computation
- Presolve
- Numerical verification
- Benchmark harness
- Comparison against established solvers
- At least one realistic industrial optimization model
- Azure GPT-5.6 agent that can formulate and iteratively solve problems through tools

The final demonstration should show:

```text
Natural-language industrial problem
                ↓
          GPT-5.6 Agent
                ↓
       Mathematical model
                ↓
        From-scratch solver
                ↓
       Verified solution
                ↓
        GPT-5.6 explanation
```

The project should clearly demonstrate that:

- The LLM provides intelligence and usability
- The sovereign solver provides mathematical correctness and optimization
- The verifier provides trust
