# Sovereign AI Mathematical Optimization Platform

## 1. Overview

Build a sovereign mathematical optimization platform for Indian industrial use cases such as:

- Refinery scheduling
- Crude blending
- Production planning
- Power dispatch
- Logistics
- Transportation
- Supply-chain optimization

The system combines:

1. A from-scratch mathematical optimization engine.
2. An AI optimization agent powered by Azure OpenAI GPT-5.6.
3. Deterministic verification and benchmarking infrastructure.

The LLM converts natural-language requirements into mathematical optimization models, selects the appropriate solver tools, analyzes results, and iterates when necessary.

The mathematical solver itself must be implemented from scratch and must not depend on an existing optimization solver such as HiGHS, SCIP, CBC, GLPK, Gurobi, CPLEX, OR-Tools, SciPy optimization, or similar libraries for solving the optimization problem.

---

## 2. Core Architecture

```text
                        USER
                          |
                          v
                 +----------------+
                 |  GPT-5.6 Agent |
                 |  Azure OpenAI  |
                 +-------+--------+
                         |
                 Natural language
                         |
                         v
              Mathematical formulation
                         |
                         v
              +----------------------+
              |   Tool Layer / API   |
              +----------+-----------+
                         |
          +--------------+---------------+
          |              |               |
          v              v               v
     Math Tool       LP Solver       MILP Solver
                         |               |
                         v               v
                    QP Solver      Optimization Core
          |              |               |
          +--------------+---------------+
                         |
                         v
                  Solution Verifier
                         |
                         v
                  Result + Metadata
                         |
                         v
                    GPT-5.6 Agent
                         |
                         v
                    Final Answer
```

---

## 3. LLM Agent

The agent uses:

**Azure OpenAI GPT-5.6**

The LLM is responsible for:

- Understanding natural-language optimization requests.
- Identifying decision variables.
- Identifying objectives.
- Extracting constraints.
- Determining variable bounds.
- Determining whether the problem is LP, QP, or MILP.
- Constructing a structured optimization model.
- Calling solver tools.
- Inspecting solver results.
- Detecting infeasibility or missing constraints.
- Iterating on the mathematical formulation.
- Explaining the final result.

The LLM must **NOT** perform numerical optimization itself when a solver tool can perform it.

For example, the LLM should not manually calculate a large matrix inverse or claim an optimal solution.

Instead:

```text
LLM
 |
 | formulate problem
 v
Solver Tool
 |
 | deterministic computation
 v
Verified Result
 |
 v
LLM
 |
 | explain result
 v
User
```

---

## 4. Tool System

The LLM interacts with deterministic tools.

Initial tools:

### `math_calculator`

Used for:

- Arithmetic
- Numeric expressions
- Unit conversions
- Small mathematical calculations

### `lp_solver`

Solves Linear Programming problems.

### `qp_solver`

Solves Quadratic Programming problems.

### `milp_solver`

Solves Mixed-Integer Linear Programming problems.

### `verify_solution`

Checks:

- Constraint feasibility
- Variable bounds
- Integrality
- Objective value
- Numerical tolerances
- Solver status
- Optimality gap where available

---

## 5. Structured Optimization Model

The LLM should never send free-form text directly to the solver.

Use a structured representation.

Example:

```json
{
  "problem_type": "MILP",
  "sense": "maximize",
  "variables": [
    {
      "name": "crude_a",
      "type": "continuous",
      "lower_bound": 0,
      "upper_bound": 50000
    },
    {
      "name": "unit_enabled",
      "type": "binary",
      "lower_bound": 0,
      "upper_bound": 1
    }
  ],
  "objective": {
    "linear": {
      "crude_a": 5000,
      "unit_enabled": 10000
    }
  },
  "constraints": [
    {
      "name": "diesel_minimum",
      "linear": {
        "crude_a": 0.7
      },
      "sense": ">=",
      "rhs": 30000
    }
  ]
}
```

The solver receives this representation.

---

## 6. Solver Core

The solver core is the most important component.

It must be developed from mathematical foundations.

Do not use another optimization solver internally.

Initial target:

```text
LP
 |
 +-- Revised Simplex
 |
 +-- Interior Point

QP
 |
 +-- Convex QP support

MILP
 |
 +-- LP Relaxation
 +-- Branch and Bound
 +-- Branch and Cut
 +-- Cutting Planes
 +-- Presolve
 +-- Primal Heuristics
 +-- Node Selection
```

---

## 7. Numerical Engine

Implement a custom sparse numerical layer.

Required concepts:

- Sparse matrix representation
- Sparse matrix-vector multiplication
- Vector operations
- Scaling
- Numerical tolerances
- Sparse factorization
- Basis management
- Feasibility checks
- Residual calculations

The engine must prioritize numerical stability over premature optimization.

---

## 8. Presolve

Implement presolve before advanced MILP functionality.

Initial techniques:

- Remove fixed variables
- Remove redundant constraints
- Bound tightening
- Singleton elimination
- Detect infeasibility
- Detect unconstrained variables
- Coefficient strengthening
- Variable substitution where safe
- Row/column simplification
- Scaling

Pipeline:

```text
Original Model
      |
      v
   Presolve
      |
      v
Reduced Model
      |
      v
    Solver
      |
      v
Solution Recovery
```

---

## 9. MILP Engine

The MILP solver should initially implement:

- LP relaxation
- Branch and Bound
- Variable branching
- Node queue
- Bound propagation
- Incumbent tracking
- Pruning
- Optimality gap

Then extend with:

- Gomory cuts
- Cover cuts
- Strong branching
- Pseudocost branching
- Best-bound node selection
- Feasibility heuristics
- Parallel branch and bound

---

## 10. GPU Acceleration

GPU acceleration is optional and must be evidence-driven.

Do not move the complete MILP algorithm to GPU.

The CPU should control:

- Branch and bound
- Node selection
- Solver orchestration
- Heuristics
- Irregular control flow

GPU candidates include:

- Sparse matrix-vector multiplication
- Vector operations
- Reductions
- Batched numerical operations
- Suitable linear algebra kernels

GPU acceleration must only be retained when benchmarks demonstrate measurable benefits.

---

## 11. Parallelism

The solver should support multi-core execution.

Potential parallel areas:

- Independent branch-and-bound nodes
- Multiple LP subproblems
- Sparse numerical kernels
- Presolve operations where safe
- Parallel reductions

Parallelism must not compromise deterministic correctness.

---

## 12. Agent Iteration

The agent should support iterative optimization.

Example:

```text
User Request
     |
     v
LLM formulation
     |
     v
Solver
     |
     v
Result
     |
     v
LLM analysis
     |
     +----> Missing constraint?
     |          |
     |          v
     |      Modify model
     |          |
     |          +----> Solver
     |
     +----> Infeasible?
     |          |
     |          v
     |      Diagnose / ask user
     |
     +----> Valid
                |
                v
             Answer
```

The agent must retain the complete model and solver results throughout an optimization session.

---

## 13. Industrial Use Cases

The first demonstrations should include:

### Refinery scheduling

Optimize:

- Crude selection
- Refinery unit utilization
- Product yields
- Product demand
- Operating cost
- Profit

### Crude blending

Optimize blend composition subject to:

- Quality constraints
- Availability
- Product specifications
- Cost

### Power dispatch

Optimize:

- Generator output
- Fuel cost
- Demand
- Capacity
- Operational constraints

### Logistics

Optimize:

- Plant-to-warehouse allocation
- Warehouse-to-customer allocation
- Transportation cost
- Capacity constraints

---

## 14. Benchmarking

Create an automated benchmark framework.

Benchmark against:

- HiGHS
- SCIP
- CBC where appropriate
- Gurobi where a valid license/environment is available

Use public benchmark families:

- MIPLIB
- Netlib
- Mittelmann

Do **NOT** use these solvers inside the product.

They are external benchmark references only.

Record:

- Problem
- Variables
- Constraints
- Nonzeros
- Solver
- Status
- Objective
- Runtime
- Memory
- Iterations
- B&B nodes
- Optimality gap
- Numerical warnings

---

## 15. Numerical Robustness Tests

Create dedicated difficult cases:

- Degenerate LPs
- Ill-conditioned matrices
- Weak LP relaxations
- Large sparse problems
- Nearly infeasible models
- Highly constrained MILPs
- Large coefficient ranges

Correctness must be checked independently from the LLM.

---

## 16. LLM Benchmark

Benchmark the GPT-5.6 agent separately.

Measure:

- Formulation accuracy
- Variable extraction accuracy
- Constraint extraction accuracy
- Objective extraction accuracy
- Tool-selection accuracy
- Number of solver calls
- Number of iterations
- Final solution correctness
- Hallucinated constraints
- Missing constraints

This allows us to distinguish:

```text
LLM performance
        vs
Solver performance
```

---

## 17. Security

Industrial optimization data may contain sensitive information.

The system should support:

- Private deployment
- Authentication
- Authorization
- Audit logs
- Encryption in transit
- Encryption at rest
- Configurable data retention
- No unnecessary transmission of optimization data

Azure OpenAI configuration must be isolated behind the agent service.

---

## 18. Technology Direction

Recommended initial stack:

### Core

- C++
- CMake
- Custom sparse numerical structures
- OpenMP for parallelism
- CUDA as optional accelerator

### Agent

- Python
- Azure OpenAI GPT-5.6
- Structured tool calling
- Pydantic schemas

### API

- FastAPI or equivalent service
- JSON optimization model format

### Testing

- GoogleTest
- Python benchmark harness

### Storage

- PostgreSQL if persistent jobs/models/results are required

---

## 19. Development Priorities

Priority order:

1. Mathematical correctness
2. Numerical stability
3. LP solver
4. Sparse numerical engine
5. Presolve
6. MILP branch and bound
7. QP
8. Cutting planes and heuristics
9. Multi-core execution
10. Industrial benchmarks
11. LLM agent
12. GPU acceleration

Do not optimize prematurely.

---

## 20. Core Principle

The system must maintain a strict separation:

```text
LLM = reasoning and orchestration

Solver = deterministic mathematical optimization

Verifier = correctness
```

This ensures that an LLM hallucination cannot silently become an industrial optimization result.
