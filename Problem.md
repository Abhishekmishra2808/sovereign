# Problem Statement: Sovereign Mathematical Optimization Solver

## Background

Almost every optimization problem in India's refining, petrochemical, power, logistics, manufacturing and planning sectors ultimately depends on a handful of foreign mathematical optimization solvers such as IBM ILOG CPLEX, Gurobi and FICO Xpress. These engines sit behind refinery scheduling, production planning, supply chain optimization, blending, energy management and many AI-driven decision-support systems.

While they are extremely capable, they come with high recurring license costs, restrictive licensing models and limited visibility into the underlying optimization algorithms. Indian developers can formulate optimization problems, but they cannot inspect, modify or tailor the solver internals to suit strategic national requirements.

Open-source alternatives such as COIN-OR CBC, HiGHS, GLPK and SCIP exist and have made significant progress, but they still lag behind commercial solvers for several classes of large-scale mixed-integer optimization problems and have not been developed, validated or optimized specifically for Indian industrial use cases.

The real challenge is not building the modeling interface; it is developing a numerically robust optimization engine that consistently finds high-quality solutions for large, sparse and highly constrained industrial problems within practical computation times.

## Description

The objective is to develop a **sovereign mathematical optimization solver core** rather than a complete modeling environment. The solver should support:

| Phase | Problem classes |
|-------|-----------------|
| Initial focus | Linear Programming (LP), Mixed-Integer Linear Programming (MILP), Quadratic Programming (QP) |
| Later extension | Mixed-Integer Quadratic Programming (MIQP), Nonlinear Programming (NLP), Mixed-Integer Nonlinear Programming (MINLP) |

### Core algorithms

- **Continuous optimization:** revised simplex and interior-point methods
- **Mixed-integer:** branch-and-bound, branch-and-cut, cutting planes, presolve, heuristics and advanced node selection strategies

### Technical requirements

- Sparse matrix techniques
- Efficient numerical linear algebra
- Multi-core parallelization
- GPU acceleration where it provides measurable benefits
- Emphasis on numerical stability, scalability and reliable convergence across large industrial optimization problems
- **Not** focused on graphical interfaces or modelling tools
- **Must not** be built upon any existing open-source solver library; shall be built from scratch from mathematical foundations

### Application scope

Optimization problems arising from:

- Refinery scheduling
- Crude blending
- Process optimization
- Production planning
- Logistics
- Power system dispatch
- Transportation
- Supply chain management

### Performance benchmark

The solver should consistently deliver optimal or near-optimal solutions for industrial-scale problems involving **thousands to millions of variables and constraints**, including:

- Highly degenerate models
- Ill-conditioned matrices
- Difficult mixed-integer formulations

…where weaker implementations exhibit excessive computation times or fail to converge.

## Expected Solution

A robust optimization engine with a basic application programming interface (API) or command-line interface is sufficient; a polished graphical user interface is not required.

The solver should:

1. Successfully solve standard benchmark problems from recognised optimization libraries such as **MIPLIB**, **Netlib** or **Mittelmann** benchmark sets
2. Compare solution quality and computational performance against at least one established commercial or open-source solver
3. Demonstrate numerical robustness by solving challenging large-scale optimization problems involving degeneracy, weak LP relaxations or ill-conditioned constraint matrices, where simpler implementations struggle to achieve reliable convergence or acceptable solution times

The resulting solver should provide a transparent, extensible and sovereign foundation for future Indian optimization software across industrial, scientific and strategic applications.
