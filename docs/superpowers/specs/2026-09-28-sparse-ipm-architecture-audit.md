# Sparse IPM Architecture Audit Report
**Date:** 2026-09-28  
**Objective:** Upgrade LP Interior Point Method from dense to sparse linear algebra

## Executive Summary

The current IPM implementation constructs **dense m×m normal equations** M = A D A^T and solves with DenseLU, consuming O(m²) memory and O(m³) time. This blocks industrial-scale problems (100k+ constraints).

**Critical Finding:** SparseLU exists for simplex bases but is **NOT appropriate** for symmetric positive-definite normal equations. A dedicated sparse Cholesky/LDL^T factorization is required.

---

## 1. Current IPM Execution Path

### File: `solver/lp/src/interior_point.cpp`

**Entry point:** `InteriorPointSolver::solve(model)` → `solve_ipm(IpmLp, options)`

**Key bottleneck function (lines 54-81):**
```cpp
void build_normal_eq(const SparseMatrixCSC& A, const std::vector<double>& d,
                     std::vector<double>& M_col_major) {
  const int m = static_cast<int>(A.nrows);
  const int n = static_cast<int>(A.ncols);
  M_col_major.assign(m * m, 0.0);  // ← DENSE m×m allocation!
  
  for (int j = 0; j < n; ++j) {
    const double dj = d[j];
    if (dj == 0.0) continue;
    for (int p = A.col_ptr[j]; p < A.col_ptr[j+1]; ++p) {
      const int r = A.row_idx[p];
      const double ar = A.values[p] * dj;
      for (int q = A.col_ptr[j]; q < A.col_ptr[j+1]; ++q) {
        const int c = A.row_idx[q];
        const double ac = A.values[q];
        M_col_major[c * m + r] += ar * ac;  // ← Fills entire m×m
      }
    }
  }
  // Regularization
  for (int i = 0; i < m; ++i) {
    M_col_major[i * m + i] += 1e-12;
  }
}
```

**Newton solve (lines 264-306):**
```cpp
bool solve_newton(const IpmLp& lp, ...) {
  // Compute D = X S^{-1}
  for (int j = 0; j < n; ++j) {
    d[j] = x[j] / s[j];
  }
  
  // Build M = A D A^T (DENSE)
  std::vector<double> M;
  build_normal_eq(lp.A, d, M);
  
  // Factor and solve with DenseLU
  DenseLU lu;
  if (!lu.factorize(std::move(M), m)) return false;
  dy = rhs;
  if (!lu.solve(dy)) return false;
  
  // Recover dx, ds
  ...
}
```

**IPM iteration loop (lines 365-483):**
- **Affine predictor:** calls `solve_newton()` with rxs = -XS*e
- **Corrector:** calls `solve_newton()` with rxs = -XS*e - dx_aff.*ds_aff + σμe
- **Both** use the same `build_normal_eq()` + `DenseLU` path

**Startup solve (lines 339-359):**
- Also uses `build_normal_eq()` + `DenseLU` for Mehrotra starting point

**Total Newton solves per IPM iteration:** 3 (startup + predictor + corrector)

---

## 2. Matrix Storage: SparseMatrixCSC

### File: `solver/sparse/include/sovereign/sparse_matrix.hpp`

```cpp
struct SparseMatrixCSC {
  std::size_t nrows, ncols;
  std::vector<int> col_ptr;    // size ncols + 1
  std::vector<int> row_idx;    // size nnz
  std::vector<double> values;  // size nnz
  
  void multiply(x, y);           // y = A*x
  void multiply_transpose(x, y); // y = A^T*x
};
```

**Storage:** Standard CSC format  
**Operations:** Efficient sparse matrix-vector products  
**Used by:** IPM, simplex, all solvers

---

## 3. Existing SparseLU

### File: `solver/numerical/include/sovereign/sparse_lu.hpp`

```cpp
class SparseLU {
 public:
  bool factorize(n, col_ptr, row_idx, values);
  bool solve(x);                    // A x = b
  bool solve_transpose(x);          // A^T x = b
  
  double pivot_threshold = 0.1;     // Threshold partial pivoting
  
 private:
  std::vector<int> l_ptr_, l_idx_;
  std::vector<double> l_val_;       // Unit lower triangular
  std::vector<int> u_ptr_, u_idx_;
  std::vector<double> u_val_;       // Upper triangular
  std::vector<int> pinv_;           // Row permutation
  std::vector<int> q_;              // Column permutation
};
```

**Algorithm:** Left-looking sparse LU with threshold partial pivoting  
**Pivoting:** Columns processed sparsest-first, pivot selection by:
1. Magnitude within threshold of largest (numerical stability)
2. Fewest remaining row entries (sparsity)

**Factorization:** P A Q = L U (unsymmetric)

**Used by:**
- Revised primal simplex (`solver/lp/src/revised_simplex.cpp:301`)
- Dual simplex (`solver/lp/src/dual_simplex.cpp`)

**Singular pivot threshold:** 1e-14 (matches DenseLU)

---

## 4. Can SparseLU Be Reused for Normal Equations?

### ❌ NO - NOT APPROPRIATE

**Reasons:**

1. **Symmetry not exploited:**
   - Normal equations M = A D A^T are symmetric positive-definite (SPD)
   - SparseLU computes full P A Q = L U (unsymmetric factorization)
   - Wastes 50% of storage and computation

2. **Pivoting strategy wrong:**
   - SparseLU uses **threshold partial pivoting** for general unsymmetric matrices
   - SPD matrices need only **diagonal pivoting** or **no pivoting** (Cholesky)
   - Threshold pivoting can introduce unnecessary fill-in

3. **Column ordering wrong:**
   - SparseLU orders columns sparsest-first (good for simplex bases with slack columns)
   - SPD factorization needs **fill-reducing ordering** (AMD, COLAMD, nested dissection)
   - Different optimization objective

4. **Storage format wrong:**
   - SparseLU stores L and U separately
   - Cholesky stores only L (or LDL^T stores L and diagonal D)
   - 2x storage overhead

**Conclusion:** Need dedicated symmetric positive-definite factorization.

---

## 5. Where A D A^T is Constructed

**Single location:** `solver/lp/src/interior_point.cpp:54-81` (`build_normal_eq`)

**Called from:**
1. **Line 288:** Affine predictor Newton solve
2. **Line 342:** Mehrotra startup
3. Corrector implicitly uses same function via `solve_newton()`

**Observation:** A is constant throughout IPM. Only diagonal D = X S^{-1} changes each iteration.

---

## 6. Sparsity Pattern Reusability

**Key insight:** For a fixed constraint matrix A:

```
Sparsity pattern of M = A D A^T is FIXED regardless of diagonal D values
```

**Proof:** Entry M[i,j] is nonzero iff ∃ column k where both A[i,k] and A[j,k] are nonzero.

**Implication:**
- **Symbolic factorization** (analyze sparsity, compute ordering) can be done **ONCE** before IPM loop
- **Numerical factorization** (compute L values) done **EVERY iteration** with new D
- Massive savings for 30-50 IPM iterations

**Exception:** A changes if:
- Problem is modified between solves
- Different LP loaded
- NOT an issue within single IPM solve

---

## 7. Numerical Safeguards in Current IPM

### Regularization (line 79):
```cpp
M_col_major[i * m + i] += 1e-12;
```
- Adds small diagonal perturbation
- Prevents singular systems
- **Hard-coded value** - not configurable

### Factorization failure handling (lines 289-293, 457-465):
```cpp
if (!lu.factorize(...)) return false;
```
- Returns `SolverStatus::NumericalError`
- Message: "Newton normal-equations factorization is singular"
- Caller (lp_solver.cpp) falls back to simplex

### Iteration limit (line 365):
```cpp
for (int it = 0; it < opt.max_iterations; ++it)
```
- Default: 100 iterations (InteriorPointOptions)
- No incremental regularization on failure

### Step-length safeguards (lines 467-470):
```cpp
alpha_p = tau * step_to_bound(x, dx);  // tau = 0.9995 (fraction-to-boundary)
alpha_p = std::min(1.0, alpha_p);
x[j] = std::max(x[j], 1e-14);  // Prevent exact zero
```

**Missing:**
- Adaptive regularization
- Iterative refinement
- Condition number estimation
- Pivot magnitude monitoring

---

## 8. Existing Tests

### LP Tests (`tests/lp/test_lp_solver.cpp`):
- 80+ lines of LP test cases
- Tests call `OptimizationEngine().solve(model)` 
- Engine auto-selects IPM or simplex based on `SOVEREIGN_LP_ALGORITHM`
- Default: IPM first, simplex fallback
- Tests verify:
  - Optimal objective value
  - Primal solution
  - Status codes
  - Verification (residuals, optimality)

### Sparse LU Tests (`tests/unit/test_sparse_lu.cpp`):
- 160 lines
- Tests SparseLU correctness on small matrices
- Compares sparse vs dense solutions

### QP Tests (`tests/qp/test_qp_solver.cpp`):
- QP also uses IPM (with Q in KKT block)
- Not directly relevant to LP-IPM sparse upgrade

### Correctness Regressions (`tests/unit/test_correctness_regressions.cpp`):
- 14 regression tests for past bugs
- Includes IPM duality gap fix (transport_100x100)

**Missing:**
- No explicit dense-vs-sparse IPM comparison tests
- No large sparse IPM benchmarks (current tests are <100 vars)
- No symmetric factorization tests

---

## 9. DenseLU Implementation

### File: `solver/numerical/include/sovereign/dense_lu.hpp`

```cpp
class DenseLU {
 public:
  bool factorize(std::vector<double> a_col_major, std::size_t n);
  bool solve(std::vector<double>& x);
  bool solve_transpose(std::vector<double>& x);
  
 private:
  std::size_t n_;
  bool ok_;
  std::vector<double> lu_;  // Stores L and U in-place
  std::vector<int> piv_;    // Row permutation
};
```

**Algorithm:** Dense LU with partial pivoting (LAPACK-style)  
**Memory:** O(n²) for n×n matrix  
**Cost:** O(n³) factorization, O(n²) solve

**Used by:**
- IPM normal equations (only place)
- NOT used by simplex (simplex uses SparseLU)

**Must remain:** Required as reference implementation and fallback.

---

## 10. CMake Configuration

### File: `solver/CMakeLists.txt`

```cmake
add_library(sovereign_core STATIC
  # ... existing files ...
  numerical/src/dense_lu.cpp
  numerical/src/sparse_lu.cpp
  lp/src/interior_point.cpp
  # ...
)
```

**No external dependencies:**
- No BLAS/LAPACK linked
- No SuiteSparse
- All linear algebra is from-scratch
- Must remain self-contained

---

## 11. Benchmark Infrastructure

### Existing benchmarks:
- `benchmarks/runners/run_benchmarks.py` - Netlib, MIPLIB suite
- `benchmarks/runners/run_evidence_pack.py` - Quick smoke test
- `benchmarks/reports/EVIDENCE.md` - Latest results

**Current largest LP benchmarked:**
- transport_50x50: 2500 vars, 100 constraints, 0.17s
- transport_100x100: mentioned in comments (10k vars) but not in regular suite

**Missing:**
- No memory profiling
- No separate timing of assembly vs factorization vs solve
- No dense-vs-sparse comparison infrastructure

---

## 12. Memory Consumption Analysis

**Example: transport_50x50 (m=100, n=2500)**

### Current dense approach:
```
A (CSC): 100 × 2500, ~5000 nnz  
  Storage: 5000 × (8 + 4 + 4) bytes = 80 KB

M (dense): 100 × 100 = 10,000 doubles  
  Storage: 10,000 × 8 bytes = 80 KB

Dense LU: Same m×m storage = 80 KB

Total: ~240 KB (manageable)
```

### Projected for transport_100x100 (m=200, n=10000):
```
A (CSC): 200 × 10000, ~20,000 nnz  
  Storage: 20,000 × 16 bytes = 320 KB

M (dense): 200 × 200 = 40,000 doubles  
  Storage: 40,000 × 8 bytes = 320 KB

Dense LU: 320 KB

Total: ~960 KB (still OK)
```

### Projected for industrial scale (m=10,000, n=100,000):
```
A (CSC): 10k × 100k, ~500k nnz (sparse)  
  Storage: 500k × 16 bytes = 8 MB

M (dense): 10k × 10k = 100M doubles  
  Storage: 100M × 8 bytes = 800 MB

Dense LU: 800 MB (in-place, but needs copy)

Total: ~1.6 GB per Newton solve
  ×3 per IPM iteration = 4.8 GB
  ×50 iterations = 240 GB cumulative allocations
  
IMPOSSIBLE!
```

### With sparse Cholesky (estimated):
```
M (sparse structure): ~5M nonzeros (5% fill-in)  
  Storage: 5M × 16 bytes = 80 MB

L (sparse): ~10M nonzeros (2× from fill)  
  Storage: 10M × 16 bytes = 160 MB

Total: ~250 MB (6× reduction)
```

**Conclusion:** Sparse approach is essential for m > 1000.

---

## 13. Key Design Decisions Required

### 1. Symmetric factorization algorithm:

| Option | Pros | Cons |
|--------|------|------|
| **Cholesky (LL^T)** | Proven stable for SPD, standard algorithm | Requires positive-definite M (regularization critical) |
| **LDL^T** | More numerically robust, handles near-singular | Slightly more complex, requires diagonal pivoting |
| **Modified Cholesky** | Guarantees positive-definite result | Alters problem slightly |

**Recommendation:** Start with **LDL^T** for robustness, fall back to modified Cholesky if needed.

### 2. Fill-reducing ordering:

| Option | Pros | Cons |
|--------|------|------|
| **AMD (Approximate Minimum Degree)** | Fast, good fill, standard | Greedy (not globally optimal) |
| **COLAMD** | Better for rectangular A | More complex |
| **Nested Dissection** | Best fill for large problems | Expensive to compute |
| **None (natural order)** | Simple | Catastrophic fill-in |

**Recommendation:** Start with **AMD**, make ordering pluggable.

### 3. Sparse normal equation assembly:

| Option | Pros | Cons |
|--------|------|------|
| **Hash table accumulation** | Simple, handles arbitrary sparsity | Slower, more memory |
| **Column-oriented outer product** | Matches A's CSC format | Complex indexing |
| **Symbolic-guided assembly** | Fastest, minimal overhead | Requires symbolic phase first |

**Recommendation:** **Symbolic-guided assembly** (compute pattern once, reuse).

### 4. Backend selection:

| Option | Pros | Cons |
|--------|------|------|
| **Always sparse** | Simplest code path | Overkill for tiny problems |
| **Always dense** | Current behavior | Doesn't solve scale issue |
| **Automatic threshold** | Best of both worlds | Adds complexity, needs tuning |
| **User-controlled flag** | Explicit control | User burden |

**Recommendation:** **Automatic threshold** (e.g., sparse if m > 100 or A nnz < 0.5 m n).

---

## 14. Proposed Architecture

### New components needed:

```
solver/numerical/include/sovereign/sparse_cholesky.hpp
solver/numerical/src/sparse_cholesky.cpp
  - SparseCholeskyLDLT class
  - symbolic_analyze(pattern) → ordering, elimination tree, fill pattern
  - numeric_factor(values) → L and D arrays
  - solve(rhs) → forward/backward substitution

solver/numerical/include/sovereign/ordering.hpp
solver/numerical/src/ordering.cpp
  - amd_ordering(pattern) → permutation
  - (future: colamd, nested dissection)

solver/lp/src/interior_point.cpp modifications:
  - build_normal_eq_sparse(A, d) → SparseMatrixCSC (not dense!)
  - SparseCholeskyLDLT member in IPM state
  - One-time symbolic analysis before iteration loop
  - Numerical refactor each iteration
  - Keep dense path as fallback

tests/unit/test_sparse_cholesky.cpp
  - Correctness: sparse vs dense on small matrices
  - Symmetric pattern detection
  - Ordering validation
  - Numerical stability (condition number)

tests/lp/test_ipm_sparse.cpp
  - Dense vs sparse IPM comparison
  - Large sparse LP benchmarks
  - Memory profiling

benchmarks/runners/benchmark_ipm_scale.py
  - Varying m, n, sparsity
  - Measure assembly, factor, solve time separately
  - Memory high-water mark
```

### Modified files:
```
solver/lp/src/interior_point.cpp
  - Replace build_normal_eq() with sparse version
  - Add backend selection logic
  - Integrate SparseCholeskyLDLT

solver/lp/include/sovereign/interior_point.hpp
  - Add IpmBackend enum {Auto, Dense, Sparse}
  - Add backend option to InteriorPointOptions

solver/CMakeLists.txt
  - Add sparse_cholesky.cpp, ordering.cpp

tests/CMakeLists.txt
  - Add test_sparse_cholesky.cpp, test_ipm_sparse.cpp
```

### Files NOT modified:
```
solver/numerical/include/sovereign/dense_lu.hpp (keep as-is)
solver/numerical/src/dense_lu.cpp (keep as-is)
solver/numerical/include/sovereign/sparse_lu.hpp (unchanged)
solver/numerical/src/sparse_lu.cpp (unchanged)
solver/lp/src/revised_simplex.cpp (unchanged)
solver/lp/src/dual_simplex.cpp (unchanged)
```

---

## 15. Implementation Risks

### High risk:
1. **Numerical instability:** SPD factorization can fail on ill-conditioned M
   - Mitigation: Adaptive regularization, modified Cholesky fallback
   
2. **Fill-in explosion:** Bad ordering can create denser L than M
   - Mitigation: Test ordering quality on real problems
   
3. **Slower on small problems:** Sparse overhead beats dense only at scale
   - Mitigation: Automatic backend selection threshold

### Medium risk:
4. **Pattern changes during IPM:** If D has exact zeros, pattern of M changes
   - Mitigation: Detect pattern change, re-analyze if needed
   
5. **Integration bugs:** Wrong indexing in sparse assembly
   - Mitigation: Comprehensive small-matrix tests vs dense

### Low risk:
6. **Regression in existing functionality:** Simplex, MILP unaffected
   - Mitigation: Full test suite must pass

---

## 16. Validation Strategy

### Phase 1: Sparse Cholesky correctness
- Implement SparseCholeskyLDLT
- Test on dense → sparse converted matrices (5×5 to 100×100)
- Compare against DenseLU on symmetric matrices
- Tolerance: 1e-10 relative error

### Phase 2: Ordering quality
- Implement AMD
- Measure fill-in on test matrices
- Compare natural vs AMD ordering
- Target: <2× fill-in on transportation problems

### Phase 3: Sparse normal equation assembly
- Implement build_normal_eq_sparse()
- Compare against dense assembly on small problems
- Tolerance: 1e-12 (assembly should be exact)

### Phase 4: IPM integration
- Integrate sparse backend into solve_newton()
- Run full LP test suite with sparse backend
- Compare objectives: dense vs sparse within 1e-7

### Phase 5: Scale validation
- Generate large sparse LPs (m=1000, 5000, 10000)
- Measure:
  - Total IPM time
  - Peak memory
  - Iteration count (should be identical)
- Target: 10× speedup at m=5000

---

## 17. Success Criteria

### Correctness:
- ✅ All existing LP tests pass with sparse backend
- ✅ Dense vs sparse objectives agree within 1e-7
- ✅ Verification passes (primal/dual residuals, optimality gap)
- ✅ No regression in simplex, MILP, QP

### Performance:
- ✅ Sparse faster than dense for m ≥ 500
- ✅ Can solve m=10,000 LP in <60s (currently impossible)
- ✅ Peak memory < 1 GB for m=10,000

### Robustness:
- ✅ Handles ill-conditioned matrices (regularization)
- ✅ Detects and recovers from factorization failure
- ✅ Falls back to dense on small problems (m < 100)

---

## 18. Open Questions

1. **Should symbolic analysis be cached across solves?**
   - If solving similar LPs repeatedly, could save ~10% time
   - Adds API complexity (warmstart-like interface)
   - Decision: Not in Phase 1, consider later

2. **Threshold for automatic sparse/dense selection?**
   - Depends on matrix structure, not just size
   - Crossover point likely m=50-200
   - Decision: Start with m > 100, tune based on benchmarks

3. **How to handle D with exact zeros?**
   - Skipping column (current code does `if (dj == 0.0) continue`)
   - Might change sparsity pattern of M
   - Decision: Detect pattern change, re-analyze if needed (rare)

4. **Should we support augmented system later?**
   - Some IPMs solve [ -D  A^T ] [ dx ] = [ ... ]
                      [  A   0  ] [ dy ]
   - Avoids forming A D A^T explicitly
   - More complex, potentially better conditioning
   - Decision: Not in scope for Phase 1, design allows future extension

---

## 19. Timeline Estimate

**Phase 1 - Sparse Cholesky (3-5 days):**
- Implement SparseCholeskyLDLT::symbolic_analyze()
- Implement SparseCholeskyLDLT::numeric_factor()
- Implement solve() and solve_transpose()
- Write unit tests vs DenseLU

**Phase 2 - Ordering (2 days):**
- Implement AMD ordering
- Test on various sparsity patterns
- Validate fill-in reduction

**Phase 3 - Assembly (2 days):**
- Implement build_normal_eq_sparse()
- Test vs dense assembly
- Handle edge cases (empty rows, duplicates)

**Phase 4 - Integration (2-3 days):**
- Modify interior_point.cpp
- Add backend selection logic
- Run full test suite
- Debug any integration issues

**Phase 5 - Benchmarking (2 days):**
- Create large sparse LPs
- Measure time/memory
- Tune thresholds
- Document results

**Total: 11-14 days**

---

## 20. Conclusion

The current IPM implementation is **fundamentally limited** by dense normal equations. The path to sparse IPM is **clear and well-defined**:

1. **SparseLU cannot be reused** - need dedicated symmetric factorization
2. **Sparsity pattern is fixed** - symbolic analysis can be cached
3. **Three Newton solves per iteration** - high payoff for optimization
4. **DenseLU must remain** - reference and fallback
5. **Clean separation** - IPM upgrade doesn't affect simplex/MILP

**Biggest risk:** Numerical stability of sparse Cholesky on ill-conditioned problems  
**Biggest opportunity:** Unlock 100k-constraint problems currently impossible

**Next step:** Proceed to Phase 2 (design approaches) per brainstorming workflow.

---

## Appendix: File Location Reference

```
solver/
├── lp/
│   ├── include/sovereign/
│   │   ├── interior_point.hpp         # InteriorPointOptions, InteriorPointSolver
│   │   ├── lp_solver.hpp              # LpSolver (dispatches to IPM/simplex)
│   │   ├── revised_simplex.hpp
│   │   └── dual_simplex.hpp
│   └── src/
│       ├── interior_point.cpp         # ← PRIMARY TARGET
│       ├── lp_solver.cpp              # Backend selection logic
│       ├── revised_simplex.cpp        # Uses SparseLU
│       └── dual_simplex.cpp           # Uses SparseLU
├── numerical/
│   ├── include/sovereign/
│   │   ├── dense_lu.hpp               # Keep unchanged
│   │   ├── sparse_lu.hpp              # Keep unchanged
│   │   ├── sparse_cholesky.hpp        # ← NEW
│   │   └── ordering.hpp               # ← NEW
│   └── src/
│       ├── dense_lu.cpp               # Keep unchanged
│       ├── sparse_lu.cpp              # Keep unchanged
│       ├── sparse_cholesky.cpp        # ← NEW
│       └── ordering.cpp               # ← NEW
├── sparse/
│   ├── include/sovereign/
│   │   └── sparse_matrix.hpp          # SparseMatrixCSC (unchanged)
│   └── src/
│       └── sparse_matrix.cpp          # (unchanged)
└── CMakeLists.txt                     # Add new files

tests/
├── lp/
│   ├── test_lp_solver.cpp             # Existing LP tests
│   ├── test_dual_simplex.cpp
│   └── test_ipm_sparse.cpp            # ← NEW
└── unit/
    ├── test_sparse_lu.cpp             # Existing
    ├── test_dense_lu.cpp              # Existing
    └── test_sparse_cholesky.cpp       # ← NEW

benchmarks/
├── runners/
│   ├── run_benchmarks.py              # Existing
│   └── benchmark_ipm_scale.py         # ← NEW
└── reports/
    └── EVIDENCE.md                    # Update with new results
```
