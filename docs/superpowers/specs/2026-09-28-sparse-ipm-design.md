# Sparse IPM Linear Algebra - Detailed Design
**Date:** 2026-09-28  
**Status:** Design Phase  
**Target:** Replace dense m×m normal equations with sparse Cholesky factorization

---

## 1. Mathematical Foundation

### Normal Equation System

The IPM Newton step solves:
```
M dy = rhs    where M = A D A^T
```

**Properties proven from code analysis:**

1. **D is strictly positive diagonal:**
   ```cpp
   d[j] = x[j] / s[j]    where x[j], s[j] > 1e-16
   ```
   - IPM maintains x > 0, s > 0 (interior point constraint)
   - Code floors both at 1e-16
   - Therefore d[j] ≥ 1e-16 / 1e-16 = 1.0

2. **M is symmetric positive semi-definite:**
   ```
   v^T M v = v^T A D A^T v = (A^T v)^T D (A^T v) = ||D^{1/2} A^T v||^2 ≥ 0
   ```

3. **M becomes positive definite with regularization:**
   ```
   M_reg = M + ε I    where ε = 1e-12 (current hardcoded value)
   ```

4. **Rank considerations:**
   - If rank(A) = m: M is positive definite (no regularization needed)
   - If rank(A) < m: M is singular → regularization essential

**Design decision: Use sparse Cholesky factorization**

Since M is guaranteed symmetric positive semi-definite and regularization makes it positive definite, we use:

```
M_reg = L D L^T
```

Where:
- L is unit lower triangular
- D is positive diagonal

This is **not** the indefinite LDL^T with 2×2 pivots. It's the symmetric positive-definite LDL^T (equivalent to Cholesky LL^T but without square roots).

**Rationale:**
- ✅ Exploits positive-definiteness guaranteed by IPM
- ✅ No square roots (more stable than LL^T)
- ✅ Simple diagonal pivoting (check D[k] > 0)
- ✅ No 2×2 indefinite blocks needed
- ✅ ~500 lines simpler than indefinite factorization

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   IPM Iteration Loop                        │
│  (solve_ipm in interior_point.cpp)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │   solve_newton()           │
        │  Compute D = X S^{-1}      │
        └────────────┬───────────────┘
                     │
                     ▼
        ┌────────────────────────────┐
        │ Backend Selection          │
        │ (auto/sparse/dense)        │
        └───┬──────────────────┬─────┘
            │                  │
      SPARSE│                  │DENSE
            │                  │
            ▼                  ▼
  ┌──────────────────┐  ┌───────────────┐
  │ Sparse Path      │  │ Dense Path    │
  │                  │  │ (unchanged)   │
  └─────┬────────────┘  └───────────────┘
        │
        ▼
  ┌──────────────────────────────────────┐
  │ FIRST ITERATION ONLY                 │
  │ 1. build_normal_eq_pattern(A)        │
  │    → SparseSymmetricPattern          │
  │ 2. AMD ordering                      │
  │ 3. Symbolic factorization            │
  └────────┬─────────────────────────────┘
           │
           ▼
  ┌──────────────────────────────────────┐
  │ EVERY ITERATION                      │
  │ 1. build_normal_eq_values(A, D)      │
  │    → numerical values of M           │
  │ 2. Numerical factorization           │
  │    M = L D L^T                       │
  │ 3. Solve: L D L^T dy = rhs           │
  │    - Forward: L z = rhs              │
  │    - Diagonal: D w = z               │
  │    - Backward: L^T dy = w            │
  └──────────────────────────────────────┘
```

---

## 3. Component Design

### 3.1 Sparse Symmetric Pattern

**File:** `solver/numerical/include/sovereign/sparse_symmetric.hpp`

```cpp
namespace sovereign {

// Sparse symmetric matrix pattern (lower triangle only, CSC format)
// Entry (i,j) stored only if i >= j
struct SparseSymmetricPattern {
  std::size_t n = 0;               // Matrix dimension
  std::vector<int> col_ptr;        // Size n+1: col_ptr[j] to col_ptr[j+1]-1
  std::vector<int> row_idx;        // Row indices, sorted, i >= j
  
  std::size_t nnz() const { return row_idx.size(); }
  
  // Validate pattern
  bool is_valid() const;
};

// Build pattern of M = A D A^T (lower triangle only)
// This is computed ONCE before IPM iterations
SparseSymmetricPattern build_normal_eq_pattern(const SparseMatrixCSC& A);

// Assemble numerical values of M into pre-allocated array
// Pattern must match result of build_normal_eq_pattern(A)
// This is called EVERY IPM iteration with new diagonal D
void build_normal_eq_values(
    const SparseMatrixCSC& A,
    const std::vector<double>& d,              // Diagonal D (size n)
    const SparseSymmetricPattern& pattern,
    std::vector<double>& values                // Output: size pattern.nnz()
);

}  // namespace sovereign
```

**Pattern construction algorithm:**

For M = A D A^T, entry M[i,k] is nonzero iff:
```
∃ column j: both A[i,j] ≠ 0 and A[k,j] ≠ 0
```

Algorithm:
1. For each column j of A:
   - Let rows = {i : A[i,j] ≠ 0}
   - For each pair (i, k) in rows × rows where i ≥ k:
     - Mark M[i,k] as structurally nonzero
2. Build CSC lower triangle from marked entries
3. Add diagonal entries (always present after regularization)

**Value assembly algorithm:**

For each structural nonzero M[i,k] in pattern:
```
M[i,k] = Σ_j A[i,j] * D[j] * A[k,j]
```

Sum over columns j where both A[i,j] and A[k,j] are nonzero.

**Critical:** Use pattern to guide summation, not hash table.

---

### 3.2 AMD Ordering

**File:** `solver/numerical/include/sovereign/amd_ordering.hpp`

```cpp
namespace sovereign {

// Approximate Minimum Degree ordering for sparse symmetric matrices
// Returns permutation P such that P M P^T has reduced fill-in
class AMDOrdering {
 public:
  // Compute ordering from symmetric pattern (lower triangle)
  // Returns permutation: perm[k] = original column for position k
  // Also computes inverse: iperm[j] = position of column j
  bool compute(
      const SparseSymmetricPattern& pattern,
      std::vector<int>& perm,
      std::vector<int>& iperm
  );
  
  // Statistics
  double ordering_time_seconds() const { return ordering_time_; }
  
 private:
  double ordering_time_ = 0.0;
};

}  // namespace sovereign
```

**Algorithm:** Classic AMD (Amestoy-Davis-Duff):

1. Construct adjacency graph of M (symmetric pattern)
2. Iteratively eliminate vertices:
   - Select vertex with minimum approximate degree
   - Form element (clique of neighbors)
   - Update approximate degrees
3. Elimination order = permutation

**Implementation notes:**
- Use degree heap for O(n log n) complexity
- Approximate degree (includes elements, not exact)
- Aggressive absorption to reduce graph size
- ~300-400 lines

**Alternatives NOT implemented initially:**
- COLAMD (for rectangular A)
- Nested dissection
- METIS
- Can add later based on benchmarks

---

### 3.3 Sparse Cholesky LDL^T

**File:** `solver/numerical/include/sovereign/sparse_cholesky.hpp`

```cpp
namespace sovereign {

// Sparse symmetric positive-definite factorization: M = L D L^T
// where L is unit lower triangular, D is positive diagonal
class SparseCholesky {
 public:
  // Two-phase factorization
  
  // Phase 1: Symbolic analysis (done once)
  // Computes:
  // - Fill-reducing permutation (AMD)
  // - Elimination tree
  // - Sparsity pattern of L
  // - Column counts of L
  bool symbolic_analyze(const SparseSymmetricPattern& pattern);
  
  // Phase 2: Numerical factorization (done every iteration)
  // Computes L and D from numerical values
  // Input values must match pattern from symbolic_analyze
  // Regularization lambda added to diagonal if > 0
  bool numeric_factor(
      const std::vector<double>& values,
      double regularization = 0.0
  );
  
  // Solve M x = b (after successful numeric_factor)
  // b input in x, solution returned in x
  bool solve(std::vector<double>& x) const;
  
  // Query
  bool ok() const { return ok_; }
  std::size_t n() const { return n_; }
  std::size_t factor_nnz() const;
  double fill_ratio() const;  // nnz(L) / nnz(M)
  
  // Timing
  double symbolic_time_seconds() const { return symbolic_time_; }
  double numeric_time_seconds() const { return numeric_time_; }
  double solve_time_seconds() const { return solve_time_; }
  
 private:
  std::size_t n_ = 0;
  bool ok_ = false;
  
  // Ordering
  std::vector<int> perm_;      // Original column -> factorization position
  std::vector<int> iperm_;     // Factorization position -> original column
  
  // Elimination tree
  std::vector<int> parent_;    // parent[k] = parent of column k in tree
  
  // Factorization storage (unit lower triangular L, diagonal D)
  std::vector<int> l_ptr_;     // Size n+1
  std::vector<int> l_idx_;     // Row indices (excluding diagonal)
  std::vector<double> l_val_;  // Values (L[j,j] = 1 implicit)
  std::vector<double> d_;      // Diagonal D (size n)
  
  // Workspace for numerical factorization
  std::vector<double> x_;
  std::vector<int> pattern_;
  std::vector<int> flag_;
  
  // Timing
  double symbolic_time_ = 0.0;
  double numeric_time_ = 0.0;
  mutable double solve_time_ = 0.0;
};

}  // namespace sovereign
```

**Symbolic analysis algorithm:**

1. Compute AMD ordering
2. Permute pattern: P M P^T
3. Build elimination tree via path compression
4. Count column nonzeros of L (symbolic factorization)
5. Allocate L storage

**Numerical factorization algorithm (left-looking):**

```
For k = 0 to n-1:
  Scatter column k of permuted M into workspace x
  
  For each ancestor j of k in elimination tree:
    if L[k,j] ≠ 0:
      Perform triangular solve: x[k] -= L[k,j] * D[j] * L[k,j]
  
  Extract diagonal: D[k] = x[k]
  
  Check: D[k] > 0  (positive-definiteness)
  
  For each i > k in pattern of column k:
    L[i,k] = x[i] / D[k]
```

**Solve algorithm:**

```
Forward solve:  L z = b      (unit lower triangular)
Diagonal solve: D w = z      (diagonal)
Backward solve: L^T x = w    (unit upper triangular)
Apply inverse permutation: x_orig = P^T x
```

**Regularization:**

If numeric_factor fails (D[k] ≤ 0 or very small):
```
M_reg = M + λ I
```

Retry with λ = 1e-12, 1e-10, 1e-8, 1e-6

If still fails → return error, caller falls back to dense

**Key invariant:** Diagonal D[k] checked > epsilon (e.g., 1e-14) for numerical safety

---

### 3.4 IPM Integration

**File:** `solver/lp/src/interior_point.cpp` (modifications)

**New IPM state:**

```cpp
struct IpmLinearSolverState {
  // Backend selection
  enum class Backend { Dense, Sparse, Auto };
  Backend backend = Backend::Auto;
  
  // Sparse state
  struct SparseState {
    SparseSymmetricPattern pattern;
    SparseCholesky cholesky;
    bool symbolic_done = false;
    std::vector<double> m_values;
  };
  
  // Dense state (existing)
  struct DenseState {
    // Uses DenseLU directly each iteration (no persistent state)
  };
  
  std::unique_ptr<SparseState> sparse;
  
  // Metrics
  int numeric_factor_count = 0;
  double total_linear_solve_time = 0.0;
};
```

**Modified solve_newton:**

```cpp
bool solve_newton_with_backend(
    const IpmLp& lp,
    const std::vector<double>& x,
    const std::vector<double>& s,
    const std::vector<double>& rp,
    const std::vector<double>& rd,
    const std::vector<double>& rxs,
    std::vector<double>& dx,
    std::vector<double>& dy,
    std::vector<double>& ds,
    IpmLinearSolverState& state
) {
  // Compute D = X S^{-1} (unchanged)
  std::vector<double> d(n);
  for (int j = 0; j < n; ++j) {
    d[j] = std::max(x[j], 1e-16) / std::max(s[j], 1e-16);
  }
  
  // Compute RHS (unchanged)
  // ...
  
  // Backend selection
  IpmLinearSolverState::Backend backend = state.backend;
  if (backend == IpmLinearSolverState::Backend::Auto) {
    // Conservative threshold: use sparse if m > 200 and sparse
    const int m = lp.m;
    const int n = lp.n;
    const std::size_t nnz = lp.A.nnz();
    const double density = static_cast<double>(nnz) / (m * n);
    
    if (m > 200 && density < 0.1) {
      backend = IpmLinearSolverState::Backend::Sparse;
    } else {
      backend = IpmLinearSolverState::Backend::Dense;
    }
  }
  
  // Sparse path
  if (backend == IpmLinearSolverState::Backend::Sparse) {
    if (!state.sparse) {
      state.sparse = std::make_unique<IpmLinearSolverState::SparseState>();
    }
    
    // First iteration: symbolic analysis
    if (!state.sparse->symbolic_done) {
      state.sparse->pattern = build_normal_eq_pattern(lp.A);
      if (!state.sparse->cholesky.symbolic_analyze(state.sparse->pattern)) {
        // Fall back to dense
        return solve_newton_dense(lp, ...);
      }
      state.sparse->symbolic_done = true;
    }
    
    // Every iteration: numerical factorization
    build_normal_eq_values(lp.A, d, state.sparse->pattern, state.sparse->m_values);
    
    // Add regularization
    const double reg = 1e-12;
    if (!state.sparse->cholesky.numeric_factor(state.sparse->m_values, reg)) {
      // Try stronger regularization
      if (!state.sparse->cholesky.numeric_factor(state.sparse->m_values, 1e-10)) {
        // Fall back to dense
        return solve_newton_dense(lp, ...);
      }
    }
    
    // Solve
    dy = rhs;
    if (!state.sparse->cholesky.solve(dy)) {
      return false;
    }
    
    state.numeric_factor_count++;
  }
  
  // Dense path (unchanged)
  else {
    return solve_newton_dense(lp, ...);
  }
  
  // Recover dx, ds (unchanged)
  // ...
  
  return true;
}
```

**Environment variable:**

```cpp
const char* backend_env = std::getenv("SOVEREIGN_IPM_LINEAR_SOLVER");
if (backend_env) {
  if (std::string(backend_env) == "dense") {
    state.backend = IpmLinearSolverState::Backend::Dense;
  } else if (std::string(backend_env) == "sparse") {
    state.backend = IpmLinearSolverState::Backend::Sparse;
  } else if (std::string(backend_env) == "auto") {
    state.backend = IpmLinearSolverState::Backend::Auto;
  }
}
```

---

## 4. Testing Strategy

### 4.1 Unit Tests - Sparse Symmetric Pattern

**File:** `tests/unit/test_sparse_symmetric.cpp`

```cpp
TEST(SparseSymmetric, PatternCorrectness) {
  // Known A, compute M = A D A^T by hand
  // Build pattern, verify against expected
}

TEST(SparseSymmetric, ValueAccumulation) {
  // Small A, various D values
  // Compare sparse vs dense computation
  // Tolerance: 1e-14 (should be exact)
}

TEST(SparseSymmetric, DiagonalPresent) {
  // Verify diagonal always included in pattern
}

TEST(SparseSymmetric, SymmetryPreserved) {
  // Only lower triangle stored
}
```

### 4.2 Unit Tests - AMD Ordering

**File:** `tests/unit/test_amd.cpp`

```cpp
TEST(AMD, BasicOrdering) {
  // Small matrix, verify permutation is valid
}

TEST(AMD, FillReduction) {
  // Known pattern, verify AMD reduces fill vs natural order
}

TEST(AMD, PermutationInverse) {
  // Check perm and iperm are inverses
}
```

### 4.3 Unit Tests - Sparse Cholesky

**File:** `tests/unit/test_sparse_cholesky.cpp`

```cpp
TEST(SparseCholesky, SmallSPD) {
  // 5×5 SPD matrix
  // Factor, solve, verify ||Mx - b|| < 1e-12
}

TEST(SparseCholesky, CompareAgainstDense) {
  // Same matrix through SparseCholesky and DenseLU
  // Solutions should agree within 1e-10
}

TEST(SparseCholesky, Regularization) {
  // Near-singular matrix
  // Verify regularization recovers factorization
}

TEST(SparseCholesky, LargerSparse) {
  // 100×100 sparse SPD
  // Measure fill ratio
}

TEST(SparseCholesky, NegativePivot) {
  // Non-SPD matrix (should fail)
  // Verify returns false, not crash
}
```

### 4.4 Integration Tests - IPM

**File:** `tests/lp/test_ipm_sparse.cpp`

```cpp
TEST(IpmSparse, DenseVsSparse_ClassicLP) {
  // Run same LP with dense and sparse backends
  // Compare: status, objective, residuals, iterations
  // Tolerance: 1e-7 for objective
}

TEST(IpmSparse, Transport50x50) {
  // m=100, n=2500
  // Should use sparse automatically
  // Verify OPTIMAL status, correct objective
}

TEST(IpmSparse, ForcedBackend) {
  // Set SOVEREIGN_IPM_LINEAR_SOLVER=sparse
  // Verify sparse path used
  // Set =dense, verify dense path used
}

TEST(IpmSparse, FallbackToDense) {
  // Problem where sparse factorization fails
  // Verify dense fallback works
}
```

---

## 5. Benchmarking Plan

### 5.1 Synthetic Problems

Generate transportation LPs:
```
m = {100, 500, 1000, 5000, 10000}
n = 25*m  (typical 5% coupling)
nnz(A) ≈ 2*m*sqrt(m)
```

### 5.2 Metrics to Record

For each problem:
```
Problem size:
  m, n, nnz(A)

Pattern/symbolic (one-time):
  nnz(M)
  nnz(L)
  fill_ratio = nnz(L) / nnz(M)
  ordering_time
  symbolic_time

Per IPM iteration:
  numeric_factor_time
  solve_time

Total:
  ipm_iterations
  total_linear_algebra_time
  dense_backend_time (for comparison)
  speedup = dense_time / sparse_time

Memory:
  dense_M_bytes = m * m * 8
  sparse_M_bytes = nnz(M) * 12
  sparse_L_bytes = nnz(L) * 12
  memory_reduction = dense_M_bytes / (sparse_M_bytes + sparse_L_bytes)
```

### 5.3 Benchmark Script

**File:** `benchmarks/runners/benchmark_sparse_ipm.py`

```python
def generate_transportation_lp(m_suppliers, n_products):
    """Generate sparse transportation LP"""
    # m constraints = m_suppliers + n_products (supply + demand)
    # n variables = m_suppliers * n_products (routes)
    # Each route has 2 nonzeros in A (one supply, one demand constraint)
    pass

def run_benchmark(problem_file, backend):
    """Run solver, extract metrics from output"""
    env = {"SOVEREIGN_IPM_LINEAR_SOLVER": backend}
    result = subprocess.run(...)
    return parse_metrics(result.stdout)

def main():
    sizes = [100, 500, 1000, 5000, 10000]
    for m in sizes:
        problem = generate_transportation_lp(m, 25*m)
        
        dense_result = run_benchmark(problem, "dense")
        sparse_result = run_benchmark(problem, "sparse")
        
        print(f"m={m}:")
        print(f"  Dense time: {dense_result['time']:.3f}s")
        print(f"  Sparse time: {sparse_result['time']:.3f}s")
        print(f"  Speedup: {dense_result['time'] / sparse_result['time']:.2f}x")
        print(f"  Memory reduction: {compute_memory_ratio(dense_result, sparse_result):.1f}x")
```

---

## 6. Success Criteria

### Phase 1: Correctness
- ✅ All existing LP tests pass with sparse backend
- ✅ Dense vs sparse objectives agree within 1e-7
- ✅ Residuals match within 1e-9
- ✅ No crashes, memory leaks, or undefined behavior

### Phase 2: Memory
- ✅ Sparse backend does NOT allocate m×m dense storage
- ✅ Memory usage < 10% of dense for m=10k sparse problems

### Phase 3: Performance
- ✅ Sparse faster than dense for m ≥ 500 on typical sparse LPs
- ✅ Can solve m=10k problem (currently impossible with dense)
- ✅ Overhead < 20% for small problems (m < 200)

---

## 7. Known Limitations (Initial Version)

**NOT implemented:**
1. Supernodal factorization (dense submatrix optimization)
2. Nested dissection ordering (better for large meshes)
3. COLAMD ordering (better for rectangular A)
4. Multithreaded factorization
5. GPU acceleration
6. Iterative refinement
7. Condition number estimation

**Handled via fallback:**
1. Indefinite matrices (should not occur in IPM)
2. Rank-deficient A (regularization + fallback to dense)
3. Numerical breakdown (adaptive regularization, then dense)

**Deferred to future:**
1. Pattern change detection (rare in IPM)
2. Symbolic analysis caching across solves
3. Exploiting block-angular structure

---

## 8. File Structure

```
solver/
├── numerical/
│   ├── include/sovereign/
│   │   ├── sparse_symmetric.hpp       # NEW: Pattern + value assembly
│   │   ├── amd_ordering.hpp           # NEW: AMD implementation
│   │   ├── sparse_cholesky.hpp        # NEW: LDL^T factorization
│   │   ├── dense_lu.hpp               # UNCHANGED
│   │   └── sparse_lu.hpp              # UNCHANGED
│   └── src/
│       ├── sparse_symmetric.cpp       # NEW
│       ├── amd_ordering.cpp           # NEW
│       ├── sparse_cholesky.cpp        # NEW
│       ├── dense_lu.cpp               # UNCHANGED
│       └── sparse_lu.cpp              # UNCHANGED
└── lp/
    ├── include/sovereign/
    │   └── interior_point.hpp         # MODIFIED: Add backend option
    └── src/
        └── interior_point.cpp         # MODIFIED: Integrate sparse path

tests/
├── unit/
│   ├── test_sparse_symmetric.cpp      # NEW
│   ├── test_amd.cpp                   # NEW
│   ├── test_sparse_cholesky.cpp       # NEW
│   ├── test_dense_lu.cpp              # UNCHANGED
│   └── test_sparse_lu.cpp             # UNCHANGED
└── lp/
    ├── test_ipm_sparse.cpp            # NEW
    └── test_lp_solver.cpp             # UNCHANGED (run with both backends)

benchmarks/
└── runners/
    └── benchmark_sparse_ipm.py        # NEW
```

**Estimated LOC:**
- sparse_symmetric.cpp: ~400 lines
- amd_ordering.cpp: ~350 lines
- sparse_cholesky.cpp: ~600 lines
- interior_point.cpp modifications: ~200 lines
- Tests: ~600 lines
- **Total new code: ~2150 lines**

---

## 9. Implementation Order

1. **Sparse symmetric pattern** (1 day)
   - Pattern construction from A
   - Value assembly
   - Unit tests

2. **AMD ordering** (1 day)
   - Classic AMD algorithm
   - Unit tests

3. **Sparse Cholesky - symbolic** (1-2 days)
   - Elimination tree
   - Column counts
   - Sparsity pattern of L

4. **Sparse Cholesky - numerical** (1-2 days)
   - Left-looking factorization
   - Regularization
   - Solve

5. **IPM integration** (1 day)
   - Backend selection
   - State management
   - Fallback logic

6. **Testing** (1-2 days)
   - Unit tests
   - Integration tests
   - Dense vs sparse comparison

7. **Benchmarking** (1 day)
   - Generate problems
   - Run suite
   - Document results

**Total: 7-10 days**

---

## 10. Next Steps

After design approval:
1. Implement sparse_symmetric.cpp
2. Implement amd_ordering.cpp
3. Implement sparse_cholesky.cpp
4. Integrate into interior_point.cpp
5. Write tests
6. Benchmark
7. Document results
8. **STOP** - do not proceed to GPU/QP/MILP

---

**End of design document. Ready for implementation upon approval.**
