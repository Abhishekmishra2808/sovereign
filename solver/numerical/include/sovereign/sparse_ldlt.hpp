#pragma once

#include "sovereign/sparse_symmetric.hpp"

#include <cstddef>
#include <vector>

namespace sovereign {

// Sparse symmetric positive-definite factorization: M = L D L^T
// where L is unit lower triangular, D is positive diagonal
//
// Two-phase approach:
// 1. Symbolic analysis (once): compute sparsity structure of L
// 2. Numerical factorization (per iteration): compute L and D values
class SparseLDLT {
 public:
  SparseLDLT() = default;

  // Phase 1: Symbolic analysis
  //
  // Computes:
  // - Fill-reducing permutation (AMD)
  // - Elimination tree
  // - Sparsity pattern of L (lower triangle, unit diagonal implicit)
  // - Column pointers and allocated storage
  //
  // This structure is reused across multiple numerical factorizations
  // as long as the sparsity pattern doesn't change.
  //
  // Returns false on error.
  bool symbolic_analyze(const SparseSymmetricPattern& pattern);

  // Phase 2: Numerical factorization
  //
  // Given numerical values matching the pattern from symbolic_analyze,
  // compute L and D such that P M P^T = L D L^T where P is the AMD permutation.
  //
  // Args:
  //   values: numerical values of M (size must match symbolic pattern nnz)
  //   regularization: lambda added to diagonal (M_reg = M + lambda*I)
  //
  // Returns false if factorization fails (negative/zero pivot, numerical breakdown).
  //
  // Diagnostics recorded in member variables (min_pivot, etc.)
  bool numeric_factor(const std::vector<double>& values, double regularization = 0.0);

  // Phase 3: Solve M x = b
  //
  // Given factorization L D L^T = P M P^T, solve for x.
  //
  // Steps:
  //   y = P b        (permute RHS)
  //   L z = y        (forward solve, unit lower triangular)
  //   D w = z        (diagonal solve)
  //   L^T v = w      (backward solve, unit upper triangular)
  //   x = P^T v      (inverse permute)
  //
  // Args:
  //   x: input b, output solution (overwritten)
  //
  // Returns false if factorization not available or solve fails.
  bool solve(std::vector<double>& x) const;

  // Query state
  bool ok() const { return ok_; }
  std::size_t n() const { return n_; }
  std::size_t factor_nnz() const;
  double fill_ratio() const;  // nnz(L) / nnz(M_input)

  // Timing (seconds)
  double symbolic_time_seconds() const { return symbolic_time_; }
  double numeric_time_seconds() const { return numeric_time_; }
  double solve_time_seconds() const { return solve_time_; }

  // Numerical diagnostics
  double min_pivot() const { return min_pivot_; }
  double regularization_used() const { return regularization_used_; }

 private:
  std::size_t n_ = 0;
  bool ok_ = false;

  // Symbolic phase results
  std::vector<int> perm_;      // AMD permutation: perm[k] = original col
  std::vector<int> iperm_;     // Inverse: iperm[j] = position of col j
  std::vector<int> parent_;    // Elimination tree: parent[k] = parent of col k
  std::vector<int> l_ptr_;     // Column pointers for L (size n+1)
  std::vector<int> l_idx_;     // Row indices for L (excluding unit diagonal)

  // Numerical phase results
  std::vector<double> l_val_;  // L values (unit diagonal implicit)
  std::vector<double> d_;      // Diagonal D (size n)

  // Workspace for numerical factorization
  std::vector<double> x_;      // Dense workspace
  std::vector<int> pattern_;   // Nonzero pattern workspace
  std::vector<int> flag_;      // Marking array

  // Statistics
  double symbolic_time_ = 0.0;
  double numeric_time_ = 0.0;
  mutable double solve_time_ = 0.0;
  double min_pivot_ = 0.0;
  double regularization_used_ = 0.0;
  std::size_t input_nnz_ = 0;  // nnz of input pattern for fill ratio
};

}  // namespace sovereign
