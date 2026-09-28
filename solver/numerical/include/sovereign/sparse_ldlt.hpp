#pragma once

#include "sovereign/sparse_symmetric.hpp"

#include <cstddef>
#include <utility>
#include <vector>

namespace sovereign {

// Sparse symmetric factorization without pivoting: M = L D L^T where L is
// unit lower triangular and D is diagonal. D is positive for positive-definite
// M, or has prescribed signs for quasi-definite M (see set_pivot_signs).
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
  // Returns false on error, or when max_factor_nnz > 0 and nnz(L) would
  // exceed it (fill_limit_exceeded() then reports true).
  bool symbolic_analyze(const SparseSymmetricPattern& pattern, std::size_t max_factor_nnz = 0);
  bool fill_limit_exceeded() const { return fill_limit_exceeded_; }

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

  // Interior-point normal equations are positive semidefinite and become
  // singular near convergence. With a positive threshold, a pivot at or below
  // threshold * (its diagonal entry in P M P^T) is replaced by a huge value,
  // which drops that direction instead of failing (the usual IPM treatment).
  void set_tiny_pivot_threshold(double threshold) { tiny_pivot_threshold_ = threshold; }
  int replaced_pivots() const { return replaced_pivots_; }

  // Expected sign (+1 or -1) of the pivot of each original index, for
  // quasi-definite matrices [H A^T; A -G] with H, G positive definite: any
  // symmetric ordering then factors with D carrying exactly these signs
  // (Vanderbei, SIAM J. Optim. 5(1), 1995). Empty means all positive.
  // A pivot of the wrong sign fails the factorization, and the tiny-pivot
  // test and replacement use the same sign.
  void set_pivot_signs(std::vector<signed char> signs) { pivot_sign_ = std::move(signs); }

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
  bool factored() const { return factored_; }
  std::size_t n() const { return n_; }
  std::size_t factor_nnz() const;
  double fill_ratio() const;  // nnz(L) / nnz(M_input)
  // Multiply-adds of one numeric factorization: sum over columns of count^2.
  double factor_flops() const;

  // Timing (seconds)
  double symbolic_time_seconds() const { return symbolic_time_; }
  double numeric_time_seconds() const { return numeric_time_; }
  double solve_time_seconds() const { return solve_time_; }

  // Numerical diagnostics
  double min_pivot() const { return min_pivot_; }
  double max_pivot() const { return max_pivot_; }
  double regularization_used() const { return regularization_used_; }

  // Debug accessors
  const std::vector<int>& perm() const { return perm_; }
  const std::vector<int>& iperm() const { return iperm_; }
  const std::vector<int>& l_ptr() const { return l_ptr_; }
  const std::vector<int>& l_idx() const { return l_idx_; }
  const std::vector<double>& l_val() const { return l_val_; }
  const std::vector<double>& d() const { return d_; }

 private:
  std::size_t n_ = 0;
  bool ok_ = false;
  bool factored_ = false;
  bool fill_limit_exceeded_ = false;

  // Symbolic phase results
  std::vector<int> perm_;      // AMD permutation: perm[k] = original col
  std::vector<int> iperm_;     // Inverse: iperm[j] = position of col j
  std::vector<int> parent_;    // Elimination tree: parent[k] = parent of col k
  std::vector<int> l_ptr_;     // Column pointers for L (size n+1)
  std::vector<int> l_idx_;     // Row indices for L (excluding unit diagonal)

  // Upper triangle of P M P^T in CSC (column k holds rows <= k). c_map_[p] is
  // where input entry p lands, so each numeric factorization is one scatter.
  std::vector<int> c_ptr_;
  std::vector<int> c_idx_;
  std::vector<int> c_map_;
  std::vector<double> c_val_;

  // Numerical phase results
  std::vector<double> l_val_;  // L values (unit diagonal implicit)
  std::vector<double> d_;      // Diagonal D (size n)

  // Workspace for numerical factorization
  std::vector<double> y_;      // Dense accumulator for row k of L
  std::vector<int> pattern_;   // Nonzero pattern of row k of L
  std::vector<int> flag_;      // Marking array
  std::vector<int> l_len_;     // Entries filled so far in each column of L
  mutable std::vector<double> work_;  // Solve workspace

  // Statistics
  double symbolic_time_ = 0.0;
  double numeric_time_ = 0.0;
  mutable double solve_time_ = 0.0;
  double min_pivot_ = 0.0;
  double max_pivot_ = 0.0;
  double regularization_used_ = 0.0;
  double tiny_pivot_threshold_ = 0.0;
  int replaced_pivots_ = 0;
  std::vector<signed char> pivot_sign_;
  std::size_t input_nnz_ = 0;  // nnz of input pattern for fill ratio
};

}  // namespace sovereign
