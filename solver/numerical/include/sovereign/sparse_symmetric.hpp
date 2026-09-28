#pragma once

#include "sovereign/sparse_matrix.hpp"

#include <cstddef>
#include <vector>

namespace sovereign {

// Sparse symmetric matrix pattern (lower triangle only, CSC format)
// Entry (i,j) stored only if i >= j (lower triangle)
struct SparseSymmetricPattern {
  std::size_t n = 0;               // Matrix dimension (n x n)
  std::vector<int> col_ptr;        // Size n+1: col_ptr[j] to col_ptr[j+1]-1
  std::vector<int> row_idx;        // Row indices, i >= j, sorted within column

  std::size_t nnz() const { return row_idx.size(); }

  // Validate pattern: lower triangle, sorted, in bounds
  bool is_valid() const;

  void clear();
};

// Build sparsity pattern of M = A D A^T (lower triangle only)
//
// For M[i,k] to be structurally nonzero (i >= k):
//   ∃ column j: both A[i,j] ≠ 0 and A[k,j] ≠ 0
//
// This pattern is independent of the values in D (as long as D[j] ≠ 0).
// Computed ONCE before IPM iterations.
//
// Note: Diagonal always included (even if computed M[k,k] would be zero,
// regularization makes it nonzero).
SparseSymmetricPattern build_normal_eq_pattern(const SparseMatrixCSC& A);

// Assemble numerical values of M = A D A^T into pre-allocated array
//
// Pattern must match result of build_normal_eq_pattern(A).
// Called EVERY IPM iteration with new diagonal D.
//
// For each structural nonzero M[i,k] in pattern:
//   M[i,k] = Σ_j A[i,j] * D[j] * A[k,j]
//
// Diagonal regularization is NOT added here (done in factorization).
//
// Args:
//   A: Constraint matrix (m x n)
//   d: Diagonal D (size n), all entries > 0
//   pattern: Sparsity structure from build_normal_eq_pattern(A)
//   values: Output array (size pattern.nnz()), overwritten
void build_normal_eq_values(
    const SparseMatrixCSC& A,
    const std::vector<double>& d,
    const SparseSymmetricPattern& pattern,
    std::vector<double>& values
);

}  // namespace sovereign
