#include "sovereign/sparse_symmetric.hpp"

#include <algorithm>
#include <unordered_set>
#include <vector>

namespace sovereign {

bool SparseSymmetricPattern::is_valid() const {
  if (col_ptr.size() != n + 1) return false;
  if (col_ptr[0] != 0) return false;
  if (static_cast<std::size_t>(col_ptr[n]) != row_idx.size()) return false;

  for (std::size_t j = 0; j < n; ++j) {
    const int start = col_ptr[j];
    const int end = col_ptr[j + 1];
    if (start > end) return false;

    // Check lower triangle: row_idx[p] >= j
    // Check sorted: row_idx[p] <= row_idx[p+1]
    for (int p = start; p < end; ++p) {
      const int i = row_idx[static_cast<std::size_t>(p)];
      if (i < static_cast<int>(j)) return false;  // Not lower triangle
      if (i < 0 || i >= static_cast<int>(n)) return false;  // Out of bounds
      if (p + 1 < end && row_idx[static_cast<std::size_t>(p + 1)] < i) {
        return false;  // Not sorted
      }
    }
  }
  return true;
}

void SparseSymmetricPattern::clear() {
  n = 0;
  col_ptr.clear();
  row_idx.clear();
}

SparseSymmetricPattern build_normal_eq_pattern(const SparseMatrixCSC& A) {
  const int m = static_cast<int>(A.nrows);
  const int n = static_cast<int>(A.ncols);

  SparseSymmetricPattern pattern;
  pattern.n = static_cast<std::size_t>(m);
  pattern.col_ptr.assign(static_cast<std::size_t>(m) + 1, 0);

  // For each column k of M (lower triangle only):
  // M[i,k] is structurally nonzero if ∃ j: A[i,j] ≠ 0 and A[k,j] ≠ 0
  //
  // Algorithm:
  // For each column j of A:
  //   Let rows = {i : A[i,j] ≠ 0}
  //   For each pair (i, k) in rows × rows where i >= k:
  //     Mark M[i,k] as structurally nonzero

  // Use sets to track unique entries per column of M
  std::vector<std::unordered_set<int>> m_cols(static_cast<std::size_t>(m));

  for (int j = 0; j < n; ++j) {
    // Collect rows with nonzeros in column j of A
    std::vector<int> rows;
    for (int p = A.col_ptr[static_cast<std::size_t>(j)];
         p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      rows.push_back(A.row_idx[static_cast<std::size_t>(p)]);
    }

    // For each pair (i, k) where both appear in this column
    for (std::size_t a = 0; a < rows.size(); ++a) {
      const int i = rows[a];
      for (std::size_t b = 0; b < rows.size(); ++b) {
        const int k = rows[b];
        if (i >= k) {
          // M[i,k] is structurally nonzero (lower triangle)
          m_cols[static_cast<std::size_t>(k)].insert(i);
        }
      }
    }
  }

  // Ensure diagonal is always present (needed for regularization)
  for (int k = 0; k < m; ++k) {
    m_cols[static_cast<std::size_t>(k)].insert(k);
  }

  // Convert sets to sorted CSC format
  int nnz = 0;
  for (int k = 0; k < m; ++k) {
    pattern.col_ptr[static_cast<std::size_t>(k)] = nnz;

    // Extract and sort row indices
    std::vector<int> col_rows(m_cols[static_cast<std::size_t>(k)].begin(),
                               m_cols[static_cast<std::size_t>(k)].end());
    std::sort(col_rows.begin(), col_rows.end());

    for (int i : col_rows) {
      pattern.row_idx.push_back(i);
      ++nnz;
    }
  }
  pattern.col_ptr[static_cast<std::size_t>(m)] = nnz;

  return pattern;
}

void build_normal_eq_values(
    const SparseMatrixCSC& A,
    const std::vector<double>& d,
    const SparseSymmetricPattern& pattern,
    std::vector<double>& values
) {
  const int m = static_cast<int>(A.nrows);
  const int n = static_cast<int>(A.ncols);

  // Resize output
  values.assign(pattern.nnz(), 0.0);

  // For each structural nonzero M[i,k] in pattern:
  //   M[i,k] = Σ_j A[i,j] * D[j] * A[k,j]
  //
  // Algorithm: For each column j of A, accumulate contributions
  // to all M[i,k] where both A[i,j] and A[k,j] are nonzero

  for (int j = 0; j < n; ++j) {
    const double dj = d[static_cast<std::size_t>(j)];
    if (dj == 0.0) continue;  // Skip if D[j] = 0

    // Collect (row, value) pairs for column j of A
    std::vector<std::pair<int, double>> col_j;
    for (int p = A.col_ptr[static_cast<std::size_t>(j)];
         p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const int i = A.row_idx[static_cast<std::size_t>(p)];
      const double aij = A.values[static_cast<std::size_t>(p)];
      col_j.push_back({i, aij});
    }

    // For each pair (i, k) in col_j × col_j where i >= k:
    //   M[i,k] += A[i,j] * D[j] * A[k,j]
    for (std::size_t a = 0; a < col_j.size(); ++a) {
      const int i = col_j[a].first;
      const double aij = col_j[a].second;

      for (std::size_t b = 0; b < col_j.size(); ++b) {
        const int k = col_j[b].first;
        const double akj = col_j[b].second;

        if (i >= k) {
          // Find position of M[i,k] in pattern
          // Column k, search for row i
          const int col_start = pattern.col_ptr[static_cast<std::size_t>(k)];
          const int col_end = pattern.col_ptr[static_cast<std::size_t>(k) + 1];

          // Binary search for row i in sorted row_idx[col_start:col_end]
          auto it = std::lower_bound(
              pattern.row_idx.begin() + col_start,
              pattern.row_idx.begin() + col_end,
              i
          );

          if (it != pattern.row_idx.begin() + col_end && *it == i) {
            const int pos = static_cast<int>(it - pattern.row_idx.begin());
            values[static_cast<std::size_t>(pos)] += aij * dj * akj;
          }
          // If not found, pattern is inconsistent (should not happen)
        }
      }
    }
  }

  // Note: Regularization (+ λ I) is NOT added here
  // It will be added during factorization
}

}  // namespace sovereign
