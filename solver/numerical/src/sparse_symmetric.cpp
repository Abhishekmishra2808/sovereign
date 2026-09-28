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

  // Column k of M is the union, over the columns j of A with A[k,j] != 0, of
  // the rows i >= k of column j. The rows of A (a transpose of its column
  // structure) and a marker array give each column in time proportional to
  // its contributions, without hashing.
  std::vector<int> row_ptr(static_cast<std::size_t>(m) + 1, 0);
  for (int j = 0; j < n; ++j) {
    for (int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      ++row_ptr[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)]) + 1];
    }
  }
  for (int k = 0; k < m; ++k) row_ptr[static_cast<std::size_t>(k) + 1] += row_ptr[static_cast<std::size_t>(k)];
  std::vector<int> row_cols(static_cast<std::size_t>(row_ptr[static_cast<std::size_t>(m)]));
  {
    std::vector<int> next(row_ptr.begin(), row_ptr.end() - 1);
    for (int j = 0; j < n; ++j) {
      for (int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
        row_cols[static_cast<std::size_t>(next[static_cast<std::size_t>(A.row_idx[static_cast<std::size_t>(p)])]++)] = j;
      }
    }
  }

  std::vector<int> mark(static_cast<std::size_t>(m), -1);
  std::vector<int> col_rows;
  for (int k = 0; k < m; ++k) {
    pattern.col_ptr[static_cast<std::size_t>(k)] = static_cast<int>(pattern.row_idx.size());
    col_rows.clear();
    // Diagonal always present (needed for regularization).
    mark[static_cast<std::size_t>(k)] = k;
    col_rows.push_back(k);
    for (int q = row_ptr[static_cast<std::size_t>(k)]; q < row_ptr[static_cast<std::size_t>(k) + 1]; ++q) {
      const int j = row_cols[static_cast<std::size_t>(q)];
      for (int p = A.col_ptr[static_cast<std::size_t>(j)]; p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
        const int i = A.row_idx[static_cast<std::size_t>(p)];
        if (i > k && mark[static_cast<std::size_t>(i)] != k) {
          mark[static_cast<std::size_t>(i)] = k;
          col_rows.push_back(i);
        }
      }
    }
    std::sort(col_rows.begin(), col_rows.end());
    pattern.row_idx.insert(pattern.row_idx.end(), col_rows.begin(), col_rows.end());
  }
  pattern.col_ptr[static_cast<std::size_t>(m)] = static_cast<int>(pattern.row_idx.size());

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
