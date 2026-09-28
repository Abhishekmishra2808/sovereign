#include "sovereign/sparse_ldlt.hpp"

#include "sovereign/amd_ordering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

namespace sovereign {

namespace {

// Build elimination tree via path compression
// parent[j] = first row index > j in column j (after it's been processed)
void compute_elimination_tree(
    const SparseSymmetricPattern& pattern,
    const std::vector<int>& perm,
    const std::vector<int>& iperm,
    std::vector<int>& parent
) {
  const int n = static_cast<int>(pattern.n);
  parent.assign(static_cast<std::size_t>(n), -1);

  std::vector<int> ancestor(static_cast<std::size_t>(n), -1);

  for (int k = 0; k < n; ++k) {
    const int j = perm[static_cast<std::size_t>(k)];  // Original column
    ancestor[static_cast<std::size_t>(k)] = -1;

    const int p_start = pattern.col_ptr[static_cast<std::size_t>(j)];
    const int p_end = pattern.col_ptr[static_cast<std::size_t>(j) + 1];

    for (int p = p_start; p < p_end; ++p) {
      const int i = pattern.row_idx[static_cast<std::size_t>(p)];

      if (i == j) continue;  // Skip diagonal

      const int pi = iperm[static_cast<std::size_t>(i)];  // Permuted position
      if (pi >= k) continue;  // Only consider already-eliminated rows

      // Path compression: find root of pi's tree
      int r = pi;
      while (ancestor[static_cast<std::size_t>(r)] != -1 &&
             ancestor[static_cast<std::size_t>(r)] != k) {
        r = ancestor[static_cast<std::size_t>(r)];
      }

      if (ancestor[static_cast<std::size_t>(r)] == -1) {
        ancestor[static_cast<std::size_t>(r)] = k;
        parent[static_cast<std::size_t>(r)] = k;
      }
    }
  }
}

// Count column nonzeros for each column of L using elimination tree
void symbolic_count(
    const SparseSymmetricPattern& pattern,
    const std::vector<int>& perm,
    const std::vector<int>& iperm,
    const std::vector<int>& parent,
    std::vector<int>& l_col_count
) {
  const int n = static_cast<int>(pattern.n);
  l_col_count.assign(static_cast<std::size_t>(n), 0);

  std::vector<int> flag(static_cast<std::size_t>(n), -1);

  for (int k = 0; k < n; ++k) {
    const int j = perm[static_cast<std::size_t>(k)];

    const int p_start = pattern.col_ptr[static_cast<std::size_t>(j)];
    const int p_end = pattern.col_ptr[static_cast<std::size_t>(j) + 1];

    for (int p = p_start; p < p_end; ++p) {
      const int i = pattern.row_idx[static_cast<std::size_t>(p)];

      if (i == j) continue;  // Skip diagonal

      const int pi = iperm[static_cast<std::size_t>(i)];
      if (pi <= k) continue;  // Only lower triangle

      // Mark this row and traverse up elimination tree
      int curr = pi;
      while (curr != -1 && flag[static_cast<std::size_t>(curr)] != k) {
        flag[static_cast<std::size_t>(curr)] = k;
        l_col_count[static_cast<std::size_t>(k)]++;
        curr = parent[static_cast<std::size_t>(curr)];
      }
    }
  }
}

}  // namespace

bool SparseLDLT::symbolic_analyze(const SparseSymmetricPattern& pattern) {
  auto start = std::chrono::steady_clock::now();

  if (!pattern.is_valid()) {
    ok_ = false;
    return false;
  }

  n_ = pattern.n;
  input_nnz_ = pattern.nnz();

  // Store input pattern for numerical factorization
  input_col_ptr_ = pattern.col_ptr;
  input_row_idx_ = pattern.row_idx;

  if (n_ == 0) {
    ok_ = true;
    perm_.clear();
    iperm_.clear();
    parent_.clear();
    l_ptr_.assign(1, 0);
    l_idx_.clear();
    symbolic_time_ = 0.0;
    return true;
  }

  // Step 1: Compute AMD ordering
  AMDOrdering amd;
  if (!amd.compute(pattern, perm_, iperm_)) {
    ok_ = false;
    return false;
  }

  // Step 2: Compute elimination tree
  compute_elimination_tree(pattern, perm_, iperm_, parent_);

  // Step 3: Count column nonzeros for L
  std::vector<int> l_col_count;
  symbolic_count(pattern, perm_, iperm_, parent_, l_col_count);

  // Step 4: Build column pointers
  const int n = static_cast<int>(n_);
  l_ptr_.resize(n_ + 1);
  l_ptr_[0] = 0;
  for (int k = 0; k < n; ++k) {
    l_ptr_[static_cast<std::size_t>(k) + 1] =
        l_ptr_[static_cast<std::size_t>(k)] + l_col_count[static_cast<std::size_t>(k)];
  }

  const std::size_t l_nnz = static_cast<std::size_t>(l_ptr_[n_]);
  l_idx_.resize(l_nnz);

  // Step 5: Fill in row indices for L
  // We'll recompute the pattern to get actual row indices
  std::vector<int> l_col_pos = l_ptr_;  // Current position in each column
  std::vector<int> flag(n_, -1);

  for (int k = 0; k < n; ++k) {
    const int j = perm_[static_cast<std::size_t>(k)];

    const int p_start = pattern.col_ptr[static_cast<std::size_t>(j)];
    const int p_end = pattern.col_ptr[static_cast<std::size_t>(j) + 1];

    // Collect rows in this column
    std::vector<int> rows;

    for (int p = p_start; p < p_end; ++p) {
      const int i = pattern.row_idx[static_cast<std::size_t>(p)];

      if (i == j) continue;

      const int pi = iperm_[static_cast<std::size_t>(i)];
      if (pi <= k) continue;

      // Traverse elimination tree and collect rows
      int curr = pi;
      while (curr != -1 && flag[static_cast<std::size_t>(curr)] != k) {
        flag[static_cast<std::size_t>(curr)] = k;
        rows.push_back(curr);
        curr = parent_[static_cast<std::size_t>(curr)];
      }
    }

    // Sort rows and store
    std::sort(rows.begin(), rows.end());
    for (int row : rows) {
      const std::size_t pos = static_cast<std::size_t>(l_col_pos[static_cast<std::size_t>(k)]++);
      l_idx_[pos] = row;
    }
  }

  ok_ = true;

  auto end = std::chrono::steady_clock::now();
  symbolic_time_ = std::chrono::duration<double>(end - start).count();

  return true;
}

bool SparseLDLT::numeric_factor(const std::vector<double>& values, double regularization) {
  auto start = std::chrono::steady_clock::now();

  if (!ok_) return false;
  if (values.size() != input_nnz_) return false;

  const int n = static_cast<int>(n_);
  if (n == 0) {
    numeric_time_ = 0.0;
    regularization_used_ = regularization;
    min_pivot_ = 0.0;
    max_pivot_ = 0.0;
    return true;
  }

  // Allocate numerical storage
  const std::size_t l_nnz = l_idx_.size();
  l_val_.resize(l_nnz);
  d_.resize(n_);

  // Workspace
  x_.assign(n_, 0.0);
  flag_.assign(n_, -1);

  regularization_used_ = regularization;
  min_pivot_ = std::numeric_limits<double>::infinity();
  max_pivot_ = -std::numeric_limits<double>::infinity();

  // Left-looking sparse LDL^T factorization
  // Factor P M P^T = L D L^T column by column
  //
  // For each column k (in permuted order):
  //   1. Scatter column k of P M P^T into dense vector x_
  //   2. For each nonzero L[i,j] where j < k in column i's pattern,
  //      subtract L[i,j] * D[j] * L[k,j] from x_[i]
  //   3. Extract diagonal: D[k] = x_[k] + regularization
  //   4. Compute L[i,k] = x_[i] / D[k] for i > k

  for (int k = 0; k < n; ++k) {
    // Clear workspace
    x_.assign(n_, 0.0);

    // Step 1: Scatter column k of P M P^T into x_
    // Column k of permuted matrix corresponds to original column perm_[k]
    const int orig_col_k = perm_[static_cast<std::size_t>(k)];

    // We need to find all entries that contribute to column k of P M P^T
    // Entry [i,k] in permuted matrix comes from [perm[i], perm[k]] in original
    // Since M is symmetric and stored as lower triangle, we need to check:
    //   1. Column orig_col_k: entries M[row, orig_col_k] where row >= orig_col_k
    //   2. Row orig_col_k: entries M[orig_col_k, col] where col < orig_col_k (by symmetry)

    // Scan column orig_col_k
    const int col_start = input_col_ptr_[static_cast<std::size_t>(orig_col_k)];
    const int col_end = input_col_ptr_[static_cast<std::size_t>(orig_col_k) + 1];

    for (int p = col_start; p < col_end; ++p) {
      const int orig_row = input_row_idx_[static_cast<std::size_t>(p)];
      const double val = values[static_cast<std::size_t>(p)];
      const int perm_row = iperm_[static_cast<std::size_t>(orig_row)];

      // Entry M[orig_row, orig_col_k] contributes to position [perm_row, k]
      x_[static_cast<std::size_t>(perm_row)] += val;
    }

    // Scan row orig_col_k by looking at all columns j < orig_col_k
    for (int orig_col_j = 0; orig_col_j < orig_col_k; ++orig_col_j) {
      const int col_j_start = input_col_ptr_[static_cast<std::size_t>(orig_col_j)];
      const int col_j_end = input_col_ptr_[static_cast<std::size_t>(orig_col_j) + 1];

      // Look for entry M[orig_col_k, orig_col_j] in column orig_col_j
      for (int p = col_j_start; p < col_j_end; ++p) {
        const int orig_row = input_row_idx_[static_cast<std::size_t>(p)];

        if (orig_row == orig_col_k) {
          // Found M[orig_col_k, orig_col_j] - this contributes to [k, iperm[orig_col_j]]
          const double val = values[static_cast<std::size_t>(p)];
          const int perm_col_j = iperm_[static_cast<std::size_t>(orig_col_j)];

          // This is a column k entry at row perm_col_j
          x_[static_cast<std::size_t>(perm_col_j)] += val;
          break;
        }
      }
    }

    // Add regularization to diagonal
    x_[static_cast<std::size_t>(k)] += regularization;

    // Step 2: Update from previous columns j < k
    // For each L[i,j] where j < k, subtract L[i,j] * D[j] * L[k,j]
    // We need L[k,j] for j < k, which are the entries in column j where row index is k

    for (int j = 0; j < k; ++j) {
      // Find L[k,j] in column j of L
      const int col_j_start = l_ptr_[static_cast<std::size_t>(j)];
      const int col_j_end = l_ptr_[static_cast<std::size_t>(j) + 1];

      double l_kj = 0.0;
      bool found = false;

      for (int p = col_j_start; p < col_j_end; ++p) {
        if (l_idx_[static_cast<std::size_t>(p)] == k) {
          l_kj = l_val_[static_cast<std::size_t>(p)];
          found = true;
          break;
        }
      }

      if (!found || std::abs(l_kj) < 1e-30) continue;

      const double d_j = d_[static_cast<std::size_t>(j)];

      // Update x_[i] -= L[i,j] * D[j] * L[k,j] for all i in column j's pattern
      for (int p = col_j_start; p < col_j_end; ++p) {
        const int i = l_idx_[static_cast<std::size_t>(p)];
        const double l_ij = l_val_[static_cast<std::size_t>(p)];
        x_[static_cast<std::size_t>(i)] -= l_ij * d_j * l_kj;
      }

      // Also update diagonal: x_[k] -= D[j] * L[k,j]^2
      x_[static_cast<std::size_t>(k)] -= d_j * l_kj * l_kj;
    }

    // Step 3: Extract diagonal D[k]
    const double d_k = x_[static_cast<std::size_t>(k)];

    // Check for numerical breakdown
    if (std::isnan(d_k) || std::isinf(d_k)) {
      ok_ = false;
      return false;
    }

    if (d_k <= 0.0) {
      // Matrix is not positive definite (even after regularization)
      ok_ = false;
      return false;
    }

    if (d_k < 1e-30) {
      // Pivot too small - numerical breakdown
      ok_ = false;
      return false;
    }

    d_[static_cast<std::size_t>(k)] = d_k;
    min_pivot_ = std::min(min_pivot_, d_k);
    max_pivot_ = std::max(max_pivot_, d_k);

    // Step 4: Compute L[i,k] = x_[i] / D[k] for i > k
    const int l_col_k_start = l_ptr_[static_cast<std::size_t>(k)];
    const int l_col_k_end = l_ptr_[static_cast<std::size_t>(k) + 1];

    for (int p = l_col_k_start; p < l_col_k_end; ++p) {
      const int i = l_idx_[static_cast<std::size_t>(p)];
      l_val_[static_cast<std::size_t>(p)] = x_[static_cast<std::size_t>(i)] / d_k;
    }
  }

  auto end = std::chrono::steady_clock::now();
  numeric_time_ = std::chrono::duration<double>(end - start).count();

  return true;
}

bool SparseLDLT::solve(std::vector<double>& x) const {
  // Placeholder for Phase 5
  (void)x;
  return false;
}

std::size_t SparseLDLT::factor_nnz() const {
  if (!ok_) return 0;
  return l_idx_.size();
}

double SparseLDLT::fill_ratio() const {
  if (!ok_ || input_nnz_ == 0) return 0.0;
  return static_cast<double>(l_idx_.size()) / static_cast<double>(input_nnz_);
}

}  // namespace sovereign
