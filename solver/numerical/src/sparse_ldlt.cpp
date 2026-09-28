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
  // Placeholder for Phase 4
  (void)values;
  (void)regularization;
  return false;
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
