#include "sovereign/sparse_ldlt.hpp"

#include "sovereign/amd_ordering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

// Up-looking LDL^T driven by the elimination tree, following T. A. Davis,
// "Algorithm 849: A concise sparse Cholesky factorization package",
// ACM TOMS 31(4), 2005. Row k of L is the set of etree paths from the
// nonzeros of column k of the permuted upper triangle, so symbolic work is
// O(nnz(L)) and numeric work is proportional to the flops of the factorization.

namespace sovereign {

bool SparseLDLT::symbolic_analyze(const SparseSymmetricPattern& pattern, std::size_t max_factor_nnz) {
  auto start = std::chrono::steady_clock::now();
  ok_ = false;
  factored_ = false;
  fill_limit_exceeded_ = false;

  if (!pattern.is_valid()) return false;

  n_ = pattern.n;
  input_nnz_ = pattern.nnz();
  const int n = static_cast<int>(n_);

  perm_.clear();
  iperm_.clear();
  if (n > 0) {
    AMDOrdering amd;
    amd.set_fill_limit(max_factor_nnz);
    if (!amd.compute(pattern, perm_, iperm_)) {
      fill_limit_exceeded_ = amd.fill_limit_exceeded();
      return false;
    }
  }

  // Permuted upper triangle: input entry (i, j), i >= j, becomes
  // (min(pi, pj), max(pi, pj)) of P M P^T.
  c_ptr_.assign(n_ + 1, 0);
  for (int j = 0; j < n; ++j) {
    for (int p = pattern.col_ptr[static_cast<std::size_t>(j)]; p < pattern.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const int pi = iperm_[static_cast<std::size_t>(pattern.row_idx[static_cast<std::size_t>(p)])];
      const int pj = iperm_[static_cast<std::size_t>(j)];
      ++c_ptr_[static_cast<std::size_t>(std::max(pi, pj)) + 1];
    }
  }
  for (int k = 0; k < n; ++k) c_ptr_[static_cast<std::size_t>(k) + 1] += c_ptr_[static_cast<std::size_t>(k)];
  c_idx_.resize(input_nnz_);
  c_map_.resize(input_nnz_);
  c_val_.assign(input_nnz_, 0.0);
  std::vector<int> next(c_ptr_.begin(), c_ptr_.end() - 1);
  for (int j = 0; j < n; ++j) {
    for (int p = pattern.col_ptr[static_cast<std::size_t>(j)]; p < pattern.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const int pi = iperm_[static_cast<std::size_t>(pattern.row_idx[static_cast<std::size_t>(p)])];
      const int pj = iperm_[static_cast<std::size_t>(j)];
      const int q = next[static_cast<std::size_t>(std::max(pi, pj))]++;
      c_idx_[static_cast<std::size_t>(q)] = std::min(pi, pj);
      c_map_[static_cast<std::size_t>(p)] = q;
    }
  }

  // Elimination tree and column counts of L.
  parent_.assign(n_, -1);
  flag_.assign(n_, -1);
  l_len_.assign(n_, 0);
  for (int k = 0; k < n; ++k) {
    flag_[static_cast<std::size_t>(k)] = k;
    for (int p = c_ptr_[static_cast<std::size_t>(k)]; p < c_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
      for (int i = c_idx_[static_cast<std::size_t>(p)]; i < k && flag_[static_cast<std::size_t>(i)] != k;
           i = parent_[static_cast<std::size_t>(i)]) {
        if (parent_[static_cast<std::size_t>(i)] == -1) parent_[static_cast<std::size_t>(i)] = k;
        ++l_len_[static_cast<std::size_t>(i)];
        flag_[static_cast<std::size_t>(i)] = k;
      }
    }
  }
  l_ptr_.assign(n_ + 1, 0);
  for (int k = 0; k < n; ++k) {
    l_ptr_[static_cast<std::size_t>(k) + 1] = l_ptr_[static_cast<std::size_t>(k)] + l_len_[static_cast<std::size_t>(k)];
  }
  l_idx_.assign(static_cast<std::size_t>(l_ptr_[n_]), 0);
  l_val_.assign(l_idx_.size(), 0.0);
  d_.assign(n_, 0.0);
  y_.assign(n_, 0.0);
  pattern_.assign(n_, 0);

  ok_ = true;
  symbolic_time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return true;
}

bool SparseLDLT::numeric_factor(const std::vector<double>& values, double regularization) {
  auto start = std::chrono::steady_clock::now();
  factored_ = false;
  if (!ok_ || values.size() != input_nnz_) return false;
  if (!pivot_sign_.empty() && pivot_sign_.size() != n_) return false;

  const int n = static_cast<int>(n_);
  regularization_used_ = regularization;
  if (n == 0) {
    min_pivot_ = 0.0;
    max_pivot_ = 0.0;
    numeric_time_ = 0.0;
    factored_ = true;
    return true;
  }

  std::fill(c_val_.begin(), c_val_.end(), 0.0);
  for (std::size_t p = 0; p < values.size(); ++p) c_val_[static_cast<std::size_t>(c_map_[p])] += values[p];

  min_pivot_ = std::numeric_limits<double>::infinity();
  max_pivot_ = -std::numeric_limits<double>::infinity();
  replaced_pivots_ = 0;
  std::fill(flag_.begin(), flag_.end(), -1);
  std::fill(l_len_.begin(), l_len_.end(), 0);

  for (int k = 0; k < n; ++k) {
    // Scatter column k of the upper triangle and find the pattern of row k of L.
    y_[static_cast<std::size_t>(k)] = 0.0;
    int top = n;
    flag_[static_cast<std::size_t>(k)] = k;
    for (int p = c_ptr_[static_cast<std::size_t>(k)]; p < c_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
      int i = c_idx_[static_cast<std::size_t>(p)];
      y_[static_cast<std::size_t>(i)] += c_val_[static_cast<std::size_t>(p)];
      int len = 0;
      for (; flag_[static_cast<std::size_t>(i)] != k; i = parent_[static_cast<std::size_t>(i)]) {
        pattern_[static_cast<std::size_t>(len++)] = i;
        flag_[static_cast<std::size_t>(i)] = k;
      }
      while (len > 0) pattern_[static_cast<std::size_t>(--top)] = pattern_[static_cast<std::size_t>(--len)];
    }

    const double diagonal = y_[static_cast<std::size_t>(k)] + regularization;
    double d_k = diagonal;
    y_[static_cast<std::size_t>(k)] = 0.0;
    for (; top < n; ++top) {
      const int i = pattern_[static_cast<std::size_t>(top)];
      const double yi = y_[static_cast<std::size_t>(i)];
      y_[static_cast<std::size_t>(i)] = 0.0;
      const int p_begin = l_ptr_[static_cast<std::size_t>(i)];
      const int p_end = p_begin + l_len_[static_cast<std::size_t>(i)];
      for (int p = p_begin; p < p_end; ++p) {
        y_[static_cast<std::size_t>(l_idx_[static_cast<std::size_t>(p)])] -= l_val_[static_cast<std::size_t>(p)] * yi;
      }
      const double l_ki = yi / d_[static_cast<std::size_t>(i)];
      d_k -= l_ki * yi;
      l_idx_[static_cast<std::size_t>(p_end)] = k;
      l_val_[static_cast<std::size_t>(p_end)] = l_ki;
      ++l_len_[static_cast<std::size_t>(i)];
    }

    const double sign =
        pivot_sign_.empty() ? 1.0 : static_cast<double>(pivot_sign_[static_cast<std::size_t>(perm_[static_cast<std::size_t>(k)])]);
    const bool replace = tiny_pivot_threshold_ > 0.0 && std::isfinite(d_k) &&
                         sign * d_k <= tiny_pivot_threshold_ * std::abs(diagonal);
    if (replace) {
      d_[static_cast<std::size_t>(k)] = sign * 1e128;
      ++replaced_pivots_;
      continue;
    }
    if (!std::isfinite(d_k) || sign * d_k < 1e-30) {
      // Wrong inertia (even after regularization) or numerical breakdown.
      std::fill(y_.begin(), y_.end(), 0.0);
      return false;
    }
    d_[static_cast<std::size_t>(k)] = d_k;
    min_pivot_ = std::min(min_pivot_, sign * d_k);
    max_pivot_ = std::max(max_pivot_, sign * d_k);
  }

  factored_ = true;
  numeric_time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return true;
}

bool SparseLDLT::solve(std::vector<double>& x) const {
  auto start = std::chrono::steady_clock::now();
  if (!factored_ || x.size() != n_) return false;
  const int n = static_cast<int>(n_);
  if (n == 0) {
    solve_time_ = 0.0;
    return true;
  }

  // y = P b, then L z = y, D w = z, L^T v = w, x = P^T v.
  work_.resize(n_);
  std::vector<double>& y = work_;
  for (int k = 0; k < n; ++k) y[static_cast<std::size_t>(k)] = x[static_cast<std::size_t>(perm_[static_cast<std::size_t>(k)])];

  for (int j = 0; j < n; ++j) {
    const double yj = y[static_cast<std::size_t>(j)];
    if (yj == 0.0) continue;
    for (int p = l_ptr_[static_cast<std::size_t>(j)]; p < l_ptr_[static_cast<std::size_t>(j) + 1]; ++p) {
      y[static_cast<std::size_t>(l_idx_[static_cast<std::size_t>(p)])] -= l_val_[static_cast<std::size_t>(p)] * yj;
    }
  }
  for (int k = 0; k < n; ++k) y[static_cast<std::size_t>(k)] /= d_[static_cast<std::size_t>(k)];
  for (int j = n - 1; j >= 0; --j) {
    double sum = y[static_cast<std::size_t>(j)];
    for (int p = l_ptr_[static_cast<std::size_t>(j)]; p < l_ptr_[static_cast<std::size_t>(j) + 1]; ++p) {
      sum -= l_val_[static_cast<std::size_t>(p)] * y[static_cast<std::size_t>(l_idx_[static_cast<std::size_t>(p)])];
    }
    y[static_cast<std::size_t>(j)] = sum;
  }

  for (int k = 0; k < n; ++k) x[static_cast<std::size_t>(perm_[static_cast<std::size_t>(k)])] = y[static_cast<std::size_t>(k)];

  solve_time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return true;
}

std::size_t SparseLDLT::factor_nnz() const {
  if (!ok_) return 0;
  return l_idx_.size();
}

double SparseLDLT::factor_flops() const {
  if (!ok_) return 0.0;
  double flops = 0.0;
  for (std::size_t k = 0; k < n_; ++k) {
    const double count = static_cast<double>(l_ptr_[k + 1] - l_ptr_[k]);
    flops += count * count;
  }
  return flops;
}

double SparseLDLT::fill_ratio() const {
  if (!ok_ || input_nnz_ == 0) return 0.0;
  return static_cast<double>(l_idx_.size()) / static_cast<double>(input_nnz_);
}

}  // namespace sovereign
