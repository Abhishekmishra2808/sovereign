#include "sovereign/sparse_lu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace sovereign {

namespace {

// Pivots below this magnitude are treated as singular (matches DenseLU).
constexpr double kSingularPivot = 1e-14;

}  // namespace

bool SparseLU::factorize(std::size_t n, const std::vector<int>& col_ptr,
                         const std::vector<int>& row_idx, const std::vector<double>& values) {
  n_ = n;
  ok_ = false;
  if (col_ptr.size() != n + 1) return false;
  const int nn = static_cast<int>(n);

  q_.resize(n);
  std::iota(q_.begin(), q_.end(), 0);
  std::stable_sort(q_.begin(), q_.end(), [&](int a, int b) {
    return col_ptr[static_cast<std::size_t>(a) + 1] - col_ptr[static_cast<std::size_t>(a)] <
           col_ptr[static_cast<std::size_t>(b) + 1] - col_ptr[static_cast<std::size_t>(b)];
  });

  // Entries each row still has in columns not yet factorized.
  std::vector<int> row_count(n, 0);
  for (int p = 0; p < col_ptr[n]; ++p) {
    const int i = row_idx[static_cast<std::size_t>(p)];
    if (i < 0 || i >= nn) return false;
    ++row_count[static_cast<std::size_t>(i)];
  }

  pinv_.assign(n, -1);
  l_ptr_.assign(1, 0);
  u_ptr_.assign(1, 0);
  l_idx_.clear();
  l_val_.clear();
  u_idx_.clear();
  u_val_.clear();
  l_ptr_.reserve(n + 1);
  u_ptr_.reserve(n + 1);
  l_idx_.reserve(static_cast<std::size_t>(col_ptr[n]) + n);
  l_val_.reserve(static_cast<std::size_t>(col_ptr[n]) + n);
  u_idx_.reserve(static_cast<std::size_t>(col_ptr[n]) + n);
  u_val_.reserve(static_cast<std::size_t>(col_ptr[n]) + n);

  std::vector<double> x(n, 0.0);
  std::vector<int> xi(n), stack(n), pstack(n), mark(n, -1);

  // Depth-first search over the graph of L from row j; appends the rows it
  // reaches to xi[top..n) in topological order and returns the new top.
  auto dfs = [&](int j, int top, int stamp) {
    int head = 0;
    stack[0] = j;
    while (head >= 0) {
      j = stack[static_cast<std::size_t>(head)];
      const int jcol = pinv_[static_cast<std::size_t>(j)];
      if (mark[static_cast<std::size_t>(j)] != stamp) {
        mark[static_cast<std::size_t>(j)] = stamp;
        pstack[static_cast<std::size_t>(head)] = jcol < 0 ? 0 : l_ptr_[static_cast<std::size_t>(jcol)] + 1;
      }
      bool done = true;
      const int p_end = jcol < 0 ? 0 : l_ptr_[static_cast<std::size_t>(jcol) + 1];
      for (int p = pstack[static_cast<std::size_t>(head)]; p < p_end; ++p) {
        const int i = l_idx_[static_cast<std::size_t>(p)];
        if (mark[static_cast<std::size_t>(i)] == stamp) continue;
        pstack[static_cast<std::size_t>(head)] = p + 1;
        stack[static_cast<std::size_t>(++head)] = i;
        done = false;
        break;
      }
      if (done) {
        --head;
        xi[static_cast<std::size_t>(--top)] = j;
      }
    }
    return top;
  };

  for (int k = 0; k < nn; ++k) {
    const int col = q_[static_cast<std::size_t>(k)];
    const int c0 = col_ptr[static_cast<std::size_t>(col)];
    const int c1 = col_ptr[static_cast<std::size_t>(col) + 1];

    // x = L \ A(:, col), touching only the rows reachable from its pattern.
    int top = nn;
    for (int p = c0; p < c1; ++p) {
      const int i = row_idx[static_cast<std::size_t>(p)];
      if (mark[static_cast<std::size_t>(i)] != k) top = dfs(i, top, k);
    }
    for (int px = top; px < nn; ++px) x[static_cast<std::size_t>(xi[static_cast<std::size_t>(px)])] = 0.0;
    for (int p = c0; p < c1; ++p) {
      x[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(p)])] += values[static_cast<std::size_t>(p)];
    }
    for (int px = top; px < nn; ++px) {
      const int j = xi[static_cast<std::size_t>(px)];
      const int jcol = pinv_[static_cast<std::size_t>(j)];
      if (jcol < 0) continue;
      const double xj = x[static_cast<std::size_t>(j)];
      if (xj == 0.0) continue;
      for (int p = l_ptr_[static_cast<std::size_t>(jcol)] + 1; p < l_ptr_[static_cast<std::size_t>(jcol) + 1]; ++p) {
        x[static_cast<std::size_t>(l_idx_[static_cast<std::size_t>(p)])] -= l_val_[static_cast<std::size_t>(p)] * xj;
      }
    }

    // Rows already pivoted form U(:, k); the rest are pivot candidates.
    double amax = 0.0;
    for (int px = top; px < nn; ++px) {
      const int i = xi[static_cast<std::size_t>(px)];
      const double v = x[static_cast<std::size_t>(i)];
      if (pinv_[static_cast<std::size_t>(i)] < 0) {
        amax = std::max(amax, std::abs(v));
      } else if (v != 0.0) {
        u_idx_.push_back(pinv_[static_cast<std::size_t>(i)]);
        u_val_.push_back(v);
      }
    }
    if (!(amax >= kSingularPivot)) return false;

    int ipiv = -1;
    int best_count = std::numeric_limits<int>::max();
    double best_abs = 0.0;
    for (int px = top; px < nn; ++px) {
      const int i = xi[static_cast<std::size_t>(px)];
      if (pinv_[static_cast<std::size_t>(i)] >= 0) continue;
      const double a = std::abs(x[static_cast<std::size_t>(i)]);
      if (a < pivot_threshold * amax) continue;
      const int cnt = row_count[static_cast<std::size_t>(i)];
      if (cnt < best_count || (cnt == best_count && a > best_abs)) {
        ipiv = i;
        best_count = cnt;
        best_abs = a;
      }
    }

    const double pivot = x[static_cast<std::size_t>(ipiv)];
    u_idx_.push_back(k);
    u_val_.push_back(pivot);
    u_ptr_.push_back(static_cast<int>(u_idx_.size()));

    pinv_[static_cast<std::size_t>(ipiv)] = k;
    l_idx_.push_back(ipiv);
    l_val_.push_back(1.0);
    for (int px = top; px < nn; ++px) {
      const int i = xi[static_cast<std::size_t>(px)];
      if (pinv_[static_cast<std::size_t>(i)] >= 0) continue;
      const double v = x[static_cast<std::size_t>(i)];
      if (v == 0.0) continue;
      l_idx_.push_back(i);
      l_val_.push_back(v / pivot);
    }
    l_ptr_.push_back(static_cast<int>(l_idx_.size()));

    for (int p = c0; p < c1; ++p) --row_count[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(p)])];
    for (int px = top; px < nn; ++px) x[static_cast<std::size_t>(xi[static_cast<std::size_t>(px)])] = 0.0;
  }

  for (int& i : l_idx_) i = pinv_[static_cast<std::size_t>(i)];
  ok_ = true;
  return true;
}

bool SparseLU::solve(std::vector<double>& x) const {
  if (!ok_ || x.size() != n_) return false;
  const int nn = static_cast<int>(n_);
  std::vector<double> b(n_);
  for (std::size_t i = 0; i < n_; ++i) b[static_cast<std::size_t>(pinv_[i])] = x[i];
  for (int k = 0; k < nn; ++k) {
    const double bk = b[static_cast<std::size_t>(k)];
    if (bk == 0.0) continue;
    for (int p = l_ptr_[static_cast<std::size_t>(k)] + 1; p < l_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
      b[static_cast<std::size_t>(l_idx_[static_cast<std::size_t>(p)])] -= l_val_[static_cast<std::size_t>(p)] * bk;
    }
  }
  for (int k = nn - 1; k >= 0; --k) {
    const int diag = u_ptr_[static_cast<std::size_t>(k) + 1] - 1;
    const double bk = b[static_cast<std::size_t>(k)] / u_val_[static_cast<std::size_t>(diag)];
    b[static_cast<std::size_t>(k)] = bk;
    if (bk == 0.0) continue;
    for (int p = u_ptr_[static_cast<std::size_t>(k)]; p < diag; ++p) {
      b[static_cast<std::size_t>(u_idx_[static_cast<std::size_t>(p)])] -= u_val_[static_cast<std::size_t>(p)] * bk;
    }
  }
  for (std::size_t k = 0; k < n_; ++k) x[static_cast<std::size_t>(q_[k])] = b[k];
  return true;
}

bool SparseLU::solve_transpose(std::vector<double>& x) const {
  if (!ok_ || x.size() != n_) return false;
  const int nn = static_cast<int>(n_);
  std::vector<double> b(n_);
  for (std::size_t k = 0; k < n_; ++k) b[k] = x[static_cast<std::size_t>(q_[k])];
  for (int k = 0; k < nn; ++k) {
    const int diag = u_ptr_[static_cast<std::size_t>(k) + 1] - 1;
    double s = b[static_cast<std::size_t>(k)];
    for (int p = u_ptr_[static_cast<std::size_t>(k)]; p < diag; ++p) {
      s -= u_val_[static_cast<std::size_t>(p)] * b[static_cast<std::size_t>(u_idx_[static_cast<std::size_t>(p)])];
    }
    b[static_cast<std::size_t>(k)] = s / u_val_[static_cast<std::size_t>(diag)];
  }
  for (int k = nn - 1; k >= 0; --k) {
    double s = b[static_cast<std::size_t>(k)];
    for (int p = l_ptr_[static_cast<std::size_t>(k)] + 1; p < l_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
      s -= l_val_[static_cast<std::size_t>(p)] * b[static_cast<std::size_t>(l_idx_[static_cast<std::size_t>(p)])];
    }
    b[static_cast<std::size_t>(k)] = s;
  }
  for (std::size_t i = 0; i < n_; ++i) x[i] = b[static_cast<std::size_t>(pinv_[i])];
  return true;
}

}  // namespace sovereign
