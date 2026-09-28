#include "sovereign/dense_lu.hpp"
#include "sovereign/gpu_spmv.hpp"

#include <algorithm>
#include <cmath>

namespace sovereign {

bool DenseLU::factorize(std::vector<double> a_col_major, std::size_t n) {
  n_ = n;
  ok_ = false;
  if (a_col_major.size() != n * n) return false;
  lu_ = std::move(a_col_major);
  bool nonsingular = false;
  if (gpu_dense_lu(n, lu_, piv_, nonsingular)) {
    ok_ = nonsingular;
    return ok_;
  }
  piv_.resize(n);
  for (std::size_t i = 0; i < n; ++i) piv_[i] = static_cast<int>(i);

  for (std::size_t k = 0; k < n; ++k) {
    std::size_t pivot_row = k;
    double best = std::abs(lu_[k * n + k]);
    for (std::size_t i = k + 1; i < n; ++i) {
      const double v = std::abs(lu_[k * n + i]);
      if (v > best) {
        best = v;
        pivot_row = i;
      }
    }
    if (best < 1e-14) {
      return false;
    }
    if (pivot_row != k) {
      for (std::size_t j = 0; j < n; ++j) {
        std::swap(lu_[j * n + k], lu_[j * n + pivot_row]);
      }
      std::swap(piv_[k], piv_[pivot_row]);
    }
    const double akk = lu_[k * n + k];
    for (std::size_t i = k + 1; i < n; ++i) {
      lu_[k * n + i] /= akk;
      const double lik = lu_[k * n + i];
      for (std::size_t j = k + 1; j < n; ++j) {
        lu_[j * n + i] -= lik * lu_[j * n + k];
      }
    }
  }
  ok_ = true;
  return true;
}

bool DenseLU::solve(std::vector<double>& x) const {
  if (!ok_ || x.size() != n_) return false;
  // Apply pivot permutation in-place via temp only for permuted RHS
  std::vector<double> b(n_);
  for (std::size_t i = 0; i < n_; ++i) {
    b[i] = x[static_cast<std::size_t>(piv_[i])];
  }
  // Forward subst Ly = Pb
  for (std::size_t i = 0; i < n_; ++i) {
    double sum = b[i];
    for (std::size_t j = 0; j < i; ++j) {
      sum -= lu_[j * n_ + i] * b[j];
    }
    b[i] = sum;
  }
  // Back subst Ux = y
  for (std::size_t ii = 0; ii < n_; ++ii) {
    const std::size_t i = n_ - 1 - ii;
    double sum = b[i];
    for (std::size_t j = i + 1; j < n_; ++j) {
      sum -= lu_[j * n_ + i] * b[j];
    }
    b[i] = sum / lu_[i * n_ + i];
  }
  x.swap(b);
  return true;
}

bool DenseLU::solve_transpose(std::vector<double>& x) const {
  if (!ok_ || x.size() != n_) return false;
  // Solve U^T y = x
  for (std::size_t i = 0; i < n_; ++i) {
    double sum = x[i];
    for (std::size_t j = 0; j < i; ++j) {
      sum -= lu_[i * n_ + j] * x[j];
    }
    x[i] = sum / lu_[i * n_ + i];
  }
  // Solve L^T z = y
  for (std::size_t ii = 0; ii < n_; ++ii) {
    const std::size_t i = n_ - 1 - ii;
    double sum = x[i];
    for (std::size_t j = i + 1; j < n_; ++j) {
      sum -= lu_[i * n_ + j] * x[j];
    }
    x[i] = sum;
  }
  // Apply P^T
  std::vector<double> z(n_, 0.0);
  for (std::size_t i = 0; i < n_; ++i) {
    z[static_cast<std::size_t>(piv_[i])] = x[i];
  }
  x.swap(z);
  return true;
}

}  // namespace sovereign
