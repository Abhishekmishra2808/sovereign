#pragma once

#include <cstddef>
#include <vector>

namespace sovereign {

// Sparse LU with threshold partial pivoting: P A Q = L U for an n x n matrix
// given in compressed sparse column form. Columns are processed sparsest
// first and each pivot is the candidate with the fewest remaining row entries
// among those within `pivot_threshold` of the largest, so simplex bases (mostly
// slack columns) factor with little fill.
class SparseLU {
 public:
  bool factorize(std::size_t n, const std::vector<int>& col_ptr,
                 const std::vector<int>& row_idx, const std::vector<double>& values);
  bool solve(std::vector<double>& x) const;           // solves A x = b, b in x
  bool solve_transpose(std::vector<double>& x) const; // solves A^T x = b

  bool ok() const { return ok_; }
  std::size_t n() const { return n_; }
  std::size_t factor_nnz() const { return l_idx_.size() + u_idx_.size(); }

  double pivot_threshold = 0.1;

 private:
  std::size_t n_ = 0;
  bool ok_ = false;
  // L is unit lower triangular (diagonal stored first in each column), U is
  // upper triangular (diagonal stored last); indices are pivot positions.
  std::vector<int> l_ptr_, l_idx_;
  std::vector<double> l_val_;
  std::vector<int> u_ptr_, u_idx_;
  std::vector<double> u_val_;
  std::vector<int> pinv_;  // original row -> pivot position
  std::vector<int> q_;     // pivot position -> original column
};

}  // namespace sovereign
