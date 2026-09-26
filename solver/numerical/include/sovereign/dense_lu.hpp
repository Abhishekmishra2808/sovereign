#pragma once

#include <cstddef>
#include <vector>

namespace sovereign {

// Dense LU with partial pivoting for an n x n column-major matrix.
class DenseLU {
 public:
  bool factorize(std::vector<double> a_col_major, std::size_t n);
  bool solve(std::vector<double>& x) const;           // solves A x = b, b in x
  bool solve_transpose(std::vector<double>& x) const; // solves A^T x = b

  bool ok() const { return ok_; }
  std::size_t n() const { return n_; }

 private:
  std::size_t n_ = 0;
  bool ok_ = false;
  std::vector<double> lu_;
  std::vector<int> piv_;
};

}  // namespace sovereign
