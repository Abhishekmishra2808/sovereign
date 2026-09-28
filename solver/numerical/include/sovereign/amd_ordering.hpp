#pragma once

#include "sovereign/sparse_symmetric.hpp"

#include <cstddef>
#include <vector>

namespace sovereign {

// Approximate Minimum Degree ordering for sparse symmetric matrices
//
// Computes a fill-reducing permutation P such that the factorization of
// P M P^T has fewer nonzeros than the factorization of M.
//
// Algorithm: greedy minimum degree on the explicit elimination graph
// (ties broken by lowest index). Degrees are exact, so the sum of degrees at
// elimination equals nnz(L) for the resulting order. Quotient-graph AMD with
// aggressive absorption and mass elimination can replace this later.
//
// Returns:
//   perm[k] = original column that is k-th in elimination order
//   iperm[j] = position in elimination order of original column j
class AMDOrdering {
 public:
  // Compute ordering from symmetric pattern (lower triangle only)
  // Returns false on error, or when the factor would exceed the fill limit.
  //
  // Output:
  //   perm: size n, perm[k] = original column for position k
  //   iperm: size n, iperm[j] = position of original column j
  bool compute(
      const SparseSymmetricPattern& pattern,
      std::vector<int>& perm,
      std::vector<int>& iperm
  );

  // Give up once nnz(L) (strictly lower part) would exceed this; 0 = no limit.
  void set_fill_limit(std::size_t limit) { fill_limit_ = limit; }
  bool fill_limit_exceeded() const { return fill_limit_exceeded_; }
  std::size_t predicted_factor_nnz() const { return predicted_factor_nnz_; }

  // Timing statistics
  double ordering_time_seconds() const { return ordering_time_; }

 private:
  double ordering_time_ = 0.0;
  std::size_t fill_limit_ = 0;
  bool fill_limit_exceeded_ = false;
  std::size_t predicted_factor_nnz_ = 0;
};

}  // namespace sovereign
