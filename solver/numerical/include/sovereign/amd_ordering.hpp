#pragma once

#include "sovereign/sparse_symmetric.hpp"

#include <vector>

namespace sovereign {

// Approximate Minimum Degree ordering for sparse symmetric matrices
//
// Computes a fill-reducing permutation P such that the factorization of
// P M P^T has fewer nonzeros than the factorization of M.
//
// Algorithm: Greedy minimum approximate degree (Amestoy-Davis-Duff variant)
// - Iteratively eliminate vertices with minimum approximate degree
// - Use aggressive absorption to reduce graph size
// - Mass elimination for indistinguishable vertices
//
// Returns:
//   perm[k] = original column that is k-th in elimination order
//   iperm[j] = position in elimination order of original column j
//
// Cost: Depends on graph structure. Measure ordering_time empirically.
class AMDOrdering {
 public:
  // Compute ordering from symmetric pattern (lower triangle only)
  // Returns true if successful, false on error
  //
  // Output:
  //   perm: size n, perm[k] = original column for position k
  //   iperm: size n, iperm[j] = position of original column j
  bool compute(
      const SparseSymmetricPattern& pattern,
      std::vector<int>& perm,
      std::vector<int>& iperm
  );

  // Timing statistics
  double ordering_time_seconds() const { return ordering_time_; }

 private:
  double ordering_time_ = 0.0;
};

}  // namespace sovereign
