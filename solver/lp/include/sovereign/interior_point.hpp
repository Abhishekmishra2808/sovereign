#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

struct InteriorPointOptions {
  // Budget is sized for the *correct* termination test. The previous test
  // divided the complementarity sum by n, which made it n times too lenient and
  // let the loop exit early on wide sparse models; with the real relative
  // duality gap the method legitimately needs more iterations, so the default
  // is raised accordingly.
  int max_iterations = 250;
  double feasibility_tol = 1e-9;
  double optimality_tol = 1e-9;
  double fraction_to_boundary = 0.999;
  bool enable_scaling = true;

  // Normal equations M = A D A^T. When true, always use the sparse LDL^T
  // (a numerical breakdown is retried with dense LU while it fits). When
  // false, SOVEREIGN_IPM_NORMAL_EQUATIONS decides (auto by default, which
  // picks sparse or dense per model).
  bool use_sparse_normal_equations = false;
};

// Mehrotra predictor-corrector primal-dual interior-point method for LP.
// min c'x  s.t. Ax = b, x >= 0  (model converted with slacks; infeasible-start).
class InteriorPointSolver {
 public:
  explicit InteriorPointSolver(InteriorPointOptions options = {});

  SolverResult solve(const OptimizationModel& model) const;

 private:
  InteriorPointOptions options_;
};

}  // namespace sovereign
