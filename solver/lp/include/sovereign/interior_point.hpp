#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

struct InteriorPointOptions {
  int max_iterations = 100;
  double feasibility_tol = 1e-8;
  double optimality_tol = 1e-8;
  double fraction_to_boundary = 0.999;
  bool enable_scaling = true;
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
