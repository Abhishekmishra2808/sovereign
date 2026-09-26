#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

struct QpInteriorPointOptions {
  int max_iterations = 100;
  double feasibility_tol = 1e-8;
  double optimality_tol = 1e-8;
  double fraction_to_boundary = 0.999;
  bool enable_scaling = false;  // keep false: Q scaling is fiddly; small QPs OK
};

// Mehrotra predictor-corrector for convex QP:
//   min  1/2 x' Q x + c' x   s.t. Ax = b, x >= 0
// Q is added to the (1,1) KKT block (Q + X^{-1}S). Falls back is handled by
// the caller (Frank–Wolfe remains available as a labeled earlier approach).
class QpInteriorPointSolver {
 public:
  explicit QpInteriorPointSolver(QpInteriorPointOptions options = {});
  SolverResult solve(const OptimizationModel& model) const;

 private:
  QpInteriorPointOptions options_;
};

}  // namespace sovereign
