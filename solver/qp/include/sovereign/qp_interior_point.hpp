#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

struct QpInteriorPointOptions {
  int max_iterations = 100;
  double feasibility_tol = 1e-8;
  double optimality_tol = 1e-8;
  double fraction_to_boundary = 0.999;
  bool enable_scaling = false;  // keep false: Q scaling is fiddly; small QPs OK
  // Robustness features, on by default; switchable for the ablation in
  // benchmarks/reports/ROBUSTNESS.md (SOVEREIGN_QP_DISABLE).
  bool iterative_refinement = true;  // refine each regularized KKT solve
  bool common_step = true;           // equal primal/dual step when Q != 0
  bool retry_mehrotra_start = true;  // second attempt from Mehrotra's start
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
