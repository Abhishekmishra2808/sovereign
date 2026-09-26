#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

struct QpOptions {
  int max_iterations = 200;
  double feasibility_tol = 1e-8;
  double optimality_tol = 1e-8;
  double barrier_mu0 = 1.0;
  double barrier_reduction = 0.2;
  int barrier_steps = 12;
};

class ConvexQpSolver {
 public:
  explicit ConvexQpSolver(QpOptions options = {});
  SolverResult solve(const OptimizationModel& model) const;

 private:
  QpOptions options_;
};

}  // namespace sovereign
