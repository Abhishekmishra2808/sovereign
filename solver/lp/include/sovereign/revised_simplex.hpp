#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

struct RevisedSimplexOptions {
  int max_iterations = 100000;
  int refactor_every = 64;  // full dense refactor cadence; pivots use product-form etas
  bool enable_scaling = true;
  double feasibility_tol = 1e-8;
  double optimality_tol = 1e-8;
  double pivot_tol = 1e-10;
};

class RevisedSimplexSolver {
 public:
  explicit RevisedSimplexSolver(RevisedSimplexOptions options = {});

  SolverResult solve(const OptimizationModel& model) const;

 private:
  RevisedSimplexOptions options_;
};

}  // namespace sovereign
