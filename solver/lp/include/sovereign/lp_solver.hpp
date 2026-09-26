#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

class LpSolver {
 public:
  SolverResult solve(const OptimizationModel& model) const;
};

}  // namespace sovereign
