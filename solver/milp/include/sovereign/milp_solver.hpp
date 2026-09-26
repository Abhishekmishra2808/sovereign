#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

class MilpSolver {
 public:
  SolverResult solve(const OptimizationModel& model) const;
};

}  // namespace sovereign
