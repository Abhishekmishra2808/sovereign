#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

class QpSolver {
 public:
  SolverResult solve(const OptimizationModel& model) const;
};

}  // namespace sovereign
