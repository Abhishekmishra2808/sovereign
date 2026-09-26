#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

class OptimizationEngine {
 public:
  SolverResult solve(const OptimizationModel& model) const;
};

}  // namespace sovereign
