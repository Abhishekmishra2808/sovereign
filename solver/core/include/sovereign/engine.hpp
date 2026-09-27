#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

struct EngineOptions {
  std::string lp_algorithm;
  bool presolve = true;
};

class OptimizationEngine {
 public:
  SolverResult solve(const OptimizationModel& model, const EngineOptions& options = {}) const;
};

}  // namespace sovereign
