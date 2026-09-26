#pragma once

#include "sovereign/types.hpp"

#include <string>
#include <unordered_map>

namespace sovereign {

struct HeuristicResult {
  bool found = false;
  double objective = 0.0;
  std::unordered_map<std::string, double> primal;
  std::string method;
};

// Round integer vars and try to repair continuous vars via a bound-respecting projection.
HeuristicResult rounding_heuristic(const OptimizationModel& milp,
                                   const std::unordered_map<std::string, double>& lp_x,
                                   double int_tol = 1e-6);

// Dive by fixing the most fractional integer toward the nearer integer, re-solving LP.
HeuristicResult diving_heuristic(const OptimizationModel& milp,
                                 const std::unordered_map<std::string, double>& lp_x,
                                 int max_dives = 16,
                                 double int_tol = 1e-6);

}  // namespace sovereign
