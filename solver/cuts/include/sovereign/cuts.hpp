#pragma once

#include "sovereign/types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace sovereign {

struct Cut {
  Constraint constraint;
  std::string source;  // "cover", "gomory"
};

// Generate simple knapsack cover cuts from a fractional binary LP solution.
std::vector<Cut> generate_cover_cuts(const OptimizationModel& milp,
                                     const std::unordered_map<std::string, double>& x,
                                     double int_tol = 1e-6,
                                     int max_cuts = 8);

// Mixed-integer rounding style cut from a single row with a fractional integer var.
std::vector<Cut> generate_mir_cuts(const OptimizationModel& milp,
                                   const std::unordered_map<std::string, double>& x,
                                   double int_tol = 1e-6,
                                   int max_cuts = 8);

}  // namespace sovereign
