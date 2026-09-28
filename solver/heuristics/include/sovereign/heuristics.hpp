#pragma once

#include "sovereign/dual_simplex.hpp"
#include "sovereign/types.hpp"

#include <string>
#include <unordered_map>

namespace sovereign {

struct HeuristicResult {
  bool found = false;
  double objective = 0.0;
  std::unordered_map<std::string, double> primal;
  std::string method;
  // LP relaxations solved while searching (diving effort accounting).
  int lp_solves = 0;
};

// Round integer vars and try to repair continuous vars via a bound-respecting projection.
HeuristicResult rounding_heuristic(const OptimizationModel& milp,
                                   const std::unordered_map<std::string, double>& lp_x,
                                   double int_tol = 1e-6);

// Coefficient diving: bound the fractional variable with the fewest locks in its
// preferred direction, re-solve, backtrack one step on infeasibility, and try
// rounding after every LP. `max_depth` <= 0 means one step per integer variable.
// `warm` (the basis that produced lp_x) lets each step re-solve with a few dual
// simplex pivots instead of from scratch. With `cutoff` set (the incumbent
// objective, in the model's sense) the dive stops once its LP bound can no
// longer beat it.
HeuristicResult diving_heuristic(const OptimizationModel& milp,
                                 const std::unordered_map<std::string, double>& lp_x,
                                 int max_depth = 0,
                                 double int_tol = 1e-6,
                                 const LpBasis* warm = nullptr,
                                 const double* cutoff = nullptr);

}  // namespace sovereign
