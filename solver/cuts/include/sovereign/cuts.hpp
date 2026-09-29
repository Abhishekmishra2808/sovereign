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

// Chvatal-Gomory style rounding cut from a single row with a fractional integer var.
// Rows that cannot be rounded soundly (a column that may go negative while its
// coefficient is fractional) are skipped rather than emitted invalidly.
std::vector<Cut> generate_mir_cuts(const OptimizationModel& milp,
                                   const std::unordered_map<std::string, double>& x,
                                   double int_tol = 1e-6,
                                   int max_cuts = 8);

// Complemented MIR cuts from single rows (any sense), over integer and
// continuous variables with at least one finite bound. Most efficacious first.
// Only the first `row_count` constraints are used as base rows.
std::vector<Cut> generate_cmir_cuts(const OptimizationModel& milp,
                                    const std::unordered_map<std::string, double>& x,
                                    double int_tol = 1e-6,
                                    int max_cuts = 8,
                                    std::size_t row_count = static_cast<std::size_t>(-1));

// Independent safety net for ANY cut, including ones added by hand.
//
// A cut is only valid if it does not remove a point we already know is
// integer-feasible. `reference_points` is every such point the search has
// accumulated (incumbents, heuristic successes). Generators can be wrong; this
// check is written independently of them, so a generator bug degrades into
// "cut rejected" rather than "invalid MILP answer".
//
// Returns the rejection reason, or an empty string if the cut is accepted.
std::string check_cut_validity(
    const OptimizationModel& milp, const Constraint& cut,
    const std::vector<std::unordered_map<std::string, double>>& reference_points,
    double tol = 1e-6);

}  // namespace sovereign
