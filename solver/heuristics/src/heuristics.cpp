#include "sovereign/heuristics.hpp"

#include "sovereign/revised_simplex.hpp"

#include <cmath>
#include <vector>

namespace sovereign {
namespace {

bool is_int_type(VariableType t) {
  return t == VariableType::Integer || t == VariableType::Binary;
}

double eval_obj(const OptimizationModel& m,
                const std::unordered_map<std::string, double>& x) {
  double obj = 0.0;
  for (const auto& kv : m.objective.linear) {
    auto it = x.find(kv.first);
    if (it != x.end()) obj += kv.second * it->second;
  }
  return obj;
}

bool satisfies_constraints(const OptimizationModel& m,
                           const std::unordered_map<std::string, double>& x,
                           double tol) {
  for (const auto& v : m.variables) {
    auto it = x.find(v.name);
    if (it == x.end()) return false;
    if (it->second < v.lower_bound - tol || it->second > v.upper_bound + tol) return false;
    if (is_int_type(v.type) && std::abs(it->second - std::round(it->second)) > tol) {
      return false;
    }
  }
  for (const auto& c : m.constraints) {
    double lhs = 0.0;
    for (const auto& kv : c.linear) {
      auto it = x.find(kv.first);
      if (it != x.end()) lhs += kv.second * it->second;
    }
    if (c.sense == ConstraintSense::Le && lhs > c.rhs + tol) return false;
    if (c.sense == ConstraintSense::Ge && lhs < c.rhs - tol) return false;
    if (c.sense == ConstraintSense::Eq && std::abs(lhs - c.rhs) > tol) return false;
  }
  return true;
}

OptimizationModel as_lp(const OptimizationModel& milp) {
  OptimizationModel lp = milp;
  lp.problem_type = ProblemType::LP;
  for (auto& v : lp.variables) {
    if (v.type == VariableType::Binary) {
      v.lower_bound = std::max(0.0, v.lower_bound);
      v.upper_bound = std::min(1.0, v.upper_bound);
    }
    v.type = VariableType::Continuous;
  }
  return lp;
}

}  // namespace

HeuristicResult rounding_heuristic(const OptimizationModel& milp,
                                   const std::unordered_map<std::string, double>& lp_x,
                                   double int_tol) {
  HeuristicResult hr;
  auto x = lp_x;
  for (const auto& v : milp.variables) {
    if (!is_int_type(v.type)) continue;
    auto it = x.find(v.name);
    if (it == x.end()) continue;
    double r = std::round(it->second);
    r = std::max(v.lower_bound, std::min(v.upper_bound, r));
    it->second = r;
  }
  // Continuous vars kept from LP; check feasibility
  if (!satisfies_constraints(milp, x, std::max(int_tol, 1e-6))) {
    return hr;
  }
  hr.found = true;
  hr.primal = x;
  hr.objective = eval_obj(milp, x);
  hr.method = "rounding";
  return hr;
}

HeuristicResult diving_heuristic(const OptimizationModel& milp,
                                 const std::unordered_map<std::string, double>& lp_x,
                                 int max_dives, double int_tol) {
  HeuristicResult hr;
  OptimizationModel node = milp;
  auto x = lp_x;

  for (int d = 0; d < max_dives; ++d) {
    // Find most fractional
    int best = -1;
    double best_frac = 0.0;
    for (std::size_t i = 0; i < node.variables.size(); ++i) {
      const auto& v = node.variables[i];
      if (!is_int_type(v.type)) continue;
      auto it = x.find(v.name);
      if (it == x.end()) continue;
      const double frac = std::abs(it->second - std::round(it->second));
      if (frac > int_tol && frac > best_frac) {
        best_frac = frac;
        best = static_cast<int>(i);
      }
    }
    if (best < 0) break;

    Variable& v = node.variables[static_cast<std::size_t>(best)];
    const double val = x[v.name];
    const double down = std::floor(val);
    const double up = std::ceil(val);
    // Fix toward nearer integer
    if (val - down <= up - val) {
      v.upper_bound = std::min(v.upper_bound, down);
      v.lower_bound = std::min(v.lower_bound, v.upper_bound);
    } else {
      v.lower_bound = std::max(v.lower_bound, up);
      v.upper_bound = std::max(v.upper_bound, v.lower_bound);
    }
    if (v.lower_bound > v.upper_bound + 1e-9) return hr;

    SolverResult lp = RevisedSimplexSolver().solve(as_lp(node));
    if (lp.status != SolverStatus::Optimal && lp.status != SolverStatus::Feasible) {
      return hr;
    }
    x = lp.primal;
  }

  // Final round
  HeuristicResult rounded = rounding_heuristic(node, x, int_tol);
  if (rounded.found && satisfies_constraints(milp, rounded.primal, 1e-5)) {
    rounded.method = "diving+rounding";
    // Recompute objective on original
    rounded.objective = eval_obj(milp, rounded.primal);
    return rounded;
  }
  if (satisfies_constraints(milp, x, 1e-5)) {
    hr.found = true;
    hr.primal = x;
    hr.objective = eval_obj(milp, x);
    hr.method = "diving";
  }
  return hr;
}

}  // namespace sovereign
