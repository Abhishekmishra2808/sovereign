#include "sovereign/heuristics.hpp"

#include "sovereign/revised_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace sovereign {
namespace {

bool is_int_type(VariableType t) {
  return t == VariableType::Integer || t == VariableType::Binary;
}

double eval_obj(const OptimizationModel& m,
                const std::unordered_map<std::string, double>& x) {
  double obj = m.objective.constant;
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

// How many constraints could be violated by rounding a variable down / up.
struct Locks {
  int down = 0;
  int up = 0;
};

std::unordered_map<std::string, Locks> variable_locks(const OptimizationModel& m) {
  std::unordered_map<std::string, Locks> locks;
  for (const auto& c : m.constraints) {
    for (const auto& kv : c.linear) {
      if (kv.second == 0.0) continue;
      Locks& l = locks[kv.first];
      const bool pos = kv.second > 0.0;
      if (c.sense != ConstraintSense::Ge) ++(pos ? l.up : l.down);
      if (c.sense != ConstraintSense::Le) ++(pos ? l.down : l.up);
    }
  }
  return locks;
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
  const double tol = std::max(int_tol, 1e-6);
  auto round_with = [&](bool use_locks, const std::unordered_map<std::string, Locks>& locks) {
    auto x = lp_x;
    for (const auto& v : milp.variables) {
      if (!is_int_type(v.type)) continue;
      auto it = x.find(v.name);
      if (it == x.end()) continue;
      double r = std::round(it->second);
      if (use_locks && std::abs(it->second - r) > int_tol) {
        auto lk = locks.find(v.name);
        const Locks l = lk == locks.end() ? Locks{} : lk->second;
        if (l.down == 0) r = std::floor(it->second);
        else if (l.up == 0) r = std::ceil(it->second);
      }
      it->second = std::max(v.lower_bound, std::min(v.upper_bound, r));
    }
    return x;
  };

  // Nearest rounding first; then round every variable in a direction no
  // constraint objects to, which is what covering (>=) rows need.
  auto x = round_with(false, {});
  if (satisfies_constraints(milp, x, tol)) {
    hr.method = "rounding";
  } else {
    x = round_with(true, variable_locks(milp));
    if (!satisfies_constraints(milp, x, tol)) return hr;
    hr.method = "lock rounding";
  }
  hr.found = true;
  hr.primal = x;
  hr.objective = eval_obj(milp, x);
  return hr;
}

HeuristicResult diving_heuristic(const OptimizationModel& milp,
                                 const std::unordered_map<std::string, double>& lp_x,
                                 int max_depth, double int_tol, const LpBasis* warm,
                                 const double* cutoff) {
  // Coefficient diving: repeatedly bound the fractional variable that has the
  // fewest locks in its preferred direction (the move least likely to make the
  // LP infeasible), re-solve warm, and backtrack one step if it does.
  HeuristicResult hr;
  const auto locks = variable_locks(milp);
  OptimizationModel node = milp;
  // Kept in step with `node`'s bounds so each re-solve skips a model copy.
  OptimizationModel relaxed = as_lp(milp);
  auto x = lp_x;
  LpBasis basis;
  if (warm != nullptr) basis = *warm;

  int depth_cap = max_depth;
  if (depth_cap <= 0) {
    depth_cap = 0;
    for (const auto& v : milp.variables) depth_cap += is_int_type(v.type) ? 1 : 0;
  }

  auto beats_cutoff = [&](double obj) {
    if (cutoff == nullptr) return true;
    const double eps = 1e-9 * std::max(1.0, std::abs(*cutoff));
    return milp.sense == Sense::Minimize ? obj < *cutoff - eps : obj > *cutoff + eps;
  };

  auto resolve = [&]() {
    ++hr.lp_solves;
    LpBasis next;
    SolverResult lp = solve_lp_dual_simplex(relaxed, basis.empty() ? nullptr : &basis, &next);
    if (lp.status == SolverStatus::Optimal) {
      basis = std::move(next);
    } else if (lp.status != SolverStatus::Infeasible) {
      lp = RevisedSimplexSolver().solve(relaxed);
      basis = LpBasis{};
    }
    if ((lp.status == SolverStatus::Optimal || lp.status == SolverStatus::Feasible) &&
        (!lp.has_objective_value || beats_cutoff(lp.objective_value))) {
      x = lp.primal;
      return true;
    }
    return false;
  };

  for (int d = 0; d <= depth_cap; ++d) {
    HeuristicResult rounded = rounding_heuristic(node, x, int_tol);
    if (rounded.found && satisfies_constraints(milp, rounded.primal, 1e-6)) {
      rounded.objective = eval_obj(milp, rounded.primal);
      if (beats_cutoff(rounded.objective)) {
        rounded.method = "diving+" + rounded.method;
        rounded.lp_solves = hr.lp_solves;
        return rounded;
      }
    }
    if (d == depth_cap) break;

    int best = -1;
    bool best_up = false;
    int best_locks = std::numeric_limits<int>::max();
    double best_dist = 1.0;
    for (std::size_t i = 0; i < node.variables.size(); ++i) {
      const auto& v = node.variables[i];
      if (!is_int_type(v.type)) continue;
      auto it = x.find(v.name);
      if (it == x.end()) continue;
      const double down_dist = it->second - std::floor(it->second);
      if (down_dist <= int_tol || down_dist >= 1.0 - int_tol) continue;
      auto lk = locks.find(v.name);
      const Locks l = lk == locks.end() ? Locks{} : lk->second;
      const bool up = l.up < l.down || (l.up == l.down && down_dist >= 0.5);
      const int nlocks = up ? l.up : l.down;
      const double dist = up ? 1.0 - down_dist : down_dist;
      if (nlocks < best_locks || (nlocks == best_locks && dist < best_dist)) {
        best = static_cast<int>(i);
        best_up = up;
        best_locks = nlocks;
        best_dist = dist;
      }
    }
    if (best < 0) break;  // integral but infeasible after rounding: give up

    Variable& v = node.variables[static_cast<std::size_t>(best)];
    Variable& rv = relaxed.variables[static_cast<std::size_t>(best)];
    const double val = x.at(v.name);
    const double old_lb = v.lower_bound, old_ub = v.upper_bound;
    bool ok = false;
    for (int attempt = 0; attempt < 2 && !ok; ++attempt) {
      const bool up = attempt == 0 ? best_up : !best_up;
      v.lower_bound = old_lb;
      v.upper_bound = old_ub;
      if (up) v.lower_bound = std::max(old_lb, std::ceil(val));
      else v.upper_bound = std::min(old_ub, std::floor(val));
      rv.lower_bound = v.lower_bound;
      rv.upper_bound = v.upper_bound;
      if (v.lower_bound > v.upper_bound + 1e-9) continue;
      ok = resolve();
    }
    if (!ok) return hr;
  }
  return hr;
}

}  // namespace sovereign
