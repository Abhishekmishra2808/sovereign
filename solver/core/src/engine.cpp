#include "sovereign/engine.hpp"

#include "sovereign/lp_solver.hpp"
#include "sovereign/milp_solver.hpp"
#include "sovereign/model_validator.hpp"
#include "sovereign/presolve.hpp"
#include "sovereign/qp_solver.hpp"

#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace sovereign {
namespace {

// MinGW's steady_clock advances in ~1 ms steps on Windows, which reports small
// solves as 0 s; the performance counter resolves well under a microsecond.
double monotonic_seconds() {
#if defined(_WIN32)
  LARGE_INTEGER frequency, now;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&now);
  return static_cast<double>(now.QuadPart) / static_cast<double>(frequency.QuadPart);
#else
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

double evaluate_objective(const OptimizationModel& model,
                          const std::unordered_map<std::string, double>& x) {
  double obj = model.objective.constant;
  for (const auto& kv : model.objective.linear) {
    auto it = x.find(kv.first);
    if (it != x.end()) obj += kv.second * it->second;
  }
  for (const auto& row : model.objective.quadratic) {
    auto xi = x.find(row.first);
    if (xi == x.end()) continue;
    for (const auto& col : row.second) {
      auto xj = x.find(col.first);
      if (xj == x.end()) continue;
      obj += 0.5 * col.second * xi->second * xj->second;
    }
  }
  return obj;
}

// Largest bound or row violation of x on the original model, relative to
// 1 + |bound|. A missing or non-finite value counts as an infinite violation.
double max_relative_violation(const OptimizationModel& model,
                              const std::unordered_map<std::string, double>& x) {
  constexpr double inf = std::numeric_limits<double>::infinity();
  double worst = 0.0;
  for (const auto& v : model.variables) {
    const auto it = x.find(v.name);
    if (it == x.end() || !std::isfinite(it->second)) return inf;
    const double val = it->second;
    if (val < v.lower_bound) worst = std::max(worst, (v.lower_bound - val) / (1.0 + std::abs(v.lower_bound)));
    if (val > v.upper_bound) worst = std::max(worst, (val - v.upper_bound) / (1.0 + std::abs(v.upper_bound)));
  }
  for (const auto& c : model.constraints) {
    double lhs = 0.0;
    for (const auto& kv : c.linear) {
      const auto it = x.find(kv.first);
      if (it != x.end()) lhs += kv.second * it->second;
    }
    const double viol = c.sense == ConstraintSense::Le   ? lhs - c.rhs
                        : c.sense == ConstraintSense::Ge ? c.rhs - lhs
                                                         : std::abs(lhs - c.rhs);
    worst = std::max(worst, viol / (1.0 + std::abs(c.rhs)));
  }
  return worst;
}

bool has_primal(const SolverResult& r) {
  return r.status == SolverStatus::Optimal || r.status == SolverStatus::Feasible;
}

}  // namespace

SolverResult OptimizationEngine::solve(const OptimizationModel& model, const EngineOptions& options) const {
  const double t0 = monotonic_seconds();

  const std::string validation = ModelValidator::validate(model);
  if (!validation.empty()) {
    SolverResult bad;
    bad.status = SolverStatus::Error;
    bad.message = validation;
    return bad;
  }

  Presolver presolver;
  PresolveResult prep;
  // Presolve only reduces LPs; other models are solved in place rather than
  // copied, which matters at millions of variables.
  const bool presolved = options.presolve && model.problem_type == ProblemType::LP;
  if (presolved) prep = presolver.run(model);
  const OptimizationModel& to_solve = presolved ? prep.reduced : model;

  SolverResult result;
  if (prep.infeasible) {
    result.status = SolverStatus::Infeasible;
    result.message = prep.message.empty() ? "Infeasible (presolve)." : prep.message;
  } else if (prep.unbounded) {
    result.status = SolverStatus::Unbounded;
    result.message = prep.message.empty() ? "Unbounded (presolve)." : prep.message;
  } else if (to_solve.variables.empty()) {
    result.status = SolverStatus::Optimal;
    result.has_objective_value = true;
    result.objective_value = 0.0;
    result.message = "Optimal (presolve fixed all variables).";
    result = presolver.recover(result, prep, model.sense);
    result.objective_value = evaluate_objective(model, result.primal);
    result.has_objective_value = true;
  } else {
    switch (to_solve.problem_type) {
      case ProblemType::LP:
        {
          std::string algorithm = options.lp_algorithm;
          // Sparse large LPs are dominated by simplex basis updates even when
          // their structure is easy for the existing sparse IPM. Preserve
          // explicit caller choices, but select IPM automatically for models
          // large enough that the default simplex path becomes impractical.
          if (algorithm.empty() && to_solve.variables.size() > 50000 &&
              to_solve.constraints.size() > 10000) {
            algorithm = "ipm";
          }
          result = LpSolver().solve(to_solve, algorithm);
        }
        break;
      case ProblemType::QP:
        result = QpSolver().solve(to_solve);
        break;
      case ProblemType::MILP:
        result = MilpSolver().solve(to_solve);
        break;
    }
    if (presolved) result = presolver.recover(result, prep, model.sense);
    if (result.status == SolverStatus::Optimal ||
        result.status == SolverStatus::Feasible) {
      result.objective_value = evaluate_objective(model, result.primal);
      result.has_objective_value = true;
    }
  }

  // Presolve's reductions are checked rather than trusted: an infeasibility
  // claim, or a recovered point that violates the original model, is solved
  // again without presolve. A presolve fault then costs time, not correctness.
  if (presolved) {
    std::string why;
    if (prep.infeasible) {
      why = "presolve reported infeasibility (" + prep.message + ")";
    } else if (has_primal(result) && max_relative_violation(model, result.primal) > 1e-6) {
      why = "the presolved solution violated the original model";
    }
    if (!why.empty()) {
      result = LpSolver().solve(model, options.lp_algorithm);
      if (has_primal(result)) {
        result.objective_value = evaluate_objective(model, result.primal);
        result.has_objective_value = true;
      }
      result.warnings.push_back("Re-solved without presolve because " + why + ".");
    }
  }

  if (presolved) {
    result.presolve_fixed_variables = prep.stats.fixed_variables;
    result.presolve_substituted_variables = prep.stats.substituted_variables;
    result.presolve_removed_constraints = prep.stats.removed_constraints;
    result.presolve_tightened_bounds = prep.stats.tightened_bounds;
    result.presolve_passes = prep.stats.passes;
  }
  result.runtime_seconds = monotonic_seconds() - t0;
  return result;
}

}  // namespace sovereign
