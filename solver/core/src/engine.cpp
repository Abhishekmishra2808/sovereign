#include "sovereign/engine.hpp"

#include "sovereign/lp_solver.hpp"
#include "sovereign/milp_solver.hpp"
#include "sovereign/model_validator.hpp"
#include "sovereign/presolve.hpp"
#include "sovereign/qp_solver.hpp"

#include <chrono>
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
        result = LpSolver().solve(to_solve, options.lp_algorithm);
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

  result.runtime_seconds = monotonic_seconds() - t0;
  return result;
}

}  // namespace sovereign
