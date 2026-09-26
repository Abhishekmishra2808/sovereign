#include "sovereign/engine.hpp"

#include "sovereign/lp_solver.hpp"
#include "sovereign/milp_solver.hpp"
#include "sovereign/model_validator.hpp"
#include "sovereign/presolve.hpp"
#include "sovereign/qp_solver.hpp"

#include <chrono>
#include <unordered_map>

namespace sovereign {
namespace {

double evaluate_objective(const OptimizationModel& model,
                          const std::unordered_map<std::string, double>& x) {
  double obj = 0.0;
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

SolverResult OptimizationEngine::solve(const OptimizationModel& model) const {
  const auto t0 = std::chrono::steady_clock::now();

  const std::string validation = ModelValidator::validate(model);
  if (!validation.empty()) {
    SolverResult bad;
    bad.status = SolverStatus::Error;
    bad.message = validation;
    return bad;
  }

  Presolver presolver;
  const PresolveResult prep = presolver.run(model);

  SolverResult result;
  if (prep.infeasible) {
    result.status = SolverStatus::Infeasible;
    result.message = prep.message.empty() ? "Infeasible (presolve)." : prep.message;
  } else if (prep.unbounded) {
    result.status = SolverStatus::Unbounded;
    result.message = prep.message.empty() ? "Unbounded (presolve)." : prep.message;
  } else if (prep.reduced.variables.empty()) {
    result.status = SolverStatus::Optimal;
    result.has_objective_value = true;
    result.objective_value = 0.0;
    result.message = "Optimal (presolve fixed all variables).";
    result = presolver.recover(result, prep, model.sense);
    result.objective_value = evaluate_objective(model, result.primal);
    result.has_objective_value = true;
  } else {
    switch (prep.reduced.problem_type) {
      case ProblemType::LP:
        result = LpSolver().solve(prep.reduced);
        break;
      case ProblemType::QP:
        result = QpSolver().solve(prep.reduced);
        break;
      case ProblemType::MILP:
        result = MilpSolver().solve(prep.reduced);
        break;
    }
    result = presolver.recover(result, prep, model.sense);
    if (result.status == SolverStatus::Optimal ||
        result.status == SolverStatus::Feasible) {
      result.objective_value = evaluate_objective(model, result.primal);
      result.has_objective_value = true;
    }
  }

  const auto t1 = std::chrono::steady_clock::now();
  result.runtime_seconds = std::chrono::duration<double>(t1 - t0).count();
  return result;
}

}  // namespace sovereign
