#include "sovereign/convex_qp.hpp"

#include "sovereign/revised_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace sovereign {
namespace {

int var_index(const OptimizationModel& m, const std::string& name) {
  for (std::size_t i = 0; i < m.variables.size(); ++i) {
    if (m.variables[i].name == name) return static_cast<int>(i);
  }
  return -1;
}

void build_q(const OptimizationModel& model, std::vector<double>& Q, std::size_t n) {
  Q.assign(n * n, 0.0);
  for (const auto& row : model.objective.quadratic) {
    const int i = var_index(model, row.first);
    if (i < 0) continue;
    for (const auto& col : row.second) {
      const int j = var_index(model, col.first);
      if (j < 0) continue;
      Q[static_cast<std::size_t>(j) * n + static_cast<std::size_t>(i)] += col.second;
      if (i != j) {
        Q[static_cast<std::size_t>(i) * n + static_cast<std::size_t>(j)] += col.second;
      }
    }
  }
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = i + 1; j < n; ++j) {
      const double mid = 0.5 * (Q[j * n + i] + Q[i * n + j]);
      Q[j * n + i] = mid;
      Q[i * n + j] = mid;
    }
    Q[i * n + i] += 1e-14;
  }
}

void build_c_min(const OptimizationModel& model, std::vector<double>& c, std::size_t n) {
  c.assign(n, 0.0);
  const double sign = (model.sense == Sense::Maximize) ? -1.0 : 1.0;
  for (std::size_t i = 0; i < n; ++i) {
    auto it = model.objective.linear.find(model.variables[i].name);
    if (it != model.objective.linear.end()) c[i] = sign * it->second;
  }
}

double qp_value_min(const std::vector<double>& Q, const std::vector<double>& c,
                    const std::vector<double>& x, std::size_t n) {
  double obj = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    obj += c[i] * x[i];
    for (std::size_t j = 0; j < n; ++j) {
      obj += 0.5 * x[i] * Q[j * n + i] * x[j];
    }
  }
  return obj;
}

std::vector<double> gradient_min(const std::vector<double>& Q, const std::vector<double>& c,
                                 const std::vector<double>& x, std::size_t n) {
  std::vector<double> g = c;
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) g[i] += Q[j * n + i] * x[j];
  }
  return g;
}

}  // namespace

ConvexQpSolver::ConvexQpSolver(QpOptions options) : options_(std::move(options)) {}

SolverResult ConvexQpSolver::solve(const OptimizationModel& model) const {
  SolverResult result;
  if (model.problem_type != ProblemType::QP && model.problem_type != ProblemType::LP) {
    result.status = SolverStatus::Error;
    result.message = "ConvexQpSolver expects QP.";
    return result;
  }
  for (const auto& v : model.variables) {
    if (v.type != VariableType::Continuous) {
      result.status = SolverStatus::Error;
      result.message = "QP variables must be continuous.";
      return result;
    }
  }
  const std::size_t n = model.variables.size();
  if (n == 0) {
    result.status = SolverStatus::Error;
    result.message = "QP has no variables.";
    return result;
  }

  if (model.objective.quadratic.empty()) {
    OptimizationModel lp = model;
    lp.problem_type = ProblemType::LP;
    return RevisedSimplexSolver().solve(lp);
  }

  std::vector<double> Q, c;
  build_q(model, Q, n);
  build_c_min(model, c, n);

  // Feasible start via LP (zero objective)
  OptimizationModel lp0 = model;
  lp0.problem_type = ProblemType::LP;
  lp0.objective.quadratic.clear();
  lp0.objective.linear.clear();
  for (const auto& v : lp0.variables) lp0.objective.linear[v.name] = 0.0;
  SolverResult start = RevisedSimplexSolver().solve(lp0);
  if (start.status == SolverStatus::Infeasible) {
    result.status = SolverStatus::Infeasible;
    result.message = "QP feasible region is empty.";
    return result;
  }
  if (start.status != SolverStatus::Optimal && start.status != SolverStatus::Feasible) {
    lp0.objective = model.objective;
    lp0.objective.quadratic.clear();
    start = RevisedSimplexSolver().solve(lp0);
    if (start.status != SolverStatus::Optimal && start.status != SolverStatus::Feasible) {
      result.status = SolverStatus::Error;
      result.message = "Failed to find a feasible starting point for QP.";
      return result;
    }
  }

  std::vector<double> x(n, 0.0);
  for (std::size_t i = 0; i < n; ++i) {
    auto it = start.primal.find(model.variables[i].name);
    x[i] = it == start.primal.end() ? model.variables[i].lower_bound : it->second;
  }

  // Frank-Wolfe: linearize QP and optimize over the polytope with LP
  std::int64_t iters = 0;
  for (int k = 0; k < options_.max_iterations; ++k) {
    std::vector<double> g = gradient_min(Q, c, x, n);

    OptimizationModel lp = model;
    lp.problem_type = ProblemType::LP;
    lp.sense = Sense::Minimize;
    lp.objective.quadratic.clear();
    lp.objective.linear.clear();
    for (std::size_t i = 0; i < n; ++i) {
      lp.objective.linear[model.variables[i].name] = g[i];
    }
    SolverResult dir = RevisedSimplexSolver().solve(lp);
    ++iters;
    if (dir.status != SolverStatus::Optimal && dir.status != SolverStatus::Feasible) {
      break;
    }

    std::vector<double> s(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      auto it = dir.primal.find(model.variables[i].name);
      s[i] = it == dir.primal.end() ? x[i] : it->second;
    }

    // FW gap
    double gap = 0.0;
    for (std::size_t i = 0; i < n; ++i) gap += g[i] * (x[i] - s[i]);
    if (gap <= options_.optimality_tol) break;

    // Exact line search on quadratic: min_{a in [0,1]} f(x + a(d)) d=s-x
    std::vector<double> d(n);
    for (std::size_t i = 0; i < n; ++i) d[i] = s[i] - x[i];
    double dQd = 0.0;
    double gTd = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      gTd += g[i] * d[i];
      for (std::size_t j = 0; j < n; ++j) dQd += d[i] * Q[j * n + i] * d[j];
    }
    double alpha = 1.0;
    if (dQd > 1e-16) alpha = std::min(1.0, std::max(0.0, -gTd / dQd));
    else if (gTd >= 0) alpha = 0.0;

    for (std::size_t i = 0; i < n; ++i) x[i] += alpha * d[i];
    if (alpha <= 1e-16) break;
  }

  double obj = model.objective.constant;
  for (const auto& kv : model.objective.linear) {
    const int i = var_index(model, kv.first);
    if (i >= 0) obj += kv.second * x[static_cast<std::size_t>(i)];
  }
  for (const auto& row : model.objective.quadratic) {
    const int i = var_index(model, row.first);
    if (i < 0) continue;
    for (const auto& col : row.second) {
      const int j = var_index(model, col.first);
      if (j < 0) continue;
      obj += 0.5 * col.second * x[static_cast<std::size_t>(i)] *
             x[static_cast<std::size_t>(j)];
    }
  }

  result.status = SolverStatus::Optimal;
  result.has_objective_value = true;
  result.objective_value = obj;
  result.iterations = iters;
  result.message = "Convex QP solved by Frank-Wolfe with LP linearization oracle "
                   "(earlier approach; default QP path is Mehrotra IPM — "
                   "set SOVEREIGN_QP_ALGORITHM=frank_wolfe to force this).";
  for (std::size_t i = 0; i < n; ++i) {
    result.primal[model.variables[i].name] = x[i];
  }
  (void)qp_value_min;
  return result;
}

}  // namespace sovereign
