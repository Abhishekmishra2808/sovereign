#include "sovereign/lp_solver.hpp"

#include "sovereign/interior_point.hpp"
#include "sovereign/revised_simplex.hpp"

#include <cstdlib>
#include <string>

namespace sovereign {

SolverResult LpSolver::solve(const OptimizationModel& model) const {
  const char* algo = std::getenv("SOVEREIGN_LP_ALGORITHM");
  const std::string choice = algo ? algo : "auto";

  auto run_simplex = [&]() {
    RevisedSimplexOptions opt;
    return RevisedSimplexSolver(opt).solve(model);
  };
  auto run_ipm = [&]() {
    InteriorPointOptions opt;
    return InteriorPointSolver(opt).solve(model);
  };

  if (choice == "simplex") {
    return run_simplex();
  }
  if (choice == "ipm") {
    return run_ipm();
  }

  // auto: prefer IPM, fall back to revised simplex on failure/non-convergence
  SolverResult ipm = run_ipm();
  if (ipm.status == SolverStatus::Optimal || ipm.status == SolverStatus::Feasible ||
      ipm.status == SolverStatus::Infeasible || ipm.status == SolverStatus::Unbounded) {
    return ipm;
  }
  SolverResult simplex = run_simplex();
  if (!ipm.message.empty()) {
    simplex.warnings.push_back("IPM fell back to revised simplex: " + ipm.message);
  }
  return simplex;
}

}  // namespace sovereign
