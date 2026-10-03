#include "sovereign/qp_solver.hpp"

#include "sovereign/convex_qp.hpp"
#include "sovereign/qp_interior_point.hpp"
#include "sovereign/verifier.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>

namespace sovereign {
namespace {

enum class QpAlgo { Auto, Ipm, FrankWolfe };

QpAlgo parse_qp_algo() {
  const char* s = std::getenv("SOVEREIGN_QP_ALGORITHM");
  if (!s) return QpAlgo::Auto;
  const std::string v(s);
  if (v == "frank_wolfe" || v == "fw" || v == "frank-wolfe") return QpAlgo::FrankWolfe;
  if (v == "ipm" || v == "barrier") return QpAlgo::Ipm;
  return QpAlgo::Auto;
}

}  // namespace

SolverResult QpSolver::solve(const OptimizationModel& model) const {
  const QpAlgo algo = parse_qp_algo();
  if (algo == QpAlgo::FrankWolfe) {
    return ConvexQpSolver().solve(model);
  }

  SolverResult ipm = QpInteriorPointSolver().solve(model);
  if (algo == QpAlgo::Ipm) return ipm;

  // auto: prefer Mehrotra QP-IPM; fall back to Frank–Wolfe (earlier approach)
  if (ipm.status == SolverStatus::Optimal || ipm.status == SolverStatus::Feasible ||
      ipm.status == SolverStatus::Infeasible || ipm.status == SolverStatus::Unbounded) {
    return ipm;
  }
  const bool has_free_variable = std::any_of(model.variables.begin(), model.variables.end(),
                                             [](const Variable& v) {
                                               return v.lower_bound <= -1e29 &&
                                                      v.upper_bound >= 1e29;
                                             });
  if (has_free_variable) {
    ipm.status = SolverStatus::NumericalError;
    ipm.message += " Frank-Wolfe fallback disabled because the QP has free variables.";
    return ipm;
  }
  SolverResult fw = ConvexQpSolver().solve(model);
  const char* off = std::getenv("SOVEREIGN_QP_DISABLE");
  const bool guard = !off || (std::string(",") + off + ",").find(",fw_guard,") == std::string::npos;
  if (guard && (fw.status == SolverStatus::Optimal || fw.status == SolverStatus::Feasible)) {
    // Frank-Wolfe can stop on a point that violates the rows (PRIMALC2) and
    // still call it optimal, so its answer is only passed on once verified.
    const VerificationResult check = SolutionVerifier().verify(model, fw, 1e-6);
    bool runaway = false;
    for (const auto& kv : fw.primal) runaway = runaway || !(std::abs(kv.second) < 1e15);
    if (runaway || check.max_constraint_violation > 1e-6 || check.max_bound_violation > 1e-6) {
      std::ostringstream oss;
      oss << "QP unsolved. QP-IPM: " << ipm.message << " | Frank-Wolfe returned ";
      if (runaway) oss << "a point with values beyond 1e15.";
      else oss << "a point violating the constraints by "
               << std::max(check.max_constraint_violation, check.max_bound_violation) << ".";
      SolverResult failed;
      failed.status = SolverStatus::NumericalError;
      failed.iterations = ipm.iterations + fw.iterations;
      failed.message = oss.str();
      return failed;
    }
    fw.warnings.push_back("QP-IPM did not converge; fell back to Frank–Wolfe (" +
                          ipm.message + ")");
  }
  return fw;
}

}  // namespace sovereign
