#include "sovereign/qp_solver.hpp"

#include "sovereign/convex_qp.hpp"
#include "sovereign/qp_interior_point.hpp"

#include <cstdlib>
#include <cstring>
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
  SolverResult fw = ConvexQpSolver().solve(model);
  if (fw.status == SolverStatus::Optimal || fw.status == SolverStatus::Feasible) {
    fw.warnings.push_back("QP-IPM did not converge; fell back to Frank–Wolfe (" +
                          ipm.message + ")");
  }
  return fw;
}

}  // namespace sovereign
