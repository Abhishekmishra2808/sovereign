#include "sovereign/lp_solver.hpp"

#include "sovereign/dual_simplex.hpp"
#include "sovereign/interior_point.hpp"
#include "sovereign/revised_simplex.hpp"

#include <cstdlib>
#include <sstream>
#include <string>

namespace sovereign {
namespace {

// Why the primary method handed off. This is the "real reason" that gets
// attached to the fallback result, so a reader of the output can tell an IPM
// that converged from one that gave up and was rescued by the simplex.
std::string describe_handoff(const SolverResult& primary) {
  std::ostringstream oss;
  oss << to_string(primary.status);
  if (!primary.message.empty()) oss << ": " << primary.message;
  return oss.str();
}

void merge_lp_diagnostics(SolverResult& target, const SolverResult& source) {
  for (const auto& entry : source.lp_diagnostics) {
    target.lp_diagnostics[entry.first] = entry.second;
  }
}

}  // namespace

SolverResult LpSolver::solve(const OptimizationModel& model, const std::string& algorithm) const {
  const char* algo = std::getenv("SOVEREIGN_LP_ALGORITHM");
  const std::string choice = algorithm.empty() ? (algo ? algo : "auto") : algorithm;
  if (choice != "auto" && choice != "simplex" && choice != "dual" && choice != "ipm") {
    SolverResult invalid;
    invalid.status = SolverStatus::Error;
    invalid.message = "Unsupported LP algorithm: " + choice;
    return invalid;
  }

  // Dual simplex first; it certifies only OPTIMAL and INFEASIBLE, so anything
  // else (an unbounded LP included) goes to the primal revised simplex.
  auto run_primal = [&]() {
    RevisedSimplexOptions opt;
    return RevisedSimplexSolver(opt).solve(model);
  };
  auto run_simplex = [&]() {
    SolverResult dual = solve_lp_dual_simplex(model, nullptr, nullptr);
    if (dual.status == SolverStatus::Optimal || dual.status == SolverStatus::Infeasible) {
      return dual;
    }
    SolverResult primal = run_primal();
    merge_lp_diagnostics(primal, dual);
    primal.warnings.push_back("Dual simplex did not finish (" + describe_handoff(dual) +
                              "); solved with the primal revised simplex.");
    return primal;
  };
  auto run_ipm = [&]() {
    InteriorPointOptions opt;
    return InteriorPointSolver(opt).solve(model);
  };

  if (choice == "simplex") {
    return run_simplex();
  }
  if (choice == "dual") {
    return solve_lp_dual_simplex(model, nullptr, nullptr);
  }
  if (choice == "ipm") {
    return run_ipm();
  }

  // auto: dual simplex first. On the Netlib set it roughly halves the shifted
  // geometric mean against interior point first, and it never pays for a
  // dense normal-equations factor (fit2p, dfl001). It certifies only OPTIMAL
  // and INFEASIBLE; anything else goes to interior point, then primal simplex.
  SolverResult dual = solve_lp_dual_simplex(model, nullptr, nullptr);
  if (dual.status == SolverStatus::Optimal || dual.status == SolverStatus::Infeasible) {
    return dual;
  }

  SolverResult ipm = run_ipm();
  merge_lp_diagnostics(ipm, dual);
  ipm.warnings.push_back("Dual simplex did not finish (" + describe_handoff(dual) +
                         "); switched to interior point.");

  switch (ipm.status) {
    case SolverStatus::Optimal:
    case SolverStatus::Infeasible:
    case SolverStatus::Unbounded:
      // A proven verdict. Nothing to rescue.
      return ipm;

    case SolverStatus::Feasible:
      // The IPM handed back a point it proved nothing about. Try to get a
      // proven answer, but keep the IPM point as a floor rather than discarding
      // a usable candidate.
      {
        SolverResult simplex = run_primal();
        merge_lp_diagnostics(simplex, dual);
        merge_lp_diagnostics(simplex, ipm);
        if (is_conclusive(simplex.status)) {
          simplex.warnings.push_back("Interior point returned an unproven point (" +
                                     describe_handoff(ipm) +
                                     "); solved to optimality by revised simplex instead.");
          return simplex;
        }
        simplex.warnings.push_back("Interior point returned an unproven point (" +
                                   describe_handoff(ipm) +
                                   "); simplex fallback also failed (" +
                                   describe_handoff(simplex) +
                                   "). Reporting the interior point as a candidate only.");
        if (simplex.has_objective_value) return simplex;
        merge_lp_diagnostics(ipm, simplex);
        return ipm;
      }

    case SolverStatus::TimeLimit:
    case SolverStatus::IterationLimit:
    case SolverStatus::NumericalError:
    case SolverStatus::Error:
    case SolverStatus::NotImplemented:
    default:
      break;
  }

  // IPM did not produce a verdict. Run the simplex, and say exactly why we are
  // running it rather than letting the fallback look like the normal path.
  SolverResult simplex = run_primal();
  merge_lp_diagnostics(simplex, dual);
  merge_lp_diagnostics(simplex, ipm);
  simplex.warnings.push_back("Interior point did not solve the problem (" +
                             describe_handoff(ipm) +
                             "); fell back to revised simplex.");

  if (is_conclusive(simplex.status)) {
    return simplex;
  }

  // Neither method reached a verdict. Returning the simplex failure verbatim
  // would discard the diagnostic that the IPM got further than the simplex did.
  // Report NumericalError (we do not know the answer) and carry both reasons.
  std::ostringstream oss;
  oss << "LP unsolved. Dual simplex: " << describe_handoff(dual)
      << " | Interior point: " << describe_handoff(ipm)
      << " | Revised simplex: " << describe_handoff(simplex);
  SolverResult failed = simplex.has_objective_value
                            ? simplex
                            : (ipm.has_objective_value ? ipm : SolverResult());
  merge_lp_diagnostics(failed, dual);
  merge_lp_diagnostics(failed, ipm);
  merge_lp_diagnostics(failed, simplex);
  failed.status = (simplex.status == SolverStatus::Infeasible ||
                   simplex.status == SolverStatus::Unbounded)
                      ? simplex.status
                      : SolverStatus::NumericalError;
  failed.message = oss.str();
  failed.has_objective_value = false;
  return failed;
}

}  // namespace sovereign
