#include "sovereign/verifier.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <unordered_map>

namespace sovereign {

VerificationResult SolutionVerifier::verify(const OptimizationModel& model,
                                             const SolverResult& result,
                                             double tol) const {
  VerificationResult out;
  out.message = "Verifier executed.";

  if (result.status == SolverStatus::NotImplemented ||
      result.status == SolverStatus::Error ||
      result.status == SolverStatus::NumericalError ||
      result.status == SolverStatus::TimeLimit ||
      result.status == SolverStatus::IterationLimit) {
    out.is_valid = false;
    out.issues.push_back("Solver status does not provide a feasible primal solution: " +
                         to_string(result.status));
    out.message = "Verification skipped for non-feasible status.";
    return out;
  }

  if (result.status == SolverStatus::Infeasible ||
      result.status == SolverStatus::Unbounded) {
    // A label alone cannot prove that no feasible point exists, or that an
    // improving ray exists. Until the solver returns checkable Farkas/ray
    // certificates, keep these outcomes unverified instead of accepting a
    // potentially false proof claim.
    out.is_valid = false;
    out.issues.push_back("Status " + to_string(result.status) +
                         " has no independently checkable certificate.");
    out.message = "Status claim is unverified.";
    return out;
  }

  std::unordered_map<std::string, double> x = result.primal;
  for (const auto& v : model.variables) {
    if (x.find(v.name) == x.end()) {
      out.issues.push_back("Missing primal value for variable " + v.name);
      continue;
    }
    const double val = x.at(v.name);
    if (!std::isfinite(val)) {
      out.issues.push_back("Non-finite primal for " + v.name);
      continue;
    }
    const double below = v.lower_bound - val;
    const double above = val - v.upper_bound;
    out.max_bound_violation = std::max(out.max_bound_violation, std::max(below, above));
    if (v.type == VariableType::Integer || v.type == VariableType::Binary) {
      const double frac = std::abs(val - std::round(val));
      out.max_integrality_violation = std::max(out.max_integrality_violation, frac);
    }
    if (v.type == VariableType::Binary) {
      if (val < -tol || val > 1.0 + tol) {
        out.issues.push_back("Binary variable out of [0,1]: " + v.name);
      }
    }
  }

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
  out.recomputed_objective = obj;

  for (const auto& c : model.constraints) {
    double lhs = 0.0;
    for (const auto& kv : c.linear) {
      auto it = x.find(kv.first);
      if (it != x.end()) lhs += kv.second * it->second;
    }
    if (!std::isfinite(lhs)) {
      out.issues.push_back("Non-finite constraint activity: " + c.name);
      continue;
    }
    double viol = 0.0;
    if (c.sense == ConstraintSense::Le) {
      viol = lhs - c.rhs;
    } else if (c.sense == ConstraintSense::Ge) {
      viol = c.rhs - lhs;
    } else {
      viol = std::abs(lhs - c.rhs);
    }
    out.max_constraint_violation = std::max(out.max_constraint_violation, viol);
  }

  if (out.max_bound_violation > tol) {
    out.issues.push_back("Bound violation exceeds tolerance.");
  }
  if (out.max_constraint_violation > tol) {
    out.issues.push_back("Constraint violation exceeds tolerance.");
  }
  if (out.max_integrality_violation > tol) {
    out.issues.push_back("Integrality violation exceeds tolerance.");
  }
  if (result.has_objective_value &&
      std::abs(result.objective_value - out.recomputed_objective) > tol * (1.0 + std::abs(obj))) {
    std::ostringstream oss;
    oss << "Reported objective differs from recomputed value by "
        << std::abs(result.objective_value - out.recomputed_objective);
    out.issues.push_back(oss.str());
  }
  if (result.status == SolverStatus::Optimal && result.optimality_gap > 1e-2) {
    out.issues.push_back("OPTIMAL status with large optimality_gap.");
  }

  // An OPTIMAL claim is a claim about DUAL feasibility, and the only objective
  // data the verifier has is the reported gap. If the solver says "optimal" but
  // reports a duality gap that is not actually tight, the primal check above
  // can still pass while the answer is a merely feasible point. Catch that
  // here rather than letting it reach a caller as a proof.
  if (result.status == SolverStatus::Optimal && result.duality_gap > tol) {
    std::ostringstream oss;
    oss << "OPTIMAL status contradicted by reported relative duality gap "
        << result.duality_gap << " (tolerance " << tol
        << "). The point may be feasible but optimality is not established.";
    out.issues.push_back(oss.str());
  }

  // A non-conclusive status must never carry a verified objective. If it does,
  // a downstream consumer could read the number as an answer.
  if (!is_conclusive(result.status) && result.status != SolverStatus::Feasible &&
      result.has_objective_value && result.optimality_gap > 1e-2) {
    std::ostringstream oss;
    oss << "Status " << to_string(result.status)
        << " is not a proof, but an objective value is attached with gap "
        << result.optimality_gap << ".";
    out.issues.push_back(oss.str());
  }

  out.is_valid = out.issues.empty();
  out.message = out.is_valid ? "Solution verified." : "Solution failed verification.";
  return out;
}

}  // namespace sovereign
