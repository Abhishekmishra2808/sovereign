#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace sovereign {

enum class ProblemType { LP, QP, MILP };
enum class Sense { Minimize, Maximize };
enum class ConstraintSense { Le, Ge, Eq };
enum class VariableType { Continuous, Integer, Binary };

// A status is a *claim about the answer*, so the distinctions matter: a node
// limit, a time limit and a numerical breakdown all mean "we do not know", but
// for different reasons and with different remedies. Collapsing them into
// ERROR (as earlier revisions did) hides the reason a search stopped, which is
// exactly the information a caller needs in order to trust or reject the run.
enum class SolverStatus {
  Optimal,          // proven optimal within tolerance
  Feasible,         // feasible point found, optimality NOT proven
  Infeasible,       // proven infeasible
  Unbounded,        // proven unbounded
  TimeLimit,        // wall-clock budget exhausted
  IterationLimit,   // iteration budget exhausted
  NumericalError,   // factorization/residual breakdown; answer not trustworthy
  Error,            // bad input, unsupported model, internal fault
  NotImplemented,
};

struct Variable {
  std::string name;
  VariableType type = VariableType::Continuous;
  double lower_bound = 0.0;
  double upper_bound = 1e30;
};

struct Objective {
  double constant = 0.0;
  std::unordered_map<std::string, double> linear;
  // Optional quadratic terms: name -> {other_name -> coeff}. Stored sparsely.
  std::unordered_map<std::string, std::unordered_map<std::string, double>> quadratic;
};

struct Constraint {
  std::string name;
  std::unordered_map<std::string, double> linear;
  ConstraintSense sense = ConstraintSense::Le;
  double rhs = 0.0;
};

struct OptimizationModel {
  ProblemType problem_type = ProblemType::LP;
  Sense sense = Sense::Minimize;
  std::vector<Variable> variables;
  Objective objective;
  std::vector<Constraint> constraints;
};

struct SolverResult {
  SolverStatus status = SolverStatus::NotImplemented;
  bool has_objective_value = false;
  double objective_value = 0.0;
  std::unordered_map<std::string, double> primal;
  double optimality_gap = 0.0;

  // Certificates / diagnostics. `duality_gap` is the relative duality gap
  // |c'x - b'y| / (1 + |c'x|) and is the quantity that actually justifies an
  // OPTIMAL claim. The residuals are the relative primal/dual inf-norms. They
  // are reported even on failure so the caller can see how far off we were.
  double duality_gap = 0.0;
  double primal_residual = 0.0;
  double dual_residual = 0.0;

  std::int64_t iterations = 0;
  std::int64_t nodes = 0;
  double runtime_seconds = 0.0;
  std::string message;
  std::vector<std::string> warnings;
};

struct VerificationResult {
  bool is_valid = false;
  double max_constraint_violation = 0.0;
  double max_bound_violation = 0.0;
  double max_integrality_violation = 0.0;
  double recomputed_objective = 0.0;
  std::vector<std::string> issues;
  std::string message;
};

std::string to_string(ProblemType type);
std::string to_string(Sense sense);
std::string to_string(ConstraintSense sense);
std::string to_string(VariableType type);
std::string to_string(SolverStatus status);

// OPTIMAL / INFEASIBLE / UNBOUNDED only. Every other status means the run stopped
// without proving anything, so an objective value attached to it is a
// *candidate*, not an answer.
bool is_conclusive(SolverStatus status);

ProblemType problem_type_from_string(const std::string& s);
Sense sense_from_string(const std::string& s);
ConstraintSense constraint_sense_from_string(const std::string& s);
VariableType variable_type_from_string(const std::string& s);

}  // namespace sovereign
