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

// Solver-side LP diagnostics are observational. They are serialized so a
// benchmark can explain a stall or numerical failure without changing pivot
// selection, tolerances, or termination decisions.
struct LpDiagnostics {
  bool scaling_applied = false;
  double coefficient_min_abs_before = 0.0;
  double coefficient_max_abs_before = 0.0;
  double coefficient_min_abs_after = 0.0;
  double coefficient_max_abs_after = 0.0;
  std::int64_t degenerate_pivots = 0;
  std::int64_t refactorizations = 0;
  std::vector<std::int64_t> objective_history_iterations;
  std::vector<double> objective_history;
  std::vector<std::int64_t> gap_history_iterations;
  std::vector<double> gap_history;
  std::vector<double> mu_history;
  std::vector<double> primal_step_history;
  std::vector<double> dual_step_history;
  std::int64_t stall_iteration = -1;
  double stall_gap = 0.0;
  double stall_mu = 0.0;
  double stall_primal_step = 0.0;
  double stall_dual_step = 0.0;
  double final_primal_residual = 0.0;
  double final_dual_residual = 0.0;
  double final_gap = 0.0;
  std::string basis_state;
  std::string stop_reason;
};

// MILP search diagnostics are observational. They describe the proof state and
// work performed by branch-and-bound without affecting branching, pruning, or
// termination decisions.
struct MipDiagnostics {
  bool has_best_bound = false;
  double best_bound = 0.0;
  double time_to_first_incumbent = -1.0;
  std::unordered_map<std::string, std::int64_t> cuts_by_family;
  std::int64_t presolve_fixed_variables = 0;
  std::int64_t presolve_substituted_variables = 0;
  std::int64_t presolve_removed_constraints = 0;
  std::int64_t presolve_tightened_bounds = 0;
  std::int64_t presolve_passes = 0;
  std::int64_t node_lp_failures = 0;
  std::int64_t dropped_subtrees = 0;
  std::int64_t numerical_error_nodes = 0;
  std::int64_t iteration_limit_nodes = 0;
  std::int64_t unbounded_nodes = 0;
  std::int64_t cut_validity_rejections = 0;
  bool debug_solution_enabled = false;
  bool debug_solution_violation = false;
  std::int64_t debug_solution_checks = 0;
  std::string debug_solution_path;
  std::string debug_solution_first_violation;
};

struct SolverResult {
  SolverStatus status = SolverStatus::NotImplemented;
  bool has_objective_value = false;
  double objective_value = 0.0;
  std::unordered_map<std::string, double> primal;
  // Optional solver-side dual/slack diagnostics. Keys are model variable names
  // for dual slacks and implementation-defined row names for row multipliers.
  std::unordered_map<std::string, double> dual;
  std::unordered_map<std::string, double> slacks;
  double optimality_gap = 0.0;

  // Certificates / diagnostics. `duality_gap` is the relative duality gap
  // |c'x - b'y| / (1 + |c'x|) and is the quantity that actually justifies an
  // OPTIMAL claim. The residuals are the relative primal/dual inf-norms. They
  // are reported even on failure so the caller can see how far off we were.
  double duality_gap = 0.0;
  double primal_residual = 0.0;
  double dual_residual = 0.0;
  // "original_model" means row_<i> keys match the model passed to the LP
  // solver. Presolve recovery cannot currently reconstruct multipliers for
  // eliminated rows, so it marks the result as "reduced_presolve_model".
  std::string dual_certificate_space;
  std::unordered_map<std::string, LpDiagnostics> lp_diagnostics;
  MipDiagnostics mip_diagnostics;

  // Presolve bookkeeping is diagnostic metadata only. The solver algorithms
  // and their tolerances do not depend on these counters.
  int presolve_fixed_variables = 0;
  int presolve_substituted_variables = 0;
  int presolve_removed_constraints = 0;
  int presolve_tightened_bounds = 0;
  int presolve_passes = 0;

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
