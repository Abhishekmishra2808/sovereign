#pragma once

#include "sovereign/types.hpp"

#include <string>
#include <vector>

namespace sovereign {

enum class PresolveActionType {
  FixVariable,
  SubstituteVariable,  // name = coeff * other + offset
  DropConstraint
};

struct PresolveAction {
  PresolveActionType type = PresolveActionType::FixVariable;
  std::string name;
  std::string other;   // for substitution
  double value = 0.0;  // fixed value, or substitution offset
  double coeff = 0.0;  // substitution coefficient
  std::string note;
};

struct PresolveStats {
  int fixed_variables = 0;
  int substituted_variables = 0;
  int removed_constraints = 0;
  int tightened_bounds = 0;
  int passes = 0;
};

struct PresolveResult {
  OptimizationModel reduced;
  bool infeasible = false;
  bool unbounded = false;
  std::string message;
  std::vector<PresolveAction> actions;  // apply in reverse for recovery
  PresolveStats stats;
  double objective_offset = 0.0;  // constant added to reduced objective (original sense)
};

struct PresolveOptions {
  double tolerance = 1e-9;
  int max_passes = 20;
  bool enable_bound_tightening = true;
  bool enable_singleton = true;
  bool enable_redundant = true;
  bool enable_substitution = true;
};

class Presolver {
 public:
  explicit Presolver(PresolveOptions options = {});

  PresolveResult run(const OptimizationModel& model) const;

  // Map a reduced-model SolverResult back to the original variable space.
  SolverResult recover(const SolverResult& reduced_result,
                       const PresolveResult& prep,
                       Sense original_sense) const;

 private:
  PresolveOptions options_;
};

}  // namespace sovereign
