#include "sovereign/model_validator.hpp"

#include <cmath>
#include <sstream>
#include <unordered_set>

namespace sovereign {

std::string ModelValidator::validate(const OptimizationModel& model) {
  std::ostringstream err;

  if (model.variables.empty()) {
    err << "Model has no variables. ";
  }

  std::unordered_set<std::string> names;
  for (const auto& v : model.variables) {
    if (v.name.empty()) {
      err << "Variable name must be non-empty. ";
      continue;
    }
    if (!names.insert(v.name).second) {
      err << "Duplicate variable name: " << v.name << ". ";
    }
    if (v.lower_bound > v.upper_bound) {
      err << "Invalid bounds for variable " << v.name << ". ";
    }
    if (v.type == VariableType::Binary) {
      if (v.lower_bound < 0.0 || v.upper_bound > 1.0) {
        err << "Binary variable " << v.name << " must be within [0,1]. ";
      }
    }
  }

  auto check_coeffs = [&](const std::unordered_map<std::string, double>& coeffs,
                          const std::string& where) {
    for (const auto& kv : coeffs) {
      if (names.find(kv.first) == names.end()) {
        err << "Unknown variable '" << kv.first << "' in " << where << ". ";
      }
      if (!std::isfinite(kv.second)) {
        err << "Non-finite coefficient for '" << kv.first << "' in " << where << ". ";
      }
    }
  };

  check_coeffs(model.objective.linear, "objective");
  for (const auto& row : model.objective.quadratic) {
    if (names.find(row.first) == names.end()) {
      err << "Unknown quadratic variable '" << row.first << "'. ";
    }
    check_coeffs(row.second, "quadratic objective");
  }

  for (const auto& c : model.constraints) {
    check_coeffs(c.linear, "constraint " + c.name);
    if (!std::isfinite(c.rhs)) {
      err << "Non-finite RHS in constraint " << c.name << ". ";
    }
  }

  if (model.problem_type == ProblemType::LP && !model.objective.quadratic.empty()) {
    err << "LP model must not contain quadratic objective terms. ";
  }
  if (model.problem_type == ProblemType::MILP) {
    // Allowed; integer/binary checked at variable type.
  }

  return err.str();
}

}  // namespace sovereign
