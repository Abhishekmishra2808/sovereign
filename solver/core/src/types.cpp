#include "sovereign/types.hpp"

#include <stdexcept>

namespace sovereign {

std::string to_string(ProblemType type) {
  switch (type) {
    case ProblemType::LP:
      return "LP";
    case ProblemType::QP:
      return "QP";
    case ProblemType::MILP:
      return "MILP";
  }
  return "UNKNOWN";
}

std::string to_string(Sense sense) {
  switch (sense) {
    case Sense::Minimize:
      return "minimize";
    case Sense::Maximize:
      return "maximize";
  }
  return "unknown";
}

std::string to_string(ConstraintSense sense) {
  switch (sense) {
    case ConstraintSense::Le:
      return "<=";
    case ConstraintSense::Ge:
      return ">=";
    case ConstraintSense::Eq:
      return "=";
  }
  return "?";
}

std::string to_string(VariableType type) {
  switch (type) {
    case VariableType::Continuous:
      return "continuous";
    case VariableType::Integer:
      return "integer";
    case VariableType::Binary:
      return "binary";
  }
  return "unknown";
}

std::string to_string(SolverStatus status) {
  switch (status) {
    case SolverStatus::Optimal:
      return "OPTIMAL";
    case SolverStatus::Feasible:
      return "FEASIBLE";
    case SolverStatus::Infeasible:
      return "INFEASIBLE";
    case SolverStatus::Unbounded:
      return "UNBOUNDED";
    case SolverStatus::Error:
      return "ERROR";
    case SolverStatus::NotImplemented:
      return "NOT_IMPLEMENTED";
  }
  return "UNKNOWN";
}

ProblemType problem_type_from_string(const std::string& s) {
  if (s == "LP" || s == "lp") return ProblemType::LP;
  if (s == "QP" || s == "qp") return ProblemType::QP;
  if (s == "MILP" || s == "milp") return ProblemType::MILP;
  throw std::invalid_argument("Unknown problem_type: " + s);
}

Sense sense_from_string(const std::string& s) {
  if (s == "minimize" || s == "min") return Sense::Minimize;
  if (s == "maximize" || s == "max") return Sense::Maximize;
  throw std::invalid_argument("Unknown sense: " + s);
}

ConstraintSense constraint_sense_from_string(const std::string& s) {
  if (s == "<=" || s == "le" || s == "LE") return ConstraintSense::Le;
  if (s == ">=" || s == "ge" || s == "GE") return ConstraintSense::Ge;
  if (s == "=" || s == "==" || s == "eq" || s == "EQ") return ConstraintSense::Eq;
  throw std::invalid_argument("Unknown constraint sense: " + s);
}

VariableType variable_type_from_string(const std::string& s) {
  if (s == "continuous") return VariableType::Continuous;
  if (s == "integer") return VariableType::Integer;
  if (s == "binary") return VariableType::Binary;
  throw std::invalid_argument("Unknown variable type: " + s);
}

}  // namespace sovereign
