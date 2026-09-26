#include "sovereign/json_io.hpp"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>

namespace sovereign {
namespace {

using json = nlohmann::json;

OptimizationModel parse_model(const json& j) {
  OptimizationModel model;
  model.problem_type = problem_type_from_string(j.at("problem_type").get<std::string>());
  model.sense = sense_from_string(j.value("sense", "minimize"));

  for (const auto& vj : j.at("variables")) {
    Variable v;
    v.name = vj.at("name").get<std::string>();
    v.type = variable_type_from_string(vj.value("type", "continuous"));
    v.lower_bound = vj.value("lower_bound", 0.0);
    v.upper_bound = vj.value("upper_bound", 1e30);
    if (v.type == VariableType::Binary) {
      v.lower_bound = vj.value("lower_bound", 0.0);
      v.upper_bound = vj.value("upper_bound", 1.0);
    }
    model.variables.push_back(v);
  }

  if (j.contains("objective")) {
    const auto& oj = j.at("objective");
    if (oj.contains("linear")) {
      for (auto it = oj.at("linear").begin(); it != oj.at("linear").end(); ++it) {
        model.objective.linear[it.key()] = it.value().get<double>();
      }
    }
    if (oj.contains("quadratic")) {
      for (auto it = oj.at("quadratic").begin(); it != oj.at("quadratic").end(); ++it) {
        for (auto jt = it.value().begin(); jt != it.value().end(); ++jt) {
          model.objective.quadratic[it.key()][jt.key()] = jt.value().get<double>();
        }
      }
    }
  }

  if (j.contains("constraints")) {
    for (const auto& cj : j.at("constraints")) {
      Constraint c;
      c.name = cj.value("name", "");
      c.sense = constraint_sense_from_string(cj.at("sense").get<std::string>());
      c.rhs = cj.at("rhs").get<double>();
      if (cj.contains("linear")) {
        for (auto it = cj.at("linear").begin(); it != cj.at("linear").end(); ++it) {
          c.linear[it.key()] = it.value().get<double>();
        }
      }
      model.constraints.push_back(c);
    }
  }

  return model;
}

json variable_to_json(const Variable& v) {
  return json{{"name", v.name},
              {"type", to_string(v.type)},
              {"lower_bound", v.lower_bound},
              {"upper_bound", v.upper_bound}};
}

json model_to_json(const OptimizationModel& model) {
  json j;
  j["problem_type"] = to_string(model.problem_type);
  j["sense"] = to_string(model.sense);
  j["variables"] = json::array();
  for (const auto& v : model.variables) {
    j["variables"].push_back(variable_to_json(v));
  }
  j["objective"] = json::object();
  j["objective"]["linear"] = model.objective.linear;
  if (!model.objective.quadratic.empty()) {
    j["objective"]["quadratic"] = model.objective.quadratic;
  }
  j["constraints"] = json::array();
  for (const auto& c : model.constraints) {
    j["constraints"].push_back(
        json{{"name", c.name},
             {"linear", c.linear},
             {"sense", to_string(c.sense)},
             {"rhs", c.rhs}});
  }
  return j;
}

}  // namespace

OptimizationModel load_model_from_json_file(const std::string& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Failed to open model file: " + path);
  }
  json j;
  in >> j;
  return parse_model(j);
}

OptimizationModel load_model_from_json_string(const std::string& json_text) {
  return parse_model(json::parse(json_text));
}

std::string model_to_json_string(const OptimizationModel& model) {
  return model_to_json(model).dump(2);
}

std::string result_to_json_string(const SolverResult& result) {
  json j;
  j["status"] = to_string(result.status);
  if (result.has_objective_value) {
    j["objective_value"] = result.objective_value;
  } else {
    j["objective_value"] = nullptr;
  }
  j["primal"] = result.primal;
  j["optimality_gap"] = result.optimality_gap;
  j["iterations"] = result.iterations;
  j["nodes"] = result.nodes;
  j["runtime_seconds"] = result.runtime_seconds;
  j["message"] = result.message;
  j["warnings"] = result.warnings;
  return j.dump(2);
}

std::string verification_to_json_string(const VerificationResult& result) {
  json j;
  j["is_valid"] = result.is_valid;
  j["max_constraint_violation"] = result.max_constraint_violation;
  j["max_bound_violation"] = result.max_bound_violation;
  j["max_integrality_violation"] = result.max_integrality_violation;
  j["recomputed_objective"] = result.recomputed_objective;
  j["issues"] = result.issues;
  j["message"] = result.message;
  return j.dump(2);
}

}  // namespace sovereign
