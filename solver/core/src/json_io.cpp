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

  const auto& variables = j.at("variables");
  model.variables.reserve(variables.size());
  for (const auto& vj : variables) {
    Variable v;
    v.name = vj.at("name").get<std::string>();
    v.type = variable_type_from_string(vj.value("type", "continuous"));
    v.lower_bound = vj.value("lower_bound", 0.0);
    v.upper_bound = vj.value("upper_bound", 1e30);
    if (v.type == VariableType::Binary) {
      v.lower_bound = vj.value("lower_bound", 0.0);
      v.upper_bound = vj.value("upper_bound", 1.0);
    }
    model.variables.push_back(std::move(v));
  }

  if (j.contains("objective")) {
    const auto& oj = j.at("objective");
    model.objective.constant = oj.value("constant", 0.0);
    if (oj.contains("linear")) {
      const auto& linear = oj.at("linear");
      model.objective.linear.reserve(linear.size());
      for (auto it = linear.begin(); it != linear.end(); ++it) {
        model.objective.linear[it.key()] = it.value().get<double>();
      }
    }
    if (oj.contains("quadratic")) {
      const auto& quadratic = oj.at("quadratic");
      model.objective.quadratic.reserve(quadratic.size());
      for (auto it = quadratic.begin(); it != quadratic.end(); ++it) {
        auto& row = model.objective.quadratic[it.key()];
        row.reserve(it.value().size());
        for (auto jt = it.value().begin(); jt != it.value().end(); ++jt) {
          row[jt.key()] = jt.value().get<double>();
        }
      }
    }
  }

  if (j.contains("constraints")) {
    const auto& constraints = j.at("constraints");
    model.constraints.reserve(constraints.size());
    for (const auto& cj : constraints) {
      Constraint c;
      c.name = cj.value("name", "");
      c.sense = constraint_sense_from_string(cj.at("sense").get<std::string>());
      c.rhs = cj.at("rhs").get<double>();
      if (cj.contains("linear")) {
        const auto& linear = cj.at("linear");
        c.linear.reserve(linear.size());
        for (auto it = linear.begin(); it != linear.end(); ++it) {
          c.linear[it.key()] = it.value().get<double>();
        }
      }
      model.constraints.push_back(std::move(c));
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
  j["objective"]["constant"] = model.objective.constant;
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
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open model file: " + path);
  }
  // Parsing a contiguous buffer is several times faster than nlohmann's
  // character-by-character stream adapter on files of hundreds of MB.
  std::string text;
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  if (size > 0) {
    text.resize(static_cast<std::size_t>(size));
    in.seekg(0, std::ios::beg);
    in.read(&text[0], static_cast<std::streamsize>(text.size()));
  }
  json j = json::parse(text);
  std::string().swap(text);
  return parse_model(j);
}

OptimizationModel load_model_from_json_string(const std::string& json_text) {
  return parse_model(json::parse(json_text));
}

std::string model_to_json_string(const OptimizationModel& model) {
  return model_to_json(model).dump(2);
}

json lp_diagnostics_to_json(const LpDiagnostics& d) {
  return json{
      {"scaling_applied", d.scaling_applied},
      {"coefficient_min_abs_before", d.coefficient_min_abs_before},
      {"coefficient_max_abs_before", d.coefficient_max_abs_before},
      {"coefficient_min_abs_after", d.coefficient_min_abs_after},
      {"coefficient_max_abs_after", d.coefficient_max_abs_after},
      {"degenerate_pivots", d.degenerate_pivots},
      {"refactorizations", d.refactorizations},
      {"objective_history_iterations", d.objective_history_iterations},
      {"objective_history", d.objective_history},
      {"gap_history_iterations", d.gap_history_iterations},
      {"gap_history", d.gap_history},
      {"mu_history", d.mu_history},
      {"primal_step_history", d.primal_step_history},
      {"dual_step_history", d.dual_step_history},
      {"stall_iteration", d.stall_iteration},
      {"stall_gap", d.stall_gap},
      {"stall_mu", d.stall_mu},
      {"stall_primal_step", d.stall_primal_step},
      {"stall_dual_step", d.stall_dual_step},
      {"final_primal_residual", d.final_primal_residual},
      {"final_dual_residual", d.final_dual_residual},
      {"final_gap", d.final_gap},
      {"basis_state", d.basis_state},
      {"stop_reason", d.stop_reason},
  };
}

json mip_diagnostics_to_json(const MipDiagnostics& d) {
  return json{
      {"has_best_bound", d.has_best_bound},
      {"best_bound", d.has_best_bound ? json(d.best_bound) : json(nullptr)},
      {"time_to_first_incumbent", d.time_to_first_incumbent >= 0.0
                                      ? json(d.time_to_first_incumbent)
                                      : json(nullptr)},
      {"cuts_by_family", d.cuts_by_family},
      {"presolve_fixed_variables", d.presolve_fixed_variables},
      {"presolve_substituted_variables", d.presolve_substituted_variables},
      {"presolve_removed_constraints", d.presolve_removed_constraints},
      {"presolve_tightened_bounds", d.presolve_tightened_bounds},
      {"presolve_passes", d.presolve_passes},
      {"node_lp_failures", d.node_lp_failures},
      {"dropped_subtrees", d.dropped_subtrees},
      {"numerical_error_nodes", d.numerical_error_nodes},
      {"iteration_limit_nodes", d.iteration_limit_nodes},
      {"unbounded_nodes", d.unbounded_nodes},
      {"cut_validity_rejections", d.cut_validity_rejections},
      {"debug_solution_enabled", d.debug_solution_enabled},
      {"debug_solution_violation", d.debug_solution_violation},
      {"debug_solution_checks", d.debug_solution_checks},
      {"debug_solution_path", d.debug_solution_path.empty() ? json(nullptr)
                                                               : json(d.debug_solution_path)},
      {"debug_solution_first_violation",
       d.debug_solution_first_violation.empty() ? json(nullptr)
                                                 : json(d.debug_solution_first_violation)},
  };
}

std::string result_to_json_string(const SolverResult& result, bool include_primal) {
  json j;
  j["status"] = to_string(result.status);
  if (result.has_objective_value) {
    j["objective_value"] = result.objective_value;
  } else {
    j["objective_value"] = nullptr;
  }
  if (include_primal) {
    j["primal"] = result.primal;
  } else {
    j["primal"] = nullptr;
    j["primal_variables"] = result.primal.size();
  }
  j["dual"] = result.dual;
  j["slacks"] = result.slacks;
  j["dual_certificate_space"] = result.dual_certificate_space.empty()
                                    ? nullptr
                                    : json(result.dual_certificate_space);
  j["lp_diagnostics"] = json::object();
  for (const auto& entry : result.lp_diagnostics) {
    j["lp_diagnostics"][entry.first] = lp_diagnostics_to_json(entry.second);
  }
  j["mip_diagnostics"] = mip_diagnostics_to_json(result.mip_diagnostics);
  j["optimality_gap"] = result.optimality_gap;
  // Certificates. A reader (or the dashboard) can now check the claim instead of
  // having to take the status string on faith.
  j["duality_gap"] = result.duality_gap;
  j["primal_residual"] = result.primal_residual;
  j["dual_residual"] = result.dual_residual;
  j["presolve"] = {
      {"fixed_variables", result.presolve_fixed_variables},
      {"substituted_variables", result.presolve_substituted_variables},
      {"removed_constraints", result.presolve_removed_constraints},
      {"tightened_bounds", result.presolve_tightened_bounds},
      {"passes", result.presolve_passes},
  };
  j["optimality_proven"] = is_conclusive(result.status);
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
