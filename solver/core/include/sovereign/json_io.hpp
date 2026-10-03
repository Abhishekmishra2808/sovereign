#pragma once

#include "sovereign/types.hpp"

#include <string>

namespace sovereign {

OptimizationModel load_model_from_json_file(const std::string& path);
OptimizationModel load_model_from_json_string(const std::string& json_text);
std::string model_to_json_string(const OptimizationModel& model);
std::string result_to_json_string(const SolverResult& result, bool include_primal = true);
std::string verification_to_json_string(const VerificationResult& result);

}  // namespace sovereign
