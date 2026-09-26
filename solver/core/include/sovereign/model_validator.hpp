#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

class ModelValidator {
 public:
  // Returns empty string if model is structurally valid; otherwise an error message.
  static std::string validate(const OptimizationModel& model);
};

}  // namespace sovereign
