#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

class SolutionVerifier {
 public:
  VerificationResult verify(const OptimizationModel& model,
                            const SolverResult& result,
                            double tol = 1e-6) const;
};

}  // namespace sovereign
