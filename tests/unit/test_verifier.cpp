#include "mini_test.hpp"

#include "sovereign/types.hpp"
#include "sovereign/verifier.hpp"

using namespace sovereign;

TEST(VerifierTest, RejectsNotImplemented) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1.0});
  SolverResult result;
  result.status = SolverStatus::NotImplemented;
  SolutionVerifier verifier;
  const VerificationResult vr = verifier.verify(model, result);
  EXPECT_FALSE(vr.is_valid);
}

TEST(VerifierTest, AcceptsFeasiblePoint) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.sense = Sense::Maximize;
  model.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 10.0});
  model.objective.linear["x"] = 3.0;
  Constraint c;
  c.name = "c1";
  c.linear["x"] = 1.0;
  c.sense = ConstraintSense::Le;
  c.rhs = 5.0;
  model.constraints.push_back(c);

  SolverResult result;
  result.status = SolverStatus::Optimal;
  result.has_objective_value = true;
  result.objective_value = 15.0;
  result.primal["x"] = 5.0;

  SolutionVerifier verifier;
  const VerificationResult vr = verifier.verify(model, result);
  EXPECT_TRUE(vr.is_valid);
  EXPECT_NEAR(vr.recomputed_objective, 15.0, 1e-9);
}
