#include "mini_test.hpp"

#include "sovereign/engine.hpp"
#include "sovereign/types.hpp"

using namespace sovereign;

static OptimizationModel make_lp() {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.sense = Sense::Maximize;
  model.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 10.0});
  model.objective.linear["x"] = 1.0;
  Constraint c;
  c.name = "c1";
  c.linear["x"] = 1.0;
  c.sense = ConstraintSense::Le;
  c.rhs = 5.0;
  model.constraints.push_back(c);
  return model;
}

TEST(EngineStubTest, LpSolvesSimpleModel) {
  OptimizationEngine engine;
  const SolverResult result = engine.solve(make_lp());
  EXPECT_EQ(result.status, SolverStatus::Optimal);
  EXPECT_TRUE(result.has_objective_value);
  EXPECT_NEAR(result.objective_value, 5.0, 1e-5);
}

TEST(EngineStubTest, QpSolvesBoundConstrained) {
  OptimizationModel model = make_lp();
  model.problem_type = ProblemType::QP;
  model.sense = Sense::Minimize;
  model.objective.linear["x"] = -6.0;
  model.objective.quadratic["x"]["x"] = 2.0;
  model.constraints.clear();
  model.variables[0].upper_bound = 10.0;
  OptimizationEngine engine;
  const SolverResult result = engine.solve(model);
  EXPECT_EQ(result.status, SolverStatus::Optimal);
  EXPECT_NEAR(result.primal.at("x"), 3.0, 5e-2);
}

TEST(EngineStubTest, MilpSolvesBinaryKnapsack) {
  OptimizationModel model;
  model.problem_type = ProblemType::MILP;
  model.sense = Sense::Maximize;
  model.variables.push_back(Variable{"x", VariableType::Binary, 0.0, 1.0});
  model.variables.push_back(Variable{"y", VariableType::Binary, 0.0, 1.0});
  model.objective.linear = {{"x", 5.0}, {"y", 3.0}};
  Constraint c{"cap", {{"x", 4.0}, {"y", 3.0}}, ConstraintSense::Le, 5.0};
  model.constraints.push_back(c);
  OptimizationEngine engine;
  const SolverResult result = engine.solve(model);
  EXPECT_EQ(result.status, SolverStatus::Optimal);
  EXPECT_NEAR(result.objective_value, 5.0, 1e-5);
}

TEST(EngineStubTest, InvalidModelReturnsError) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  OptimizationEngine engine;
  const SolverResult result = engine.solve(model);
  EXPECT_EQ(result.status, SolverStatus::Error);
}
