#include "mini_test.hpp"

#include "sovereign/model_validator.hpp"
#include "sovereign/types.hpp"

using namespace sovereign;

TEST(ValidatorTest, AcceptsValidModel) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1.0});
  model.objective.linear["x"] = 1.0;
  Constraint c;
  c.name = "c1";
  c.linear["x"] = 1.0;
  c.sense = ConstraintSense::Le;
  c.rhs = 1.0;
  model.constraints.push_back(c);
  EXPECT_TRUE(ModelValidator::validate(model).empty());
}

TEST(ValidatorTest, RejectsUnknownVariable) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1.0});
  model.objective.linear["y"] = 1.0;
  const std::string err = ModelValidator::validate(model);
  EXPECT_FALSE(err.empty());
  EXPECT_NE(err.find("Unknown variable"), std::string::npos);
}
