#include "mini_test.hpp"

#include "sovereign/json_io.hpp"
#include "sovereign/types.hpp"

using namespace sovereign;

TEST(SchemaTest, RoundTripMinimalLp) {
  const std::string json = R"({
    "problem_type": "LP",
    "sense": "maximize",
    "variables": [
      {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": 10},
      {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": 10}
    ],
    "objective": {"linear": {"x": 3, "y": 2}},
    "constraints": [
      {"name": "c1", "linear": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10}
    ]
  })";

  const OptimizationModel model = load_model_from_json_string(json);
  EXPECT_EQ(model.problem_type, ProblemType::LP);
  EXPECT_EQ(model.sense, Sense::Maximize);
  ASSERT_EQ(model.variables.size(), 2u);
  EXPECT_EQ(model.variables[0].name, "x");
  EXPECT_EQ(model.objective.linear.at("x"), 3.0);
  ASSERT_EQ(model.constraints.size(), 1u);
  EXPECT_EQ(model.constraints[0].sense, ConstraintSense::Le);

  const std::string dumped = model_to_json_string(model);
  const OptimizationModel again = load_model_from_json_string(dumped);
  EXPECT_EQ(again.variables.size(), model.variables.size());
  EXPECT_EQ(again.constraints.size(), model.constraints.size());
}

TEST(SchemaTest, StatusStrings) {
  EXPECT_EQ(to_string(SolverStatus::NotImplemented), "NOT_IMPLEMENTED");
  EXPECT_EQ(to_string(ProblemType::MILP), "MILP");
}
