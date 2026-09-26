#include "mini_test.hpp"

#include "sovereign/cuts.hpp"
#include "sovereign/heuristics.hpp"
#include "sovereign/types.hpp"

#include <unordered_map>

using namespace sovereign;

TEST(CutsTest, CoverCutGenerated) {
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.variables.push_back(Variable{"x", VariableType::Binary, 0, 1});
  m.variables.push_back(Variable{"y", VariableType::Binary, 0, 1});
  m.variables.push_back(Variable{"z", VariableType::Binary, 0, 1});
  Constraint c{"cap", {{"x", 4}, {"y", 3}, {"z", 2}}, ConstraintSense::Le, 5};
  m.constraints.push_back(c);
  std::unordered_map<std::string, double> x{{"x", 0.8}, {"y", 0.7}, {"z", 0.6}};
  auto cuts = generate_cover_cuts(m, x);
  EXPECT_TRUE(cuts.size() >= 1u);
}

TEST(HeuristicsTest, RoundingFindsBinary) {
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Binary, 0, 1});
  m.objective.linear["x"] = 1.0;
  Constraint c{"c", {{"x", 1.0}}, ConstraintSense::Le, 1.0};
  m.constraints.push_back(c);
  auto h = rounding_heuristic(m, {{"x", 0.7}});
  EXPECT_TRUE(h.found);
  EXPECT_NEAR(h.primal.at("x"), 1.0, 1e-9);
}
