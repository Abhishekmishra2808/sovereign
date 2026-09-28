#include "mini_test.hpp"

#include "sovereign/cuts.hpp"
#include "sovereign/heuristics.hpp"
#include "sovereign/types.hpp"

#include <cmath>
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

static bool satisfies(const Constraint& c, const std::unordered_map<std::string, double>& p) {
  double lhs = 0.0;
  for (const auto& kv : c.linear) lhs += kv.second * p.at(kv.first);
  if (c.sense == ConstraintSense::Le) return lhs <= c.rhs + 1e-9;
  if (c.sense == ConstraintSense::Ge) return lhs >= c.rhs - 1e-9;
  return std::abs(lhs - c.rhs) <= 1e-9;
}

TEST(CutsTest, CoverCutAccountsForNegativeTerms) {
  // x1 + x2 - 2y <= 1: (1,1,1) is feasible, so the cover x1 + x2 <= 1 is invalid.
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.variables.push_back(Variable{"x1", VariableType::Binary, 0, 1});
  m.variables.push_back(Variable{"x2", VariableType::Binary, 0, 1});
  m.variables.push_back(Variable{"y", VariableType::Binary, 0, 1});
  m.constraints.push_back(Constraint{"link", {{"x1", 1}, {"x2", 1}, {"y", -2}}, ConstraintSense::Le, 1});
  const std::unordered_map<std::string, double> lp{{"x1", 0.9}, {"x2", 0.9}, {"y", 0.9}};
  const std::unordered_map<std::string, double> feasible{{"x1", 1}, {"x2", 1}, {"y", 1}};
  for (const auto& cut : generate_cover_cuts(m, lp)) EXPECT_TRUE(satisfies(cut.constraint, feasible));
}

TEST(CutsTest, RoundingCutKeepsNegativeContinuousTerms) {
  // 1.5x - y <= 0.5 with y >= 0 continuous: x = y = 1 is feasible, so "x <= 0" is invalid.
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.variables.push_back(Variable{"x", VariableType::Integer, 0, 5});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0, 1e30});
  m.constraints.push_back(Constraint{"row", {{"x", 1.5}, {"y", -1}}, ConstraintSense::Le, 0.5});
  const std::unordered_map<std::string, double> lp{{"x", 0.9}, {"y", 0.85}};
  const std::unordered_map<std::string, double> feasible{{"x", 1}, {"y", 1}};
  for (const auto& cut : generate_mir_cuts(m, lp)) EXPECT_TRUE(satisfies(cut.constraint, feasible));
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
