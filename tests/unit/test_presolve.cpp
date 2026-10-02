#include "mini_test.hpp"

#include "sovereign/engine.hpp"
#include "sovereign/presolve.hpp"
#include "sovereign/types.hpp"
#include "sovereign/verifier.hpp"

using namespace sovereign;

TEST(PresolveTest, FixesEqualBounds) {
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 3.0, 3.0});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 10.0});
  m.objective.linear = {{"x", 2.0}, {"y", 1.0}};
  Constraint c{"c", {{"y", 1.0}}, ConstraintSense::Ge, 1.0};
  m.constraints.push_back(c);

  PresolveResult p = Presolver().run(m);
  EXPECT_FALSE(p.infeasible);
  EXPECT_TRUE(static_cast<int>(p.reduced.variables.size()) <= 1);
  EXPECT_TRUE(p.stats.fixed_variables >= 1);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), 3.0, 1e-8);
  EXPECT_NEAR(r.primal.at("y"), 1.0, 1e-5);
  EXPECT_NEAR(r.objective_value, 2.0 * 3.0 + 1.0, 1e-5);
}

TEST(PresolveTest, SingletonEqualityFixesVar) {
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 10.0});
  m.objective.linear["x"] = 1.0;
  Constraint c{"eq", {{"x", 2.0}}, ConstraintSense::Eq, 6.0};  // x = 3
  m.constraints.push_back(c);

  PresolveResult p = Presolver().run(m);
  EXPECT_FALSE(p.infeasible);
  EXPECT_TRUE(p.reduced.variables.empty() || p.stats.fixed_variables >= 1);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), 3.0, 1e-8);
  EXPECT_NEAR(r.objective_value, 3.0, 1e-8);
}

TEST(PresolveTest, DetectsInfeasibleBounds) {
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1.0});
  m.objective.linear["x"] = 1.0;
  Constraint c1{"ge", {{"x", 1.0}}, ConstraintSense::Ge, 5.0};
  m.constraints.push_back(c1);

  PresolveResult p = Presolver().run(m);
  EXPECT_TRUE(p.infeasible);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Infeasible);
}

TEST(PresolveTest, RemovesRedundantConstraint) {
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 5.0});
  m.objective.linear["x"] = 1.0;
  Constraint tight{"tight", {{"x", 1.0}}, ConstraintSense::Le, 5.0};
  Constraint loose{"loose", {{"x", 1.0}}, ConstraintSense::Le, 100.0};
  m.constraints = {tight, loose};

  PresolveResult p = Presolver().run(m);
  EXPECT_FALSE(p.infeasible);
  EXPECT_TRUE(p.stats.removed_constraints >= 1);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 5.0, 1e-5);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}

TEST(PresolveTest, EqualitySubstitution) {
  // min x + y  s.t. x + y = 4, x,y >= 0  => substitute x = 4 - y
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear = {{"x", 1.0}, {"y", 1.0}};
  Constraint eq{"eq", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Eq, 4.0};
  m.constraints.push_back(eq);

  PresolveResult p = Presolver().run(m);
  EXPECT_FALSE(p.infeasible);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 4.0, 1e-5);
  EXPECT_NEAR(r.primal.at("x") + r.primal.at("y"), 4.0, 1e-5);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}

TEST(PresolveTest, FixingLaterVariableKeepsEarlierNames) {
  // c is fixed; a and b come before it and must survive compaction intact.
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"a", VariableType::Continuous, 0.0, 10.0});
  m.variables.push_back(Variable{"b", VariableType::Continuous, 0.0, 10.0});
  m.variables.push_back(Variable{"c", VariableType::Continuous, 2.0, 2.0});
  m.objective.linear = {{"a", 1.0}, {"b", 2.0}};
  m.constraints.push_back(Constraint{"sum", {{"a", 1.0}, {"b", 1.0}, {"c", 1.0}}, ConstraintSense::Eq, 7.0});
  m.constraints.push_back(Constraint{"cap", {{"a", 1.0}, {"b", -1.0}}, ConstraintSense::Le, 3.0});

  PresolveResult p = Presolver().run(m);
  EXPECT_FALSE(p.infeasible);
  for (const auto& v : p.reduced.variables) EXPECT_FALSE(v.name.empty());

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("a"), 4.0, 1e-5);
  EXPECT_NEAR(r.primal.at("b"), 1.0, 1e-5);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}

TEST(PresolveTest, SubstitutionKeepsUpperBoundOfEliminatedVariable) {
  // 2x - y = 0 eliminates x = y/2; x <= 3 must become y <= 6 even though y has
  // no upper bound of its own, or the recovered x leaves its range.
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 3.0});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"z", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear = {{"x", -1.0}, {"z", 1.0}};
  m.constraints.push_back(Constraint{"link", {{"x", 2.0}, {"y", -1.0}}, ConstraintSense::Eq, 0.0});
  m.constraints.push_back(Constraint{"use", {{"y", 1.0}, {"z", -1.0}}, ConstraintSense::Le, 100.0});

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), 3.0, 1e-5);
  EXPECT_NEAR(r.objective_value, -3.0, 1e-5);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}

TEST(PresolveTest, UnconstrainedDualFix) {
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 2.0, 9.0});
  m.objective.linear["x"] = 5.0;  // no constraints => fix at lb=2
  // need a dummy constraint-free model - validator allows empty constraints

  PresolveResult p = Presolver().run(m);
  EXPECT_FALSE(p.infeasible);
  EXPECT_FALSE(p.unbounded);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), 2.0, 1e-8);
  EXPECT_NEAR(r.objective_value, 10.0, 1e-8);
}

TEST(PresolveTest, ExistingLpStillWorks) {
  OptimizationModel m;
  m.problem_type = ProblemType::LP;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear = {{"x", 3.0}, {"y", 2.0}};
  Constraint c1{"c1", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Le, 4.0};
  Constraint c2{"c2", {{"x", 2.0}, {"y", 1.0}}, ConstraintSense::Le, 5.0};
  m.constraints = {c1, c2};

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 9.0, 1e-5);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}
