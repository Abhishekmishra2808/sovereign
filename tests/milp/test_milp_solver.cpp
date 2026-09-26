#include "mini_test.hpp"

#include "sovereign/branch_and_bound.hpp"
#include "sovereign/engine.hpp"
#include "sovereign/types.hpp"
#include "sovereign/verifier.hpp"

using namespace sovereign;

TEST(MilpSolveTest, BinaryKnapsack) {
  // max 5x + 3y + 2z
  // 4x + 3y + 2z <= 7
  // binary => opt (1,1,0) value 8
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Binary, 0.0, 1.0});
  m.variables.push_back(Variable{"y", VariableType::Binary, 0.0, 1.0});
  m.variables.push_back(Variable{"z", VariableType::Binary, 0.0, 1.0});
  m.objective.linear = {{"x", 5.0}, {"y", 3.0}, {"z", 2.0}};
  Constraint c{"cap", {{"x", 4.0}, {"y", 3.0}, {"z", 2.0}}, ConstraintSense::Le, 7.0};
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 8.0, 1e-5);
  EXPECT_NEAR(r.primal.at("x"), 1.0, 1e-6);
  EXPECT_NEAR(r.primal.at("y"), 1.0, 1e-6);
  EXPECT_NEAR(r.primal.at("z"), 0.0, 1e-6);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}

TEST(MilpSolveTest, IntegerRounding) {
  // min x  s.t. x >= 2.5, x integer, x <= 10 => x=3
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Integer, 0.0, 10.0});
  m.objective.linear["x"] = 1.0;
  Constraint c{"ge", {{"x", 1.0}}, ConstraintSense::Ge, 2.5};
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 3.0, 1e-6);
  EXPECT_NEAR(r.primal.at("x"), 3.0, 1e-6);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}

TEST(MilpSolveTest, InfeasibleBinary) {
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Binary, 0.0, 1.0});
  m.objective.linear["x"] = 1.0;
  Constraint c1{"ge", {{"x", 1.0}}, ConstraintSense::Ge, 2.0};
  m.constraints.push_back(c1);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Infeasible);
}

TEST(MilpSolveTest, MixedContinuousInteger) {
  // max 2x + y
  // x + y <= 3.5
  // x continuous >=0, y binary
  // Opt: x=3.5,y=0 value 7 OR x=2.5,y=1 value 6 => 7
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Binary, 0.0, 1.0});
  m.objective.linear = {{"x", 2.0}, {"y", 1.0}};
  Constraint c{"c", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Le, 3.5};
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 7.0, 1e-5);
  EXPECT_NEAR(r.primal.at("y"), 0.0, 1e-6);
  EXPECT_NEAR(r.primal.at("x"), 3.5, 1e-5);
  EXPECT_TRUE(SolutionVerifier().verify(m, r).is_valid);
}

TEST(MilpSolveTest, SmallAssignmentLike) {
  // min 3x + 4y
  // x + y = 1
  // binary => opt x=1,y=0 value 3
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Binary, 0.0, 1.0});
  m.variables.push_back(Variable{"y", VariableType::Binary, 0.0, 1.0});
  m.objective.linear = {{"x", 3.0}, {"y", 4.0}};
  Constraint c{"eq", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Eq, 1.0};
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 3.0, 1e-6);
  EXPECT_NEAR(r.primal.at("x"), 1.0, 1e-6);
  EXPECT_NEAR(r.primal.at("y"), 0.0, 1e-6);
}

TEST(MilpNearTieTest, AcceptBetterIntegerEvenWhenWithinMipGap) {
  // Regression for integer-feasibility-BEFORE-bound-prune.
  //
  // min 100 x + 100 y + 100.05 z
  //     x + y + z >= 2
  //     x,y,z binary
  //
  // Integer solutions include (1,1,0)=200 (true opt) and (1,0,1)=(0,1,1)=200.05.
  // Relative gap (200.05 - 200) / 200.05 ≈ 2.5e-4.
  // With mip_gap = 1e-3, a near-miss incumbent of 200.05 would make the true
  // optimum look "within gap." If bound-prune ran before the integrality
  // check, the solver could keep 200.05 and never record 200. The correct
  // order accepts any strictly better integer-feasible node first.
  OptimizationModel m;
  m.problem_type = ProblemType::MILP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Binary, 0.0, 1.0});
  m.variables.push_back(Variable{"y", VariableType::Binary, 0.0, 1.0});
  m.variables.push_back(Variable{"z", VariableType::Binary, 0.0, 1.0});
  m.objective.linear = {{"x", 100.0}, {"y", 100.0}, {"z", 100.05}};
  Constraint c{"cover", {{"x", 1.0}, {"y", 1.0}, {"z", 1.0}}, ConstraintSense::Ge, 2.0};
  m.constraints.push_back(c);

  BranchAndBoundOptions opt;
  opt.mip_gap = 1e-3;
  opt.enable_cuts = false;
  opt.enable_heuristics = false;
  opt.branch_rule = BranchRule::MostFractional;
  opt.parallel_workers = 1;
  opt.max_nodes = 10000;

  SolverResult r = BranchAndBoundSolver(opt).solve(m);
  EXPECT_TRUE(r.status == SolverStatus::Optimal || r.status == SolverStatus::Feasible);
  EXPECT_NEAR(r.objective_value, 200.0, 1e-4);
  EXPECT_NEAR(r.primal.at("x") + r.primal.at("y"), 2.0, 1e-5);
  EXPECT_NEAR(r.primal.at("z"), 0.0, 1e-5);
  EXPECT_TRUE(SolutionVerifier().verify(m, r, 1e-4).is_valid);
}
