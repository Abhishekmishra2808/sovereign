#include "mini_test.hpp"

#include "sovereign/engine.hpp"
#include "sovereign/qp_interior_point.hpp"
#include "sovereign/types.hpp"
#include "sovereign/verifier.hpp"

#include <string>

using namespace sovereign;

TEST(QpSolveTest, BoundConstrainedQuadratic) {
  // min (x-3)^2 = x^2 - 6x + 9  => 1/2 * 2 x^2 - 6x
  // 0 <= x <= 10  => opt x=3, obj = -9 from linear+quad part without constant
  // We store objective as 1/2 * Qxx x^2 + c x with Qxx=2, c=-6
  // Reported obj = 0.5*2*9 + (-6)*3 = 9 - 18 = -9
  OptimizationModel m;
  m.problem_type = ProblemType::QP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 10.0});
  m.objective.linear["x"] = -6.0;
  m.objective.quadratic["x"]["x"] = 2.0;

  SolverResult r = QpInteriorPointSolver().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), 3.0, 1e-2);
  EXPECT_NEAR(r.objective_value, -9.0, 5e-2);
  EXPECT_TRUE(r.message.find("Mehrotra") != std::string::npos);
  EXPECT_TRUE(SolutionVerifier().verify(m, r, 1e-3).is_valid);

  // Default engine path also prefers IPM.
  SolverResult r2 = OptimizationEngine().solve(m);
  EXPECT_EQ(r2.status, SolverStatus::Optimal);
  EXPECT_NEAR(r2.primal.at("x"), 3.0, 1e-2);
}

TEST(QpSolveTest, QpWithLinearConstraint) {
  // min x^2 + y^2  s.t. x + y >= 2, x,y >= 0 => opt (1,1) obj=2
  // Q = diag(2,2) for 1/2 x'Qx => 1/2*2 x^2 = x^2
  OptimizationModel m;
  m.problem_type = ProblemType::QP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.objective.quadratic["x"]["x"] = 2.0;
  m.objective.quadratic["y"]["y"] = 2.0;
  Constraint c{"ge", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Ge, 2.0};
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x") + r.primal.at("y"), 2.0, 5e-2);
  EXPECT_NEAR(r.objective_value, 2.0, 1e-1);
}

TEST(QpSolveTest, PortfolioStyleTwoAsset) {
  // min 0.5*[x y] [2 0; 0 2] [x;y] - [1 1][x;y] = x^2+y^2 - x - y
  // s.t. x+y = 1, x,y >= 0  => opt (0.5,0.5), obj = 0.5 - 1 = -0.5
  // Q diag entries 2 for 1/2 x'Qx form.
  OptimizationModel m;
  m.problem_type = ProblemType::QP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1.0});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1.0});
  m.objective.linear = {{"x", -1.0}, {"y", -1.0}};
  m.objective.quadratic["x"]["x"] = 2.0;
  m.objective.quadratic["y"]["y"] = 2.0;
  Constraint c{"eq", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Eq, 1.0};
  m.constraints.push_back(c);

  SolverResult r = QpInteriorPointSolver().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), 0.5, 5e-2);
  EXPECT_NEAR(r.primal.at("y"), 0.5, 5e-2);
  EXPECT_NEAR(r.objective_value, -0.5, 1e-1);
  EXPECT_TRUE(r.message.find("Mehrotra") != std::string::npos);
}

TEST(QpSolveTest, FreeVariablesInEqualityConstrainedQp) {
  // min 1/2 (x^2 + y^2) + z  s.t. x + y = -4, z - x = 0, all free
  // => z = x, so minimize 1/2 x^2 + 1/2 y^2 + x with y = -4 - x:
  //    x - (-4 - x) + 1 = 0 => x = -2.5, y = -1.5, obj = 3.125 + 1.125 - 2.5 = 1.75
  OptimizationModel m;
  m.problem_type = ProblemType::QP;
  m.sense = Sense::Minimize;
  for (const char* n : {"x", "y", "z"}) m.variables.push_back(Variable{n, VariableType::Continuous, -1e30, 1e30});
  m.objective.linear["z"] = 1.0;
  m.objective.quadratic["x"]["x"] = 1.0;
  m.objective.quadratic["y"]["y"] = 1.0;
  m.constraints.push_back(Constraint{"sum", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Eq, -4.0});
  m.constraints.push_back(Constraint{"link", {{"z", 1.0}, {"x", -1.0}}, ConstraintSense::Eq, 0.0});

  SolverResult r = QpInteriorPointSolver().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), -2.5, 1e-5);
  EXPECT_NEAR(r.primal.at("y"), -1.5, 1e-5);
  EXPECT_NEAR(r.primal.at("z"), -2.5, 1e-5);
  EXPECT_NEAR(r.objective_value, 1.75, 1e-6);
  EXPECT_EQ(r.primal.size(), static_cast<std::size_t>(3));
}

TEST(QpSolveTest, UpperBoundOnlyVariable) {
  // min 1/2 x^2 - 5x, x <= 2 with no lower bound => x = 2, obj = -8
  OptimizationModel m;
  m.problem_type = ProblemType::QP;
  m.sense = Sense::Minimize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, -1e30, 2.0});
  m.objective.linear["x"] = -5.0;
  m.objective.quadratic["x"]["x"] = 1.0;

  SolverResult r = QpInteriorPointSolver().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.primal.at("x"), 2.0, 1e-5);
  EXPECT_NEAR(r.objective_value, -8.0, 1e-5);
}
