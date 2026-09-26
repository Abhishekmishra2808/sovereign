#include "mini_test.hpp"

#include "sovereign/engine.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/types.hpp"
#include "sovereign/verifier.hpp"

using namespace sovereign;

static OptimizationModel make_model(ProblemType type, Sense sense) {
  OptimizationModel m;
  m.problem_type = type;
  m.sense = sense;
  return m;
}

TEST(LpSolveTest, ClassicTwoVarMaximize) {
  // max 3x + 2y
  // x + y <= 4
  // 2x + y <= 5
  // x,y >= 0
  // Optimum: x=1, y=3, obj=9  (intersection of x+y=4 and 2x+y=5)
  OptimizationModel m = make_model(ProblemType::LP, Sense::Maximize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear["x"] = 3.0;
  m.objective.linear["y"] = 2.0;
  Constraint c1;
  c1.name = "c1";
  c1.linear = {{"x", 1.0}, {"y", 1.0}};
  c1.sense = ConstraintSense::Le;
  c1.rhs = 4.0;
  Constraint c2;
  c2.name = "c2";
  c2.linear = {{"x", 2.0}, {"y", 1.0}};
  c2.sense = ConstraintSense::Le;
  c2.rhs = 5.0;
  m.constraints = {c1, c2};

  OptimizationEngine engine;
  SolverResult r = engine.solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_TRUE(r.has_objective_value);
  EXPECT_NEAR(r.objective_value, 9.0, 1e-5);
  if (r.primal.count("x") && r.primal.count("y")) {
    EXPECT_NEAR(r.primal.at("x"), 1.0, 1e-5);
    EXPECT_NEAR(r.primal.at("y"), 3.0, 1e-5);
  } else {
    EXPECT_TRUE(false);
  }

  SolutionVerifier v;
  VerificationResult vr = v.verify(m, r);
  EXPECT_TRUE(vr.is_valid);
}

TEST(LpSolveTest, MinimizeWithEquality) {
  // min x + y
  // x + y = 2
  // x,y >= 0
  // Optimum: any feasible, obj=2; simplex finds a vertex e.g. (2,0) or (0,2)
  OptimizationModel m = make_model(ProblemType::LP, Sense::Minimize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear["x"] = 1.0;
  m.objective.linear["y"] = 1.0;
  Constraint c;
  c.name = "eq";
  c.linear = {{"x", 1.0}, {"y", 1.0}};
  c.sense = ConstraintSense::Eq;
  c.rhs = 2.0;
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 2.0, 1e-5);
  EXPECT_NEAR(r.primal.at("x") + r.primal.at("y"), 2.0, 1e-5);
}

TEST(LpSolveTest, Infeasible) {
  OptimizationModel m = make_model(ProblemType::LP, Sense::Minimize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear["x"] = 1.0;
  Constraint c1;
  c1.name = "ge";
  c1.linear = {{"x", 1.0}};
  c1.sense = ConstraintSense::Ge;
  c1.rhs = 5.0;
  Constraint c2;
  c2.name = "le";
  c2.linear = {{"x", 1.0}};
  c2.sense = ConstraintSense::Le;
  c2.rhs = 1.0;
  m.constraints = {c1, c2};

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Infeasible);
}

TEST(LpSolveTest, Unbounded) {
  OptimizationModel m = make_model(ProblemType::LP, Sense::Maximize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear["x"] = 1.0;
  Constraint c;
  c.name = "ge0";
  c.linear = {{"x", 1.0}};
  c.sense = ConstraintSense::Ge;
  c.rhs = 0.0;
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Unbounded);
}

TEST(LpSolveTest, Degenerate) {
  // Degenerate feasible region tip: max x
  // x <= 1, x <= 1 (duplicate), x >= 0
  OptimizationModel m = make_model(ProblemType::LP, Sense::Maximize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear["x"] = 1.0;
  Constraint c1{"c1", {{"x", 1.0}}, ConstraintSense::Le, 1.0};
  Constraint c2{"c2", {{"x", 1.0}}, ConstraintSense::Le, 1.0};
  Constraint c3{"c3", {{"x", 1.0}}, ConstraintSense::Le, 1.0};
  m.constraints = {c1, c2, c3};

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 1.0, 1e-5);
  EXPECT_NEAR(r.primal.at("x"), 1.0, 1e-5);
}

TEST(LpSolveTest, ScaledCoefficients) {
  // max x
  // 1e6 x <= 1e6  => x <= 1
  OptimizationModel m = make_model(ProblemType::LP, Sense::Maximize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear["x"] = 1.0;
  Constraint c{"c", {{"x", 1e6}}, ConstraintSense::Le, 1e6};
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 1.0, 1e-4);
  EXPECT_NEAR(r.primal.at("x"), 1.0, 1e-4);
}

TEST(LpSolveTest, SparseDietLike) {
  // min 0.18x1 + 0.23x2 + 0.05x3
  // 107x1 + 500x2 + 0x3 >= 500
  // etc. simplified:
  // min x + 2y + 3z
  // x + y >= 2
  // y + z >= 2
  // x,y,z >= 0
  // Opt: x=2,y=0,z=2 obj=8 OR x=0,y=2,z=0 obj=4 -> optimum 4 at (0,2,0)
  OptimizationModel m = make_model(ProblemType::LP, Sense::Minimize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"z", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear = {{"x", 1.0}, {"y", 2.0}, {"z", 3.0}};
  Constraint c1{"c1", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Ge, 2.0};
  Constraint c2{"c2", {{"y", 1.0}, {"z", 1.0}}, ConstraintSense::Ge, 2.0};
  m.constraints = {c1, c2};

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 4.0, 1e-5);
  EXPECT_NEAR(r.primal.at("y"), 2.0, 1e-5);

  SolutionVerifier v;
  EXPECT_TRUE(v.verify(m, r).is_valid);
}

TEST(LpSolveTest, UpperBounds) {
  // max x + y, 0 <= x <= 2, 0 <= y <= 3, x+y <= 4
  // Opt: x=2,y=2 obj=4 or x=1,y=3 obj=4
  OptimizationModel m = make_model(ProblemType::LP, Sense::Maximize);
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 2.0});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 3.0});
  m.objective.linear = {{"x", 1.0}, {"y", 1.0}};
  Constraint c{"c", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Le, 4.0};
  m.constraints.push_back(c);

  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 4.0, 1e-5);
}

TEST(LpSolveTest, SampleJsonModel) {
  const std::string json = R"({
    "problem_type": "LP",
    "sense": "maximize",
    "variables": [
      {"name": "x", "type": "continuous", "lower_bound": 0, "upper_bound": 10},
      {"name": "y", "type": "continuous", "lower_bound": 0, "upper_bound": 10}
    ],
    "objective": {"linear": {"x": 3, "y": 2}},
    "constraints": [
      {"name": "resource", "linear": {"x": 1, "y": 1}, "sense": "<=", "rhs": 10}
    ]
  })";
  OptimizationModel m = load_model_from_json_string(json);
  SolverResult r = OptimizationEngine().solve(m);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 30.0, 1e-5);
  EXPECT_NEAR(r.primal.at("x"), 10.0, 1e-5);
  EXPECT_NEAR(r.primal.at("y"), 0.0, 1e-5);
}
