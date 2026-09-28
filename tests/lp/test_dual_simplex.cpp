#include "mini_test.hpp"

#include "sovereign/dual_simplex.hpp"
#include "sovereign/revised_simplex.hpp"
#include "sovereign/types.hpp"

#include <algorithm>
#include <cstdint>
#include <string>

using namespace sovereign;

namespace {

Constraint row(const std::string& name, std::unordered_map<std::string, double> linear,
               ConstraintSense sense, double rhs) {
  Constraint c;
  c.name = name;
  c.linear = std::move(linear);
  c.sense = sense;
  c.rhs = rhs;
  return c;
}

// max 3x + 2y  s.t.  x + y <= 4,  2x + y <= 5,  x, y >= 0.  Optimum x=1, y=3, obj 9.
OptimizationModel classic() {
  OptimizationModel m;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear = {{"x", 3.0}, {"y", 2.0}};
  m.constraints = {row("c1", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Le, 4.0),
                   row("c2", {{"x", 2.0}, {"y", 1.0}}, ConstraintSense::Le, 5.0)};
  return m;
}

struct Lcg {
  std::uint64_t s;
  int next(int lo, int hi) {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return lo + static_cast<int>((s >> 33) % static_cast<std::uint64_t>(hi - lo + 1));
  }
};

}  // namespace

TEST(DualSimplexTest, ColdStartClassic) {
  LpBasis basis;
  SolverResult r = solve_lp_dual_simplex(classic(), nullptr, &basis);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 9.0, 1e-7);
  EXPECT_NEAR(r.primal.at("x"), 1.0, 1e-7);
  EXPECT_NEAR(r.primal.at("y"), 3.0, 1e-7);
  EXPECT_EQ(basis.cols.size(), 2u);
  EXPECT_EQ(basis.rows.size(), 2u);
}

TEST(DualSimplexTest, WarmStartAfterBranchingBound) {
  LpBasis parent;
  solve_lp_dual_simplex(classic(), nullptr, &parent);
  OptimizationModel child = classic();
  child.variables[0].upper_bound = 0.5;  // x <= 0.5  =>  x=0.5, y=3.5, obj 8.5
  SolverResult warm = solve_lp_dual_simplex(child, &parent, nullptr);
  SolverResult cold = solve_lp_dual_simplex(child, nullptr, nullptr);
  EXPECT_EQ(warm.status, SolverStatus::Optimal);
  EXPECT_NEAR(warm.objective_value, 8.5, 1e-7);
  EXPECT_NEAR(cold.objective_value, 8.5, 1e-7);
  EXPECT_TRUE(warm.iterations <= cold.iterations);
}

TEST(DualSimplexTest, WarmStartWithAppendedCutRow) {
  LpBasis parent;
  solve_lp_dual_simplex(classic(), nullptr, &parent);
  OptimizationModel cut = classic();
  cut.constraints.push_back(row("cut", {{"y", 1.0}}, ConstraintSense::Le, 2.0));  // x=1.5, y=2
  LpBasis after;
  SolverResult r = solve_lp_dual_simplex(cut, &parent, &after);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 8.5, 1e-7);
  EXPECT_EQ(after.rows.size(), 3u);
}

TEST(DualSimplexTest, CertifiesInfeasibility) {
  OptimizationModel m;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 2.0});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 2.0});
  m.objective.linear = {{"x", 1.0}};
  m.constraints = {row("need5", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Ge, 5.0)};
  SolverResult r = solve_lp_dual_simplex(m, nullptr, nullptr);
  EXPECT_EQ(r.status, SolverStatus::Infeasible);
}

TEST(DualSimplexTest, NeverClaimsOptimalOnUnboundedLp) {
  OptimizationModel m;
  m.sense = Sense::Maximize;
  m.variables.push_back(Variable{"x", VariableType::Continuous, 0.0, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 1e30});
  m.objective.linear = {{"x", 1.0}};
  m.constraints = {row("c", {{"x", 1.0}, {"y", -1.0}}, ConstraintSense::Le, 1.0)};
  SolverResult r = solve_lp_dual_simplex(m, nullptr, nullptr);
  EXPECT_TRUE(r.status != SolverStatus::Optimal);
  EXPECT_TRUE(r.status != SolverStatus::Infeasible);
}

TEST(DualSimplexTest, EqualityGeRowsAndFreeVariable) {
  // min x + 2y  s.t.  x + y = 3,  x - y >= -1,  x free,  0 <= y <= 10.  Optimum y=0, x=3.
  OptimizationModel m;
  m.variables.push_back(Variable{"x", VariableType::Continuous, -1e30, 1e30});
  m.variables.push_back(Variable{"y", VariableType::Continuous, 0.0, 10.0});
  m.objective.linear = {{"x", 1.0}, {"y", 2.0}};
  m.constraints = {row("eq", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Eq, 3.0),
                   row("ge", {{"x", 1.0}, {"y", -1.0}}, ConstraintSense::Ge, -1.0)};
  SolverResult r = solve_lp_dual_simplex(m, nullptr, nullptr);
  EXPECT_EQ(r.status, SolverStatus::Optimal);
  EXPECT_NEAR(r.objective_value, 3.0, 1e-7);
}

TEST(DualSimplexTest, MatchesPrimalSimplexOnRandomBoundedLps) {
  Lcg rng{12345};
  int compared = 0;
  for (int trial = 0; trial < 40; ++trial) {
    OptimizationModel m;
    m.sense = trial % 2 ? Sense::Maximize : Sense::Minimize;
    const int n = 6 + trial % 5, rows = 4 + trial % 4;
    for (int j = 0; j < n; ++j) {
      m.variables.push_back(Variable{"x" + std::to_string(j), VariableType::Continuous, 0.0,
                                     static_cast<double>(rng.next(1, 10))});
      m.objective.linear["x" + std::to_string(j)] = rng.next(-5, 5);
    }
    for (int i = 0; i < rows; ++i) {
      std::unordered_map<std::string, double> lin;
      for (int j = 0; j < n; ++j) {
        const int a = rng.next(-4, 6);
        if (a != 0) lin["x" + std::to_string(j)] = a;
      }
      const ConstraintSense s = i % 3 == 2 ? ConstraintSense::Ge : ConstraintSense::Le;
      m.constraints.push_back(row("r" + std::to_string(i), lin, s,
                                  s == ConstraintSense::Le ? rng.next(5, 30) : rng.next(-20, 2)));
    }

    LpBasis basis;
    SolverResult dual = solve_lp_dual_simplex(m, nullptr, &basis);
    SolverResult primal = RevisedSimplexSolver().solve(m);
    if (primal.status == SolverStatus::Infeasible) {
      EXPECT_TRUE(dual.status != SolverStatus::Optimal);
      continue;
    }
    if (primal.status != SolverStatus::Optimal) continue;
    EXPECT_EQ(dual.status, SolverStatus::Optimal);
    EXPECT_NEAR(dual.objective_value, primal.objective_value, 1e-6);
    ++compared;

    // Branch on the first variable and check the warm start against a cold solve.
    OptimizationModel child = m;
    child.variables[0].upper_bound = std::max(0.0, dual.primal.at("x0") - 0.5);
    SolverResult warm = solve_lp_dual_simplex(child, &basis, nullptr);
    SolverResult cold = RevisedSimplexSolver().solve(child);
    EXPECT_EQ(warm.status == SolverStatus::Infeasible, cold.status == SolverStatus::Infeasible);
    if (cold.status == SolverStatus::Optimal && warm.status == SolverStatus::Optimal) {
      EXPECT_NEAR(warm.objective_value, cold.objective_value, 1e-6);
    }
  }
  EXPECT_TRUE(compared >= 20);
}
