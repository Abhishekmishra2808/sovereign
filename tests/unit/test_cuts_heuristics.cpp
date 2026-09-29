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

TEST(CutsTest, CmirCutsNeverRemoveFeasiblePoints) {
  // Random single-row mixed-integer sets, checked exhaustively: every integer
  // point in the (capped) box, and for each one the extreme ends of the
  // continuous variable's feasible interval, must satisfy every cMIR cut.
  unsigned state = 12345u;
  auto rnd = [&state]() {
    state = state * 1664525u + 1013904223u;
    return (state >> 8) / static_cast<double>(1u << 24);
  };
  const double kInf = 1e30;
  const double lows[] = {0, 0, -2, 1, 0};
  const double highs[] = {1, 4, 3, 5, kInf};
  const double ylows[] = {0, -3, -kInf, 0};
  const double yhighs[] = {kInf, 4, 2, 1.5};
  int cuts_checked = 0;
  for (int trial = 0; trial < 3000; ++trial) {
    OptimizationModel m;
    m.problem_type = ProblemType::MILP;
    Constraint row{"r", {}, static_cast<ConstraintSense>(static_cast<int>(rnd() * 3) % 3), 0.0};
    std::unordered_map<std::string, double> lp;
    for (int j = 0; j < 3; ++j) {
      const int kind = static_cast<int>(rnd() * 5) % 5;
      const std::string name = "x" + std::to_string(j);
      m.variables.push_back(Variable{name, kind == 0 ? VariableType::Binary : VariableType::Integer,
                                     lows[kind], highs[kind]});
      row.linear[name] = std::round((rnd() * 14 - 7) * 4) / 4;
      const double hi = highs[kind] < kInf ? highs[kind] : lows[kind] + 6;
      lp[name] = lows[kind] + rnd() * (hi - lows[kind]);
    }
    const bool with_y = rnd() < 0.7;
    const int ykind = static_cast<int>(rnd() * 4) % 4;
    if (with_y) {
      m.variables.push_back(Variable{"y", VariableType::Continuous, ylows[ykind], yhighs[ykind]});
      row.linear["y"] = std::round((rnd() * 10 - 5) * 4) / 4;
      lp["y"] = std::max(ylows[ykind], std::min(yhighs[ykind], rnd() * 6 - 3));
    }
    row.rhs = std::round((rnd() * 20 - 8) * 8) / 8;
    m.constraints.push_back(row);

    for (const auto& cut : generate_cmir_cuts(m, lp, 1e-6, 8)) {
      ++cuts_checked;
      const auto& c = cut.constraint;
      EXPECT_TRUE(c.sense == ConstraintSense::Le);
      const double cy = with_y && c.linear.count("y") ? c.linear.at("y") : 0.0;
      double cap[3];
      for (int j = 0; j < 3; ++j) cap[j] = m.variables[j].upper_bound < kInf ? m.variables[j].upper_bound
                                                                            : m.variables[j].lower_bound + 6;
      for (double a = m.variables[0].lower_bound; a <= cap[0]; ++a)
        for (double b = m.variables[1].lower_bound; b <= cap[1]; ++b)
          for (double d = m.variables[2].lower_bound; d <= cap[2]; ++d) {
            const double xs[3] = {a, b, d};
            double rest = row.rhs;
            double lhs = 0.0;
            for (int j = 0; j < 3; ++j) {
              const std::string name = "x" + std::to_string(j);
              rest -= row.linear.at(name) * xs[j];
              if (c.linear.count(name)) lhs += c.linear.at(name) * xs[j];
            }
            double lo = with_y ? ylows[ykind] : 0.0;
            double hi = with_y ? yhighs[ykind] : 0.0;
            const double ay = with_y ? row.linear.at("y") : 0.0;
            // Feasible interval of y given the integer part: ay * y (sense) rest.
            const bool le_part = row.sense != ConstraintSense::Ge;
            const bool ge_part = row.sense != ConstraintSense::Le;
            if (ay == 0.0) {
              if ((le_part && rest < -1e-12) || (ge_part && rest > 1e-12)) continue;
            } else {
              const double t = rest / ay;
              if (le_part) (ay > 0 ? hi : lo) = ay > 0 ? std::min(hi, t) : std::max(lo, t);
              if (ge_part) (ay > 0 ? lo : hi) = ay > 0 ? std::max(lo, t) : std::min(hi, t);
            }
            if (lo > hi + 1e-12) continue;
            const double worst_y = cy > 0 ? hi : cy < 0 ? lo : 0.0;
            if (std::abs(worst_y) >= kInf) {
              EXPECT_TRUE(false);  // the cut cuts off an unbounded ray of feasible points
              continue;
            }
            EXPECT_TRUE(lhs + cy * worst_y <= c.rhs + 1e-7);
          }
    }
  }
  EXPECT_TRUE(cuts_checked > 200);
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
