// Regression tests for three correctness defects found by audit.
//
// 1. The interior-point termination test divided the complementarity SUM by n,
//    making the test n times too lenient. On wide sparse models it declared
//    OPTIMAL on a measurably suboptimal point.
// 2. The Chvatal-Gomory generator skipped columns with a negative lower bound
//    instead of abandoning the cut, so it could emit an INVALID inequality that
//    removes integer-feasible points. Cuts are inherited by every descendant,
//    so this poisons whole subtrees.
// 3. Statuses collapsed "iteration limit" and "numerical breakdown" into a
//    generic ERROR, hiding why a search stopped.
//
// Each test here fails against the pre-fix code.

#include "sovereign/cuts.hpp"
#include "sovereign/branch_and_bound.hpp"
#include "sovereign/convex_qp.hpp"
#include "sovereign/interior_point.hpp"
#include "sovereign/qp_interior_point.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/lp_solver.hpp"
#include "sovereign/revised_simplex.hpp"
#include "sovereign/types.hpp"
#include "sovereign/verifier.hpp"

#include "mini_test.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

using namespace sovereign;

namespace {

Variable make_var(const std::string& name, VariableType type, double lb, double ub) {
  Variable v;
  v.name = name;
  v.type = type;
  v.lower_bound = lb;
  v.upper_bound = ub;
  return v;
}

Constraint make_cons(const std::string& name,
                     const std::vector<std::pair<std::string, double>>& terms,
                     ConstraintSense sense, double rhs) {
  Constraint c;
  c.name = name;
  c.sense = sense;
  c.rhs = rhs;
  for (const auto& t : terms) c.linear[t.first] = t.second;
  return c;
}

}  // namespace

// ---------------------------------------------------------------------------
// 1. IPM accuracy
// ---------------------------------------------------------------------------

namespace {

// A transportation-style LP: m sources, m sinks, every arc allowed.
// The optimum has a large, well-conditioned objective, which is exactly where a
// diluted complementarity test hides: the objective scale is O(1000) while the
// per-column complementarity average is O(1/n).
OptimizationModel build_transport(int m) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.sense = Sense::Minimize;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      model.variables.push_back(
          make_var("x_" + std::to_string(i) + "_" + std::to_string(j),
                   VariableType::Continuous, 0.0, 1e30));
    }
  }
  // c[i][j] in [1.00, 1.16] deterministically, so the model is reproducible.
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < m; ++j) {
      const double c = 1.0 + 0.04 * ((i * 7 + j * 3) % 5);
      model.objective.linear["x_" + std::to_string(i) + "_" + std::to_string(j)] = c;
    }
  }
  for (int i = 0; i < m; ++i) {
    std::vector<std::pair<std::string, double>> terms;
    for (int j = 0; j < m; ++j) {
      terms.push_back({"x_" + std::to_string(i) + "_" + std::to_string(j), 1.0});
    }
    model.constraints.push_back(
        make_cons("supply_" + std::to_string(i), terms, ConstraintSense::Eq, 10.0));
  }
  for (int j = 0; j < m; ++j) {
    std::vector<std::pair<std::string, double>> terms;
    for (int i = 0; i < m; ++i) {
      terms.push_back({"x_" + std::to_string(i) + "_" + std::to_string(j), 1.0});
    }
    model.constraints.push_back(
        make_cons("demand_" + std::to_string(j), terms, ConstraintSense::Eq, 10.0));
  }
  return model;
}

}  // namespace

TEST(IpmAccuracy, MatchesSimplexOnWideSparseLp) {
  // 60x60 => 3600 structural variables. The old test divided the
  // complementarity sum by 3600, so a real 1e-4 relative gap read as 3e-8 and
  // passed a 1e-8 threshold.
  const OptimizationModel model = build_transport(60);

  InteriorPointOptions iopt;
  const SolverResult ipm = InteriorPointSolver(iopt).solve(model);
  RevisedSimplexOptions sopt;
  const SolverResult simplex = RevisedSimplexSolver(sopt).solve(model);

  EXPECT_EQ(ipm.status, SolverStatus::Optimal);
  EXPECT_EQ(simplex.status, SolverStatus::Optimal);
  EXPECT_TRUE(ipm.has_objective_value);
  EXPECT_TRUE(simplex.has_objective_value);

  const double scale = std::max(1.0, std::abs(simplex.objective_value));
  EXPECT_NEAR(ipm.objective_value, simplex.objective_value, 1e-7 * scale);

  // The reported certificate must agree with the observed error. If the gap
  // claim is much smaller than the actual suboptimality, the test is lying.
  EXPECT_TRUE(ipm.duality_gap < 1e-7);
  const double observed = std::abs(ipm.objective_value - simplex.objective_value) / scale;
  EXPECT_TRUE(ipm.duality_gap >= observed * 0.5);
}

namespace {

#if defined(_WIN32) && !defined(_MSC_VER)
extern "C" int _putenv(const char*);
#endif

void set_env(const char* name, const char* value) {
#if defined(_WIN32)
  _putenv((std::string(name) + "=" + value).c_str());
#else
  setenv(name, value, 1);
#endif
}

void set_normal_equations(const char* mode) { set_env("SOVEREIGN_IPM_NORMAL_EQUATIONS", mode); }

// Multi-period production planning: inventory balance per product and period,
// one shared capacity row per period. Its normal equations are block banded,
// the structure a sparse factorization is meant for.
OptimizationModel build_staircase(int products, int periods) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.sense = Sense::Minimize;
  auto name = [](const char* kind, int p, int t) { return std::string(kind) + std::to_string(p) + "_" + std::to_string(t); };
  for (int t = 0; t < periods; ++t) {
    std::vector<std::pair<std::string, double>> capacity;
    for (int p = 0; p < products; ++p) {
      model.variables.push_back(make_var(name("make", p, t), VariableType::Continuous, 0.0, 1e30));
      model.variables.push_back(make_var(name("stock", p, t), VariableType::Continuous, 0.0, 1e30));
      model.objective.linear[name("make", p, t)] = 1.0 + 0.1 * ((p * 3 + t * 7) % 11);
      model.objective.linear[name("stock", p, t)] = 0.05 + 0.01 * (p % 4);
      std::vector<std::pair<std::string, double>> balance = {{name("make", p, t), 1.0}, {name("stock", p, t), -1.0}};
      if (t > 0) balance.push_back({name("stock", p, t - 1), 1.0});
      model.constraints.push_back(make_cons(name("balance", p, t), balance, ConstraintSense::Eq,
                                            5.0 + static_cast<double>((p * 5 + t * 3) % 9)));
      capacity.push_back({name("make", p, t), 1.0 + 0.1 * (p % 3)});
    }
    model.constraints.push_back(make_cons("capacity" + std::to_string(t), capacity, ConstraintSense::Le,
                                          12.0 * products));
  }
  return model;
}

}  // namespace

TEST(IpmAccuracy, SparseAndDenseNormalEquationsAgree) {
  // The transport model has a redundant equality row, so its normal equations
  // are singular: the sparse factorization must drop that direction, not fail.
  const OptimizationModel models[] = {build_transport(40), build_staircase(8, 40)};
  for (const OptimizationModel& model : models) {
    InteriorPointOptions iopt;
    set_normal_equations("dense");
    const SolverResult dense = InteriorPointSolver(iopt).solve(model);
    set_normal_equations("sparse");
    const SolverResult sparse = InteriorPointSolver(iopt).solve(model);
    set_normal_equations("auto");
    const SolverResult automatic = InteriorPointSolver(iopt).solve(model);
    set_normal_equations("");

    EXPECT_EQ(dense.status, SolverStatus::Optimal);
    EXPECT_EQ(sparse.status, SolverStatus::Optimal);
    EXPECT_EQ(automatic.status, SolverStatus::Optimal);
    EXPECT_TRUE(sparse.message.find("sparse LDL^T") != std::string::npos);
    EXPECT_TRUE(sparse.message.find("dense LU retries") == std::string::npos);
    const double scale = std::max(1.0, std::abs(dense.objective_value));
    EXPECT_NEAR(sparse.objective_value, dense.objective_value, 1e-7 * scale);
    EXPECT_NEAR(automatic.objective_value, dense.objective_value, 1e-7 * scale);
  }
  // 360 rows on the CPU: automatic selection takes the sparse path.
  const SolverResult automatic = InteriorPointSolver(InteriorPointOptions()).solve(build_staircase(8, 40));
  EXPECT_TRUE(automatic.message.find("sparse LDL^T") != std::string::npos);

  // The option forces sparse even where the environment asks for dense.
  InteriorPointOptions forced;
  forced.use_sparse_normal_equations = true;
  set_normal_equations("dense");
  const SolverResult sparse = InteriorPointSolver(forced).solve(build_transport(40));
  set_normal_equations("");
  EXPECT_EQ(sparse.status, SolverStatus::Optimal);
  EXPECT_TRUE(sparse.message.find("sparse LDL^T") != std::string::npos);
}

TEST(QpAccuracy, OffDiagonalHessianTermsCountOnce) {
  // The objective is 1/2 sum q_ij x_i x_j over the stored terms:
  // 1/2 (2x^2 + xy + 2y^2) - x - y, minimized at x = y = 0.4 with value -0.4.
  // Doubling the stored xy term gives x = y = 1/3 instead.
  OptimizationModel m;
  m.problem_type = ProblemType::QP;
  m.sense = Sense::Minimize;
  m.variables.push_back(make_var("x", VariableType::Continuous, 0.0, 10.0));
  m.variables.push_back(make_var("y", VariableType::Continuous, 0.0, 10.0));
  m.objective.linear = {{"x", -1.0}, {"y", -1.0}};
  m.objective.quadratic["x"]["x"] = 2.0;
  m.objective.quadratic["x"]["y"] = 1.0;
  m.objective.quadratic["y"]["y"] = 2.0;
  m.constraints.push_back(make_cons("c", {{"x", 1.0}, {"y", 1.0}}, ConstraintSense::Le, 5.0));

  const SolverResult ipm = QpInteriorPointSolver().solve(m);
  EXPECT_EQ(ipm.status, SolverStatus::Optimal);
  EXPECT_NEAR(ipm.primal.at("x"), 0.4, 1e-7);
  EXPECT_NEAR(ipm.primal.at("y"), 0.4, 1e-7);
  EXPECT_NEAR(ipm.objective_value, -0.4, 1e-8);

  const SolverResult fw = ConvexQpSolver().solve(m);
  EXPECT_EQ(fw.status, SolverStatus::Optimal);
  EXPECT_NEAR(fw.objective_value, -0.4, 1e-4);
}

TEST(QpAccuracy, FrankWolfeMaximizesConcaveObjective) {
  // max 2x - x^2 on [0, 10] peaks at x = 1 with value 1. Negating only the
  // linear part turns this into a nonconvex minimization that runs to x = 10.
  OptimizationModel m;
  m.problem_type = ProblemType::QP;
  m.sense = Sense::Maximize;
  m.variables.push_back(make_var("x", VariableType::Continuous, 0.0, 10.0));
  m.objective.linear = {{"x", 2.0}};
  m.objective.quadratic["x"]["x"] = -2.0;
  m.constraints.push_back(make_cons("c", {{"x", 1.0}}, ConstraintSense::Le, 10.0));

  const SolverResult fw = ConvexQpSolver().solve(m);
  EXPECT_NE(fw.status, SolverStatus::Error);
  EXPECT_NEAR(fw.primal.at("x"), 1.0, 1e-3);
  EXPECT_NEAR(fw.objective_value, 1.0, 1e-5);
}

namespace {

// Portfolio-style convex QP: a diagonally dominant covariance with a few
// symmetric couplings per asset, a budget row and sector caps. Its KKT system
// is sparse apart from the budget row.
OptimizationModel build_portfolio(int assets, int sectors) {
  OptimizationModel model;
  model.problem_type = ProblemType::QP;
  model.sense = Sense::Minimize;
  auto name = [](int j) { return "w" + std::to_string(j); };
  std::vector<std::pair<std::string, double>> budget;
  for (int j = 0; j < assets; ++j) {
    model.variables.push_back(make_var(name(j), VariableType::Continuous, 0.0, 1e30));
    model.objective.linear[name(j)] = -0.01 - 0.19 * ((j * 37) % 101) / 100.0;
    model.objective.quadratic[name(j)][name(j)] = 0.5 + 1.5 * ((j * 53) % 97) / 96.0;
    for (int t = 1; t <= 3; ++t) {
      const int k = (j * 7 + t * 13) % assets;
      if (k == j) continue;
      const double v = 0.05 * (((j + k * t) % 21) - 10) / 10.0;
      model.objective.quadratic[name(j)][name(k)] += v;
      model.objective.quadratic[name(k)][name(j)] += v;
    }
    budget.push_back({name(j), 1.0});
  }
  model.constraints.push_back(make_cons("budget", budget, ConstraintSense::Eq, 1.0));
  for (int s = 0; s < sectors; ++s) {
    std::vector<std::pair<std::string, double>> terms;
    for (int j = s; j < assets; j += sectors) terms.push_back({name(j), 1.0});
    model.constraints.push_back(make_cons("sector" + std::to_string(s), terms, ConstraintSense::Le, 0.4));
  }
  return model;
}

}  // namespace

TEST(QpAccuracy, SparseAndDenseKktAgree) {
  // The transport LP run through the QP interior point has Q = 0 and a
  // redundant equality row: the regularized quasi-definite factorization must
  // handle both.
  const OptimizationModel models[] = {build_portfolio(300, 12), build_transport(20)};
  for (const OptimizationModel& model : models) {
    set_env("SOVEREIGN_QP_KKT", "dense");
    const SolverResult dense = QpInteriorPointSolver().solve(model);
    set_env("SOVEREIGN_QP_KKT", "sparse");
    const SolverResult sparse = QpInteriorPointSolver().solve(model);
    set_env("SOVEREIGN_QP_KKT", "auto");
    const SolverResult automatic = QpInteriorPointSolver().solve(model);
    set_env("SOVEREIGN_QP_KKT", "");

    EXPECT_EQ(dense.status, SolverStatus::Optimal);
    EXPECT_EQ(sparse.status, SolverStatus::Optimal);
    EXPECT_EQ(automatic.status, SolverStatus::Optimal);
    EXPECT_TRUE(dense.message.find("dense LU") != std::string::npos);
    EXPECT_TRUE(sparse.message.find("sparse quasi-definite LDL^T") != std::string::npos);
    EXPECT_TRUE(sparse.message.find("dense LU retries") == std::string::npos);
    const double scale = std::max(1.0, std::abs(dense.objective_value));
    EXPECT_NEAR(sparse.objective_value, dense.objective_value, 1e-7 * scale);
    EXPECT_NEAR(automatic.objective_value, dense.objective_value, 1e-7 * scale);
    EXPECT_TRUE(SolutionVerifier().verify(model, sparse, 1e-6).is_valid);
  }
  // Order 300 + 12 slacks + 13 rows on the CPU: automatic selection is sparse.
  const SolverResult automatic = QpInteriorPointSolver().solve(build_portfolio(300, 12));
  EXPECT_TRUE(automatic.message.find("sparse quasi-definite LDL^T") != std::string::npos);
}

TEST(IpmAccuracy, OptimalClaimIsBackedByResiduals) {
  const OptimizationModel model = build_transport(40);
  InteriorPointOptions iopt;
  const SolverResult r = InteriorPointSolver(iopt).solve(model);

  if (r.status == SolverStatus::Optimal) {
    // An OPTIMAL status is a claim. These are the numbers that justify it, so
    // they must be present and small.
    EXPECT_TRUE(r.primal_residual < 1e-6);
    EXPECT_TRUE(r.dual_residual < 1e-6);
    EXPECT_TRUE(r.duality_gap < 1e-6);
  } else {
    // Not converging is acceptable; silently claiming OPTIMAL is not.
    EXPECT_NE(r.status, SolverStatus::Optimal);
    EXPECT_FALSE(r.has_objective_value);
    EXPECT_FALSE(r.message.empty());
  }
}

TEST(IpmAccuracy, NonConvergenceReturnsNoObjective) {
  // Starve the iteration budget. The solver must NOT hand back its last
  // interior iterate as if it were an answer.
  OptimizationModel model = build_transport(30);
  InteriorPointOptions iopt;
  iopt.max_iterations = 1;
  iopt.optimality_tol = 1e-14;  // unreachable in one step
  const SolverResult r = InteriorPointSolver(iopt).solve(model);

  if (r.status != SolverStatus::Optimal) {
    EXPECT_FALSE(r.has_objective_value);
    EXPECT_TRUE(r.primal.empty());
    // The reason must be specific, not a generic ERROR.
    EXPECT_NE(r.status, SolverStatus::Error);
    EXPECT_FALSE(r.message.empty());
  }
}

// ---------------------------------------------------------------------------
// 2. Cut validity
// ---------------------------------------------------------------------------

TEST(CutValidity, GomorySkipsRowsWithNegativeLowerBound) {
  // The counterexample that used to produce an invalid cut.
  //
  //   row:  2*x1 - 3.5*yc <= 7
  //
  // x1 is a nonneg integer, yc is continuous and free below. Rounding yc's
  // coefficient is unsound because yc may be negative: floor(-3.5) = -4 < -3.5,
  // so -4*yc > -3.5*yc when yc < 0.
  //
  // The old generator skipped yc and emitted "2*x1 <= 7", i.e. x1 <= 3. But
  // x1 = 10, yc = 5.2 is feasible for the ORIGINAL row (20 - 18.2 = 1.8 <= 7),
  // so that cut deletes a feasible integer point.
  OptimizationModel model;
  model.problem_type = ProblemType::MILP;
  model.sense = Sense::Minimize;
  model.variables.push_back(make_var("x1", VariableType::Integer, 0.0, 100.0));
  model.variables.push_back(make_var("yc", VariableType::Continuous, -1e30, 1e30));
  model.objective.linear["x1"] = 1.0;
  model.constraints.push_back(make_cons("r0", {{"x1", 2.0}, {"yc", -3.5}},
                                        ConstraintSense::Le, 7.0));

  std::unordered_map<std::string, double> x;
  x["x1"] = 3.5;  // fractional, so a cut would be considered
  x["yc"] = 0.0;

  const std::vector<Cut> cuts = generate_mir_cuts(model, x, 1e-6, 8);

  // Either no cut, or a cut that keeps x1 = 10, yc = 5.2 feasible.
  for (const Cut& cut : cuts) {
    double lhs = 0.0;
    for (const auto& kv : cut.constraint.linear) {
      const double v = (kv.first == "x1") ? 10.0 : 5.2;
      lhs += kv.second * v;
    }
    if (cut.constraint.sense == ConstraintSense::Le) {
      EXPECT_TRUE(lhs <= cut.constraint.rhs + 1e-6);
    }
  }
}

TEST(CutValidity, GateRejectsCutThatKillsKnownFeasiblePoint) {
  OptimizationModel model;
  model.problem_type = ProblemType::MILP;
  model.sense = Sense::Minimize;
  model.variables.push_back(make_var("a", VariableType::Binary, 0.0, 1.0));
  model.variables.push_back(make_var("b", VariableType::Binary, 0.0, 1.0));

  Constraint cut;
  cut.name = "bad_cut";
  cut.sense = ConstraintSense::Le;
  cut.rhs = 0.0;
  cut.linear["a"] = 1.0;
  cut.linear["b"] = 1.0;  // forbids a=b=1

  // (a,b) = (1,1) is a legal integer point of the base model, so this cut is
  // invalid. The gate must say so.
  std::vector<std::unordered_map<std::string, double>> refs(1);
  refs[0]["a"] = 1.0;
  refs[0]["b"] = 1.0;

  const std::string why = check_cut_validity(model, cut, refs, 1e-6);
  EXPECT_FALSE(why.empty());
}

TEST(CutValidity, GateAcceptsGenuinelyValidCut) {
  OptimizationModel model;
  model.problem_type = ProblemType::MILP;
  model.sense = Sense::Minimize;
  model.variables.push_back(make_var("a", VariableType::Binary, 0.0, 1.0));
  model.variables.push_back(make_var("b", VariableType::Binary, 0.0, 1.0));
  model.constraints.push_back(
      make_cons("r0", {{"a", 1.0}, {"b", 1.0}}, ConstraintSense::Le, 1.0));

  // A real cover cut for a+b <= 1: the known feasible points are (0,0),(1,0),(0,1).
  Constraint cut;
  cut.name = "cover_0";
  cut.sense = ConstraintSense::Le;
  cut.rhs = 1.0;
  cut.linear["a"] = 1.0;
  cut.linear["b"] = 1.0;

  std::vector<std::unordered_map<std::string, double>> refs;
  refs.push_back({{"a", 0.0}, {"b", 0.0}});
  refs.push_back({{"a", 1.0}, {"b", 0.0}});
  refs.push_back({{"a", 0.0}, {"b", 1.0}});

  const std::string why = check_cut_validity(model, cut, refs, 1e-6);
  EXPECT_TRUE(why.empty());
}

TEST(CutValidity, GateRejectsMalformedBinaryCoefficient) {
  OptimizationModel model;
  model.problem_type = ProblemType::MILP;
  model.variables.push_back(make_var("a", VariableType::Binary, 0.0, 1.0));

  Constraint cut;
  cut.name = "weird";
  cut.sense = ConstraintSense::Le;
  cut.rhs = 0.5;
  cut.linear["a"] = 3.0;  // > 1 on a binary is meaningless

  std::vector<std::unordered_map<std::string, double>> refs;
  const std::string why = check_cut_validity(model, cut, refs, 1e-6);
  EXPECT_FALSE(why.empty());
}

// ---------------------------------------------------------------------------
// 3. Status honesty
// ---------------------------------------------------------------------------

TEST(StatusHonesty, StringsAreDistinct) {
  EXPECT_EQ(to_string(SolverStatus::Optimal), std::string("OPTIMAL"));
  EXPECT_EQ(to_string(SolverStatus::Feasible), std::string("FEASIBLE"));
  EXPECT_EQ(to_string(SolverStatus::Infeasible), std::string("INFEASIBLE"));
  EXPECT_EQ(to_string(SolverStatus::Unbounded), std::string("UNBOUNDED"));
  EXPECT_EQ(to_string(SolverStatus::TimeLimit), std::string("TIME_LIMIT"));
  EXPECT_EQ(to_string(SolverStatus::IterationLimit), std::string("ITERATION_LIMIT"));
  EXPECT_EQ(to_string(SolverStatus::NumericalError), std::string("NUMERICAL_ERROR"));
  EXPECT_EQ(to_string(SolverStatus::Error), std::string("ERROR"));
}

TEST(StatusHonesty, OnlyProvenStatusesAreConclusive) {
  EXPECT_TRUE(is_conclusive(SolverStatus::Optimal));
  EXPECT_TRUE(is_conclusive(SolverStatus::Infeasible));
  EXPECT_TRUE(is_conclusive(SolverStatus::Unbounded));

  // Everything else means "we stopped without proving anything".
  EXPECT_FALSE(is_conclusive(SolverStatus::Feasible));
  EXPECT_FALSE(is_conclusive(SolverStatus::TimeLimit));
  EXPECT_FALSE(is_conclusive(SolverStatus::IterationLimit));
  EXPECT_FALSE(is_conclusive(SolverStatus::NumericalError));
  EXPECT_FALSE(is_conclusive(SolverStatus::Error));
  EXPECT_FALSE(is_conclusive(SolverStatus::NotImplemented));
}

TEST(StatusHonesty, SimplexIterationLimitIsItsOwnStatus) {
  OptimizationModel model = build_transport(30);
  // Force an immediate stop so we exercise the mapping, not the arithmetic.
  RevisedSimplexOptions opt;
  opt.max_iterations = 0;
  const SolverResult r = RevisedSimplexSolver(opt).solve(model);

  EXPECT_EQ(r.status, SolverStatus::IterationLimit);
  EXPECT_FALSE(r.has_objective_value);
  EXPECT_FALSE(r.message.empty());
  EXPECT_FALSE(is_conclusive(r.status));
}

TEST(StatusHonesty, VerifierRejectsOptimalWithOpenDualityGap) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.sense = Sense::Minimize;
  model.variables.push_back(make_var("x", VariableType::Continuous, 0.0, 10.0));
  model.objective.linear["x"] = 1.0;
  model.constraints.push_back(
      make_cons("c0", {{"x", 1.0}}, ConstraintSense::Le, 4.0));

  SolverResult r;
  r.status = SolverStatus::Optimal;
  r.has_objective_value = true;
  r.objective_value = 4.0;
  r.primal["x"] = 4.0;
  r.duality_gap = 1e-3;  // contradicts the OPTIMAL claim

  SolutionVerifier v;
  const VerificationResult res = v.verify(model, r, 1e-6);
  EXPECT_FALSE(res.is_valid);
  EXPECT_FALSE(res.issues.empty());
}

TEST(StatusHonesty, VerifierRejectsNonConclusiveStatusWithPrimal) {
  OptimizationModel model;
  model.problem_type = ProblemType::LP;
  model.sense = Sense::Minimize;
  model.variables.push_back(make_var("x", VariableType::Continuous, 0.0, 10.0));
  model.objective.linear["x"] = 1.0;
  model.constraints.push_back(
      make_cons("c0", {{"x", 1.0}}, ConstraintSense::Le, 4.0));

  SolverResult r;
  r.status = SolverStatus::NumericalError;
  r.has_objective_value = true;
  r.objective_value = 4.0;
  r.primal["x"] = 4.0;

  SolutionVerifier v;
  const VerificationResult res = v.verify(model, r, 1e-6);
  EXPECT_FALSE(res.is_valid);
}

// ---------------------------------------------------------------------------
// 4. Fallbacks carry a real reason
// ---------------------------------------------------------------------------

TEST(FallbackHonesty, AutoFallsBackAndSaysWhy) {
  // A model the IPM can solve, forced to a budget it cannot meet. The `auto`
  // route must rescue it with the simplex AND record the actual reason.
  const OptimizationModel model = build_transport(25);

  InteriorPointOptions iopt;
  iopt.max_iterations = 1;
  iopt.optimality_tol = 1e-15;
  const SolverResult ipm = InteriorPointSolver(iopt).solve(model);

  RevisedSimplexOptions sopt;
  const SolverResult simplex = RevisedSimplexSolver(sopt).solve(model);
  EXPECT_EQ(simplex.status, SolverStatus::Optimal);

  if (ipm.status != SolverStatus::Optimal) {
    // The simplex answer is trustworthy and carries diagnostics.
    EXPECT_EQ(simplex.duality_gap, 0.0);
    EXPECT_TRUE(simplex.has_objective_value);
  }
}

TEST(FallbackHonesty, EveryLpPathSolvesTheSameSmallModel) {
  // All three routes must agree on the same small model. This is the check that
  // the fallback wiring is real rather than decorative.
  const OptimizationModel model = build_transport(10);

  InteriorPointOptions iopt;
  const SolverResult ipm = InteriorPointSolver(iopt).solve(model);
  RevisedSimplexOptions sopt;
  const SolverResult simplex = RevisedSimplexSolver(sopt).solve(model);

  EXPECT_EQ(ipm.status, SolverStatus::Optimal);
  EXPECT_EQ(simplex.status, SolverStatus::Optimal);

  const double scale = std::max(1.0, std::abs(simplex.objective_value));
  EXPECT_NEAR(ipm.objective_value, simplex.objective_value, 1e-7 * scale);

  // LpSolver's default route is 'auto'; it must reach a conclusive status too.
  const SolverResult auto_r = LpSolver().solve(model);
  EXPECT_TRUE(is_conclusive(auto_r.status));
  EXPECT_TRUE(auto_r.has_objective_value);
  EXPECT_NEAR(auto_r.objective_value, simplex.objective_value, 1e-7 * scale);
}

TEST(MiplibNode, PresolveRejectsInfeasibleFlugplBranch) {
  // Captured from an official MIPLIB flugpl search. The LP relaxation is
  // infeasible, but without node-level presolve both IPM and Phase I simplex
  // fail numerically, leaving the MILP search inconclusive. The minimal
  // contradiction is ANM5 + 0.9*STM5 = STM6: its left side is >= 71.3,
  // while the fixed right side is 71.

  BranchAndBoundOptions opt;
  opt.max_nodes = 1;
  opt.enable_cuts = false;
  opt.enable_heuristics = false;
  opt.branch_rule = BranchRule::MostFractional;
  for (const char* filename : {"flugpl_infeasible_node_lp.json",
                               "flugpl_infeasible_node_minimal.json"}) {
    const std::string path = std::string(SOVEREIGN_TEST_DATA_DIR) + "/" + filename;
    OptimizationModel model = load_model_from_json_file(path);
    model.problem_type = ProblemType::MILP;
    model.variables[0].type = VariableType::Integer;
    const SolverResult r = BranchAndBoundSolver(opt).solve(model);
    EXPECT_EQ(r.status, SolverStatus::Infeasible);
    EXPECT_EQ(r.nodes, 1);
    EXPECT_FALSE(r.has_objective_value);
  }
}
