#include "sovereign/branch_and_bound.hpp"

#include "sovereign/cuts.hpp"
#include "sovereign/dual_simplex.hpp"
#include "sovereign/heuristics.hpp"
#include "sovereign/lp_solver.hpp"
#include "sovereign/json_io.hpp"
#include "sovereign/presolve.hpp"
#include "sovereign/revised_simplex.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace sovereign {
namespace {

double now_seconds() {
#if defined(_WIN32)
  LARGE_INTEGER frequency, now;
  QueryPerformanceFrequency(&frequency);
  QueryPerformanceCounter(&now);
  return static_cast<double>(now.QuadPart) / static_cast<double>(frequency.QuadPart);
#else
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

struct BoundChange {
  int var = -1;
  double lower = 0.0;
  double upper = 0.0;
};

// Open nodes are kept small: they share their parent's model (which already
// carries any cuts added on the path) and store only the branching bounds on
// top of it. A full model per open node exhausts memory on large trees.
struct SearchNode {
  std::shared_ptr<const OptimizationModel> model;
  std::vector<BoundChange> bounds;
  double bound = 0.0;
  int depth = 0;
  std::int64_t id = 0;
  // Optimal basis of the parent's LP; shared by both children.
  std::shared_ptr<const LpBasis> basis;
  // The branching that created this node, so its LP can update pseudocosts.
  int branch_var = -1;
  bool branch_up = false;
  double branch_distance = 0.0;
  double parent_obj = 0.0;
};

struct PseudoCostStats {
  double down_sum = 0.0;
  double up_sum = 0.0;
  int down_count = 0;
  int up_count = 0;
};

bool is_integer_type(VariableType t) {
  return t == VariableType::Integer || t == VariableType::Binary;
}

OptimizationModel make_lp_relaxation(const OptimizationModel& milp) {
  OptimizationModel lp = milp;
  lp.problem_type = ProblemType::LP;
  for (auto& v : lp.variables) {
    if (v.type == VariableType::Binary) {
      v.lower_bound = std::max(0.0, v.lower_bound);
      v.upper_bound = std::min(1.0, v.upper_bound);
    }
    v.type = VariableType::Continuous;
  }
  return lp;
}

void attach_presolve_stats(SolverResult& result, const PresolveStats& stats) {
  result.presolve_fixed_variables += stats.fixed_variables;
  result.presolve_substituted_variables += stats.substituted_variables;
  result.presolve_removed_constraints += stats.removed_constraints;
  result.presolve_tightened_bounds += stats.tightened_bounds;
  result.presolve_passes += stats.passes;
}

bool is_integer_feasible(const OptimizationModel& milp,
                         const std::unordered_map<std::string, double>& x,
                         double tol) {
  for (const auto& v : milp.variables) {
    if (!is_integer_type(v.type)) continue;
    auto it = x.find(v.name);
    if (it == x.end()) return false;
    if (std::abs(it->second - std::round(it->second)) > tol) return false;
  }
  return true;
}

void snap_integer_primal(const OptimizationModel& milp,
                         std::unordered_map<std::string, double>& x, double tol) {
  for (const auto& v : milp.variables) {
    if (!is_integer_type(v.type)) continue;
    auto it = x.find(v.name);
    if (it == x.end()) continue;
    if (std::abs(it->second - std::round(it->second)) <= 10 * tol) {
      it->second = std::round(it->second);
    }
  }
}

std::vector<int> fractional_vars(const OptimizationModel& milp,
                                 const std::unordered_map<std::string, double>& x,
                                 double tol) {
  std::vector<int> out;
  for (std::size_t i = 0; i < milp.variables.size(); ++i) {
    if (!is_integer_type(milp.variables[i].type)) continue;
    auto it = x.find(milp.variables[i].name);
    if (it == x.end()) continue;
    if (std::abs(it->second - std::round(it->second)) > tol) {
      out.push_back(static_cast<int>(i));
    }
  }
  return out;
}

int pick_most_fractional(const OptimizationModel& milp,
                         const std::unordered_map<std::string, double>& x,
                         double tol) {
  int best = -1;
  double best_score = -1.0;
  for (int i : fractional_vars(milp, x, tol)) {
    const double val = x.at(milp.variables[static_cast<std::size_t>(i)].name);
    const double frac = std::min(val - std::floor(val), std::ceil(val) - val);
    const double score = 0.5 - std::abs(frac - 0.5);
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best;
}

int pick_pseudo_cost(const OptimizationModel& milp,
                     const std::unordered_map<std::string, double>& x, double tol,
                     const std::unordered_map<std::string, PseudoCostStats>& stats) {
  int best = -1;
  double best_score = -1.0;
  for (int i : fractional_vars(milp, x, tol)) {
    const Variable& v = milp.variables[static_cast<std::size_t>(i)];
    const double val = x.at(v.name);
    const double fdown = val - std::floor(val);
    const double fup = std::ceil(val) - val;
    auto it = stats.find(v.name);
    double down_est = fdown;
    double up_est = fup;
    if (it != stats.end()) {
      if (it->second.down_count > 0) {
        down_est = fdown * (it->second.down_sum / it->second.down_count);
      }
      if (it->second.up_count > 0) {
        up_est = fup * (it->second.up_sum / it->second.up_count);
      }
    }
    const double score = std::min(down_est, up_est) + 1e-4 * std::max(down_est, up_est);
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best >= 0 ? best : pick_most_fractional(milp, x, tol);
}

bool node_dual_simplex_enabled() {
  static const bool enabled = [] {
    const char* s = std::getenv("SOVEREIGN_NODE_DUAL_SIMPLEX");
    return !s || (std::string(s) != "0" && std::string(s) != "off");
  }();
  return enabled;
}

struct DebugSolution {
  using Point = std::unordered_map<std::string, double>;

  static constexpr double kTolerance = 1e-6;

  bool enabled = false;
  bool violation = false;
  std::int64_t checks = 0;
  std::string path;
  std::string first_violation;
  Point values;

  static bool constraint_satisfied(const Constraint& c, const Point& point,
                                   double tolerance, std::string* detail) {
    double lhs = 0.0;
    for (const auto& kv : c.linear) {
      const auto it = point.find(kv.first);
      if (it == point.end() || !std::isfinite(it->second)) {
        if (detail != nullptr) *detail = "missing/non-finite variable " + kv.first;
        return false;
      }
      lhs += kv.second * it->second;
    }
    double residual = 0.0;
    if (c.sense == ConstraintSense::Le) {
      residual = std::max(0.0, lhs - c.rhs);
    } else if (c.sense == ConstraintSense::Ge) {
      residual = std::max(0.0, c.rhs - lhs);
    } else {
      residual = std::abs(lhs - c.rhs);
    }
    const double scale = std::max({1.0, std::abs(lhs), std::abs(c.rhs)});
    if (residual <= tolerance * scale) return true;
    if (detail != nullptr) {
      std::ostringstream oss;
      oss << "constraint=" << c.name << " lhs=" << std::setprecision(17) << lhs
          << " rhs=" << c.rhs << " residual=" << residual << " scale=" << scale;
      *detail = oss.str();
    }
    return false;
  }

  static bool feasible(const OptimizationModel& model, const Point& point,
                       double tolerance, std::string* detail) {
    for (const auto& v : model.variables) {
      const auto it = point.find(v.name);
      if (it == point.end() || !std::isfinite(it->second)) {
        if (detail != nullptr) *detail = "missing/non-finite variable " + v.name;
        return false;
      }
      const double scale_lower = std::max(1.0, std::abs(v.lower_bound));
      const double scale_upper = std::max(1.0, std::abs(v.upper_bound));
      if (std::isfinite(v.lower_bound) &&
          v.lower_bound - it->second > tolerance * scale_lower) {
        if (detail != nullptr) {
          std::ostringstream oss;
          oss << "variable=" << v.name << " value=" << std::setprecision(17) << it->second
              << " lower=" << v.lower_bound;
          *detail = oss.str();
        }
        return false;
      }
      if (std::isfinite(v.upper_bound) &&
          it->second - v.upper_bound > tolerance * scale_upper) {
        if (detail != nullptr) {
          std::ostringstream oss;
          oss << "variable=" << v.name << " value=" << std::setprecision(17) << it->second
              << " upper=" << v.upper_bound;
          *detail = oss.str();
        }
        return false;
      }
    }
    for (const auto& c : model.constraints) {
      if (!constraint_satisfied(c, point, tolerance, detail)) return false;
    }
    return true;
  }

  void fail(std::int64_t node, int depth, const std::string& component,
            const std::string& data) {
    if (violation) return;
    violation = true;
    std::ostringstream oss;
    oss << "node=" << node << " depth=" << depth << " component=" << component
        << " data=" << data;
    first_violation = oss.str();
    std::cerr << "[debug-solution] violation " << first_violation << '\n';
  }

  bool load(const OptimizationModel& model) {
    const char* env = std::getenv("SOVEREIGN_DEBUG_SOLUTION");
    if (env == nullptr || std::string(env).empty()) return false;
    enabled = true;
    path = env;
    try {
      std::ifstream in(path);
      if (!in) {
        fail(-1, -1, "load", "could not open " + path);
        return true;
      }
      nlohmann::json document;
      in >> document;
      const nlohmann::json* source = &document;
      if (document.is_object() && document.contains("primal")) source = &document["primal"];
      if (!source->is_object()) {
        fail(-1, -1, "load", "solution must be an object or contain an object named primal");
        return true;
      }
      for (auto it = source->begin(); it != source->end(); ++it) {
        if (!it.value().is_number()) {
          fail(-1, -1, "load", "non-numeric value for " + it.key());
          return true;
        }
        values[it.key()] = it.value().get<double>();
      }
      std::string detail;
      if (!feasible(model, values, kTolerance, &detail)) {
        fail(-1, -1, "reference", detail);
      }
    } catch (const std::exception& ex) {
      fail(-1, -1, "load", ex.what());
    }
    return true;
  }

  bool contains(const OptimizationModel& model) const {
    std::string ignored;
    return feasible(model, values, kTolerance, &ignored);
  }

  double objective(const OptimizationModel& model) const {
    double value = model.objective.constant;
    for (const auto& kv : model.objective.linear) {
      const auto it = values.find(kv.first);
      if (it != values.end()) value += kv.second * it->second;
    }
    return value;
  }

  void check_presolve(const PresolveResult& prep, std::int64_t node, int depth) {
    if (!enabled || violation) return;
    ++checks;
    Point projected = values;
    for (const auto& action : prep.actions) {
      if (action.type == PresolveActionType::FixVariable) {
        const auto it = values.find(action.name);
        if (it == values.end() ||
            std::abs(it->second - action.value) >
                kTolerance * std::max(1.0, std::abs(action.value))) {
          fail(node, depth, "presolve", "fixed variable " + action.name +
                                     " value was changed by presolve");
          return;
        }
        projected.erase(action.name);
      } else if (action.type == PresolveActionType::SubstituteVariable) {
        const auto name_it = values.find(action.name);
        const auto other_it = values.find(action.other);
        if (name_it == values.end() || other_it == values.end() ||
            std::abs(name_it->second -
                     (action.coeff * other_it->second + action.value)) >
                kTolerance *
                    std::max({1.0, std::abs(name_it->second),
                              std::abs(action.coeff * other_it->second + action.value)})) {
          fail(node, depth, "presolve", "substitution " + action.name + "=" +
                                     action.other + " changed the known solution");
          return;
        }
        projected.erase(action.name);
      }
    }
    std::string detail;
    if (!feasible(prep.reduced, projected, kTolerance, &detail)) {
      fail(node, depth, "presolve", detail);
    }
  }

  void check_lp_status(const SolverResult& result, bool contains_reference,
                       std::int64_t node, int depth, const std::string& component) {
    if (!enabled || violation || !contains_reference) return;
    ++checks;
    if (result.status == SolverStatus::Infeasible ||
        result.status == SolverStatus::Unbounded ||
        result.status == SolverStatus::NumericalError ||
        result.status == SolverStatus::IterationLimit ||
        result.status == SolverStatus::Error) {
      fail(node, depth, component, to_string(result.status) + ": " + result.message);
    }
  }

  void check_cut(const OptimizationModel& model, const Constraint& cut,
                 std::int64_t node, int depth, const std::string& family) {
    if (!enabled || violation || !contains(model)) return;
    ++checks;
    std::string detail;
    if (!constraint_satisfied(cut, values, kTolerance, &detail)) {
      fail(node, depth, "cut/" + family, detail);
    }
  }

  void check_bound(Sense sense, double bound, double reference_objective,
                   bool contains_reference, std::int64_t node, int depth,
                   const std::string& component) {
    if (!enabled || violation || !contains_reference || !std::isfinite(bound)) return;
    ++checks;
    const double scale = std::max({1.0, std::abs(bound), std::abs(reference_objective)});
    const bool invalid =
        sense == Sense::Minimize
            ? bound > reference_objective + kTolerance * scale
            : bound < reference_objective - kTolerance * scale;
    if (invalid) {
      std::ostringstream oss;
      oss << "bound=" << std::setprecision(17) << bound
          << " reference_objective=" << reference_objective;
      fail(node, depth, component, oss.str());
    }
  }

  void check_cutoff(bool contains_reference, std::int64_t node, int depth,
                    Sense sense, double node_bound, double incumbent) {
    if (!enabled || violation || !contains_reference) return;
    ++checks;
    std::ostringstream oss;
    oss << "node_bound=" << std::setprecision(17) << node_bound
        << " incumbent=" << incumbent << " reference_objective=" << objective_cache;
    fail(node, depth, "objective_cutoff", oss.str());
    (void)sense;
  }

  bool protect_reference_from_cutoff(bool contains_reference) {
    if (!enabled || !contains_reference) return false;
    ++checks;
    return true;
  }

  double objective_cache = 0.0;
};

SolverResult solve_node_lp(const OptimizationModel& node_model,
                           const RevisedSimplexOptions& /*lp_opt*/,
                           const LpBasis* warm = nullptr, LpBasis* basis_out = nullptr,
                           DebugSolution* debug = nullptr, std::int64_t debug_node = -1,
                           int debug_depth = -1, const char* debug_component = "node LP") {
  OptimizationModel relax = make_lp_relaxation(node_model);
  const bool reference_in_node = debug != nullptr && debug->enabled && debug->contains(relax);
  auto observe = [&](const SolverResult& result) {
    if (debug != nullptr && debug->enabled) {
      debug->check_lp_status(result, reference_in_node, debug_node, debug_depth,
                             debug_component);
    }
  };
  if (basis_out != nullptr) *basis_out = LpBasis{};
  if (node_dual_simplex_enabled()) {
    SolverResult dual = solve_lp_dual_simplex(relax, warm, basis_out);
    if (dual.status == SolverStatus::Optimal || dual.status == SolverStatus::Infeasible) {
      observe(dual);
      return dual;
    }
    observe(dual);
    if (basis_out != nullptr) *basis_out = LpBasis{};
  }
  // Branch bounds change at every node. The engine's one-time root presolve
  // cannot detect contradictions introduced by a later branch, and sending
  // those infeasible nodes to Phase I can make simplex cycle for 100k pivots.
  Presolver presolver;
  const PresolveResult prep = presolver.run(relax);
  if (debug != nullptr && debug->enabled && reference_in_node) {
    if (prep.infeasible || prep.unbounded) {
      debug->fail(debug_node, debug_depth, "presolve",
                  prep.message.empty() ? "presolve returned a terminal status" : prep.message);
    } else {
      debug->check_presolve(prep, debug_node, debug_depth);
    }
  }
  if (prep.infeasible || prep.unbounded) {
    SolverResult r;
    r.status = prep.infeasible ? SolverStatus::Infeasible : SolverStatus::Unbounded;
    r.message = prep.message;
    attach_presolve_stats(r, prep.stats);
    observe(r);
    return r;
  }
  if (prep.reduced.variables.empty()) {
    SolverResult r;
    r.status = SolverStatus::Optimal;
    r.has_objective_value = true;
    r = presolver.recover(r, prep, relax.sense);
    r.objective_value = relax.objective.constant;
    for (const auto& kv : relax.objective.linear) {
      r.objective_value += kv.second * r.primal.at(kv.first);
    }
    r.message = "Optimal (node presolve fixed all variables).";
    attach_presolve_stats(r, prep.stats);
    observe(r);
    return r;
  }
  // Route through LpSolver so SOVEREIGN_LP_ALGORITHM=auto|ipm|simplex applies
  // to MILP node relaxations (not just standalone LPs).
  SolverResult result = presolver.recover(LpSolver().solve(prep.reduced), prep, relax.sense);
  attach_presolve_stats(result, prep.stats);

  // Retry with pure simplex if auto mode returns NUMERICAL_ERROR
  // Sometimes IPM warmstart corrupts the simplex solve; pure simplex can succeed
  if (result.status == SolverStatus::NumericalError) {
    SolverResult simplex_result = presolver.recover(
        LpSolver().solve(prep.reduced, "simplex"), prep, relax.sense);
    if (simplex_result.status == SolverStatus::Optimal &&
        simplex_result.has_objective_value) {
      attach_presolve_stats(simplex_result, prep.stats);
      observe(simplex_result);
      return simplex_result;
    }
    observe(simplex_result);
  }

  observe(result);
  return result;
}

// An integer-feasible node LP point satisfies the node's rows (cuts included)
// only to the LP tolerance, and rounding its integers moves the original rows
// further. Re-solving the continuous part with the integers fixed makes the
// incumbent satisfy the original rows. If that LP fails, the point is kept.
void polish_incumbent(const OptimizationModel& milp, std::unordered_map<std::string, double>& x,
                      double& objective, const RevisedSimplexOptions& lp_opt,
                      DebugSolution* debug = nullptr) {
  OptimizationModel fixed = milp;
  bool any_continuous = false;
  for (auto& v : fixed.variables) {
    if (!is_integer_type(v.type)) {
      any_continuous = true;
      continue;
    }
    auto it = x.find(v.name);
    if (it == x.end()) return;
    v.lower_bound = v.upper_bound = std::round(it->second);
  }
  if (!any_continuous) return;
  const SolverResult r = solve_node_lp(fixed, lp_opt, nullptr, nullptr, debug, -1, -1,
                                       "heuristic polish");
  if (r.status != SolverStatus::Optimal || !r.has_objective_value) return;
  std::unordered_map<std::string, double> polished = x;
  for (const auto& v : fixed.variables) {
    if (is_integer_type(v.type)) {
      polished[v.name] = v.lower_bound;
      continue;
    }
    auto it = r.primal.find(v.name);
    if (it == r.primal.end()) return;
    polished[v.name] = it->second;
  }
  x = std::move(polished);
  objective = r.objective_value;
}

#if defined(_WIN32)
struct ParallelLpJob {
  const OptimizationModel* model = nullptr;
  const RevisedSimplexOptions* opt = nullptr;
  const LpBasis* warm = nullptr;
  DebugSolution* debug = nullptr;
  std::int64_t debug_node = -1;
  int debug_depth = -1;
  SolverResult result;
};

DWORD WINAPI parallel_lp_thread(LPVOID param) {
  auto* job = reinterpret_cast<ParallelLpJob*>(param);
  job->result = solve_node_lp(*job->model, *job->opt, job->warm, nullptr, job->debug,
                              job->debug_node, job->debug_depth, "strong branching");
  return 0;
}

void solve_two_lps_parallel(const OptimizationModel& down, const OptimizationModel& up,
                            const RevisedSimplexOptions& lp_opt, const LpBasis* warm,
                            SolverResult& down_r, SolverResult& up_r, bool enable_parallel,
                            DebugSolution* debug = nullptr, std::int64_t debug_node = -1,
                            int debug_depth = -1) {
  if (!enable_parallel) {
    down_r = solve_node_lp(down, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                           "strong branching");
    up_r = solve_node_lp(up, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                         "strong branching");
    return;
  }
  ParallelLpJob jobs[2];
  jobs[0].model = &down;
  jobs[0].opt = &lp_opt;
  jobs[0].warm = warm;
  jobs[0].debug = debug;
  jobs[0].debug_node = debug_node;
  jobs[0].debug_depth = debug_depth;
  jobs[1].model = &up;
  jobs[1].opt = &lp_opt;
  jobs[1].warm = warm;
  jobs[1].debug = debug;
  jobs[1].debug_node = debug_node;
  jobs[1].debug_depth = debug_depth;
  HANDLE h0 = CreateThread(nullptr, 0, parallel_lp_thread, &jobs[0], 0, nullptr);
  HANDLE h1 = CreateThread(nullptr, 0, parallel_lp_thread, &jobs[1], 0, nullptr);
  if (h0 && h1) {
    HANDLE hs[2] = {h0, h1};
    WaitForMultipleObjects(2, hs, TRUE, INFINITE);
    CloseHandle(h0);
    CloseHandle(h1);
    down_r = jobs[0].result;
    up_r = jobs[1].result;
  } else {
    if (h0) {
      WaitForSingleObject(h0, INFINITE);
      CloseHandle(h0);
      down_r = jobs[0].result;
      up_r = solve_node_lp(up, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                           "strong branching");
    } else if (h1) {
      WaitForSingleObject(h1, INFINITE);
      CloseHandle(h1);
      down_r = solve_node_lp(down, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                             "strong branching");
      up_r = jobs[1].result;
    } else {
      down_r = solve_node_lp(down, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                             "strong branching");
      up_r = solve_node_lp(up, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                           "strong branching");
    }
  }
}
#else
void solve_two_lps_parallel(const OptimizationModel& down, const OptimizationModel& up,
                            const RevisedSimplexOptions& lp_opt, const LpBasis* warm,
                            SolverResult& down_r, SolverResult& up_r, bool,
                            DebugSolution* debug = nullptr, std::int64_t debug_node = -1,
                            int debug_depth = -1) {
  down_r = solve_node_lp(down, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                         "strong branching");
  up_r = solve_node_lp(up, lp_opt, warm, nullptr, debug, debug_node, debug_depth,
                       "strong branching");
}
#endif

// Add cuts to a node, subject to an independent validity gate.
//
// Every candidate is checked by check_cut_validity() before it is allowed into
// the node model. Cuts are inherited by all descendants, so a single invalid cut
// does not merely weaken one node -- it can remove the true optimum from an
// entire subtree and the search will still report OPTIMAL for whatever survives.
// The gate is deliberately written independently of the generators so that a
// generator bug degrades into "cut rejected", never into a wrong answer.
int apply_cuts(OptimizationModel& model, const std::unordered_map<std::string, double>& x,
               double integer_tol, int max_cuts,
               const std::vector<std::unordered_map<std::string, double>>& reference_points,
               std::vector<std::string>* rejections, std::size_t base_rows, bool with_cmir,
               std::unordered_map<std::string, std::int64_t>* cut_counts,
               DebugSolution* debug = nullptr, std::int64_t debug_node = -1,
               int debug_depth = -1) {
  auto covers = generate_cover_cuts(model, x, integer_tol);
  std::vector<Cut> cmir;
  if (with_cmir) cmir = generate_cmir_cuts(model, x, integer_tol, 8, base_rows);
  auto gomory = generate_mir_cuts(model, x, integer_tol);

  std::vector<Cut> candidates;
  candidates.reserve(covers.size() + cmir.size() + gomory.size());
  for (const auto& c : covers) candidates.push_back(c);
  for (const auto& c : cmir) candidates.push_back(c);
  for (const auto& c : gomory) candidates.push_back(c);

  int added = 0;
  int rejected = 0;
  for (const auto& cut : candidates) {
    if (added >= max_cuts) break;
    if (debug != nullptr && debug->enabled) {
      debug->check_cut(model, cut.constraint, debug_node, debug_depth, cut.source);
    }
    const std::string why = check_cut_validity(model, cut.constraint, reference_points,
                                               std::max(integer_tol, 1e-6));
    if (!why.empty()) {
      ++rejected;
      if (rejections != nullptr) rejections->push_back(why);
      continue;
    }
    model.constraints.push_back(cut.constraint);
    if (cut_counts != nullptr) ++(*cut_counts)[cut.source];
    ++added;
  }

  if (rejected > 0 && rejections != nullptr) {
    // One summary line, not one per cut: a pathological generator must not be
    // able to blow up the warnings vector.
    std::ostringstream oss;
    oss << "Cut validity gate rejected " << rejected << " candidate cut"
        << (rejected == 1 ? "" : "s") << " (see cut_rejections).";
    rejections->push_back(oss.str());
  }
  return added;
}

double relative_gap(Sense sense, double bound, double incumbent) {
  const double scale = std::max(1.0, std::abs(incumbent));
  if (sense == Sense::Minimize) return (incumbent - bound) / scale;
  return (bound - incumbent) / scale;
}

bool better_incumbent(Sense sense, double cand, double incumbent, bool has) {
  if (!has) return true;
  return sense == Sense::Minimize ? cand < incumbent - 1e-12 : cand > incumbent + 1e-12;
}

bool can_prune_by_bound(Sense sense, double node_bound, double incumbent, bool has,
                        double mip_gap) {
  if (!has) return false;
  return relative_gap(sense, node_bound, incumbent) <= mip_gap;
}

struct BestBoundCompareMin {
  bool operator()(const SearchNode& a, const SearchNode& b) const {
    if (a.bound != b.bound) return a.bound > b.bound;
    return a.id > b.id;
  }
};
struct BestBoundCompareMax {
  bool operator()(const SearchNode& a, const SearchNode& b) const {
    if (a.bound != b.bound) return a.bound < b.bound;
    return a.id > b.id;
  }
};

int pick_strong_branch(const OptimizationModel& milp,
                       const std::unordered_map<std::string, double>& x,
                       double parent_obj, double tol, int max_candidates,
                       const RevisedSimplexOptions& lp_opt, const LpBasis* warm,
                       bool parallel_lps,
                       std::unordered_map<std::string, PseudoCostStats>* stats,
                       DebugSolution* debug = nullptr, std::int64_t debug_node = -1,
                       int debug_depth = -1) {
  auto cands = fractional_vars(milp, x, tol);
  if (cands.empty()) return -1;
  std::sort(cands.begin(), cands.end(), [&](int a, int b) {
    const double va = x.at(milp.variables[static_cast<std::size_t>(a)].name);
    const double vb = x.at(milp.variables[static_cast<std::size_t>(b)].name);
    const double fa = std::min(va - std::floor(va), std::ceil(va) - va);
    const double fb = std::min(vb - std::floor(vb), std::ceil(vb) - vb);
    return std::abs(fa - 0.5) < std::abs(fb - 0.5);
  });
  if (static_cast<int>(cands.size()) > max_candidates) {
    cands.resize(static_cast<std::size_t>(max_candidates));
  }

  int best = -1;
  double best_score = -1.0;
  for (int i : cands) {
    const Variable& bv = milp.variables[static_cast<std::size_t>(i)];
    const double val = x.at(bv.name);
    const double floor_v = std::floor(val);
    const double ceil_v = std::ceil(val);

    OptimizationModel down = milp;
    down.variables[static_cast<std::size_t>(i)].upper_bound =
        std::min(down.variables[static_cast<std::size_t>(i)].upper_bound, floor_v);
    OptimizationModel up = milp;
    up.variables[static_cast<std::size_t>(i)].lower_bound =
        std::max(up.variables[static_cast<std::size_t>(i)].lower_bound, ceil_v);

    SolverResult rd, ru;
    solve_two_lps_parallel(down, up, lp_opt, warm, rd, ru, parallel_lps, debug,
                           debug_node, debug_depth);

    const bool down_inf = rd.status == SolverStatus::Infeasible;
    const bool up_inf = ru.status == SolverStatus::Infeasible;
    const double down_obj = rd.has_objective_value ? rd.objective_value : parent_obj;
    const double up_obj = ru.has_objective_value ? ru.objective_value : parent_obj;

    const double fdown = val - floor_v;
    const double fup = ceil_v - val;
    if (stats && fdown > tol) {
      auto& st = (*stats)[bv.name];
      st.down_sum += std::abs(down_obj - parent_obj) / fdown;
      ++st.down_count;
    }
    if (stats && fup > tol) {
      auto& st = (*stats)[bv.name];
      st.up_sum += std::abs(up_obj - parent_obj) / fup;
      ++st.up_count;
    }

    const double down_deg = down_inf ? 1e6 : std::max(0.0, std::abs(down_obj - parent_obj));
    const double up_deg = up_inf ? 1e6 : std::max(0.0, std::abs(up_obj - parent_obj));
    const double score = std::min(down_deg, up_deg) + 1e-4 * std::max(down_deg, up_deg);
    if (score > best_score) {
      best_score = score;
      best = i;
    }
  }
  return best >= 0 ? best : pick_most_fractional(milp, x, tol);
}

double product_score(double down_gain, double up_gain) {
  return std::max(down_gain, 1e-6) * std::max(up_gain, 1e-6);
}

// Reliability branching. Candidates are ranked by pseudocost; those with too
// few observations are strong-branched (warm-started) until a few in a row fail
// to improve the best score, and every strong branch feeds the pseudocosts, so
// the expensive part fades out as the search learns.
int pick_reliability_branch(const OptimizationModel& milp,
                            const std::unordered_map<std::string, double>& x, double parent_obj,
                            double tol, int reliability, int max_strong,
                            const RevisedSimplexOptions& lp_opt, const LpBasis* warm,
                            bool parallel_lps,
                            std::unordered_map<std::string, PseudoCostStats>& stats,
                            DebugSolution* debug = nullptr, std::int64_t debug_node = -1,
                            int debug_depth = -1) {
  const auto cands = fractional_vars(milp, x, tol);
  if (cands.empty()) return -1;

  double down_sum = 0.0, up_sum = 0.0;
  int down_n = 0, up_n = 0;
  for (const auto& kv : stats) {
    if (kv.second.down_count > 0) {
      down_sum += kv.second.down_sum / kv.second.down_count;
      ++down_n;
    }
    if (kv.second.up_count > 0) {
      up_sum += kv.second.up_sum / kv.second.up_count;
      ++up_n;
    }
  }
  const double avg_down = down_n > 0 ? down_sum / down_n : 1.0;
  const double avg_up = up_n > 0 ? up_sum / up_n : 1.0;

  struct Scored {
    int var;
    double score;
    bool reliable;
  };
  std::vector<Scored> scored;
  scored.reserve(cands.size());
  for (int i : cands) {
    const std::string& name = milp.variables[static_cast<std::size_t>(i)].name;
    const double val = x.at(name);
    const double fdown = val - std::floor(val), fup = std::ceil(val) - val;
    auto it = stats.find(name);
    const bool known = it != stats.end();
    const double unit_down = known && it->second.down_count > 0
                                 ? it->second.down_sum / it->second.down_count : avg_down;
    const double unit_up = known && it->second.up_count > 0
                               ? it->second.up_sum / it->second.up_count : avg_up;
    const bool reliable =
        known && std::min(it->second.down_count, it->second.up_count) >= reliability;
    scored.push_back({i, product_score(fdown * unit_down, fup * unit_up), reliable});
  }
  std::sort(scored.begin(), scored.end(),
            [](const Scored& a, const Scored& b) { return a.score > b.score; });

  int best = scored.front().var;
  double best_score = -1.0;
  int strong_done = 0, no_improve = 0;
  constexpr int kLookahead = 4;
  for (const Scored& s : scored) {
    double score = s.score;
    if (!s.reliable && strong_done < max_strong && no_improve < kLookahead) {
      const Variable& bv = milp.variables[static_cast<std::size_t>(s.var)];
      const double val = x.at(bv.name);
      OptimizationModel down = milp;
      down.variables[static_cast<std::size_t>(s.var)].upper_bound =
          std::min(bv.upper_bound, std::floor(val));
      OptimizationModel up = milp;
      up.variables[static_cast<std::size_t>(s.var)].lower_bound =
          std::max(bv.lower_bound, std::ceil(val));
      SolverResult rd, ru;
      solve_two_lps_parallel(down, up, lp_opt, warm, rd, ru, parallel_lps, debug,
                             debug_node, debug_depth);
      ++strong_done;

      const bool down_inf = rd.status == SolverStatus::Infeasible;
      const bool up_inf = ru.status == SolverStatus::Infeasible;
      if (down_inf && up_inf) return s.var;  // both children die; any choice prunes the node
      const double fdown = val - std::floor(val), fup = std::ceil(val) - val;
      double down_deg = down_inf ? 1e6 : 0.0, up_deg = up_inf ? 1e6 : 0.0;
      if (!down_inf && rd.has_objective_value) {
        down_deg = std::abs(rd.objective_value - parent_obj);
        auto& st = stats[bv.name];
        st.down_sum += down_deg / std::max(fdown, tol);
        ++st.down_count;
      }
      if (!up_inf && ru.has_objective_value) {
        up_deg = std::abs(ru.objective_value - parent_obj);
        auto& st = stats[bv.name];
        st.up_sum += up_deg / std::max(fup, tol);
        ++st.up_count;
      }
      score = product_score(down_deg, up_deg);
      no_improve = score > best_score ? 0 : no_improve + 1;
    }
    if (score > best_score) {
      best_score = score;
      best = s.var;
    }
  }
  return best;
}

}  // namespace

BranchAndBoundSolver::BranchAndBoundSolver(BranchAndBoundOptions options)
    : options_(std::move(options)) {}

SolverResult BranchAndBoundSolver::solve(const OptimizationModel& model) const {
  SolverResult result;
  result.status = SolverStatus::Error;
  DebugSolution debug;
  const bool debug_enabled = debug.load(model);
  if (debug_enabled) {
    debug.objective_cache = debug.objective(model);
    result.mip_diagnostics.debug_solution_enabled = true;
    result.mip_diagnostics.debug_solution_path = debug.path;
  }
  auto attach_debug = [&]() {
    if (!debug_enabled) return;
    result.mip_diagnostics.debug_solution_violation = debug.violation;
    result.mip_diagnostics.debug_solution_checks = debug.checks;
    result.mip_diagnostics.debug_solution_first_violation = debug.first_violation;
    if (debug.violation && !debug.first_violation.empty()) {
      result.warnings.push_back("debug_solution_violation: " + debug.first_violation);
    }
  };

  if (model.problem_type != ProblemType::MILP &&
      model.problem_type != ProblemType::LP) {
    result.message = "BranchAndBoundSolver expects MILP (or LP).";
    return result;
  }

  bool any_integer = false;
  for (const auto& v : model.variables) {
    if (is_integer_type(v.type)) {
      any_integer = true;
      break;
    }
  }
  if (!any_integer) {
    OptimizationModel lp = model;
    lp.problem_type = ProblemType::LP;
    return RevisedSimplexSolver().solve(lp);
  }

  for (const auto& v : model.variables) {
    if (v.type == VariableType::Binary) {
      if (v.lower_bound < -1e-9 || v.upper_bound > 1.0 + 1e-9) {
        result.message = "Binary variable " + v.name + " has invalid bounds.";
        return result;
      }
    }
  }

  RevisedSimplexOptions lp_opt;
  lp_opt.feasibility_tol = options_.feasibility_tol;
  lp_opt.enable_scaling = true;
  // PART 2: Reduce refactor frequency for B&B node LPs
  // Node LPs are small and warm-started. Shorter eta chains (refactor every 20
  // pivots instead of 64) reduce numerical drift and prevent cycling from
  // accumulated rounding errors in product-form updates.
  lp_opt.refactor_every = 20;

  const Sense sense = model.sense;
  const bool parallel_strong = options_.parallel_workers != 1;

  const double t_start = now_seconds();
  bool has_incumbent = false;
  double incumbent = 0.0;
  std::unordered_map<std::string, double> incumbent_x;
  DebugSolution* debug_ptr = debug_enabled ? &debug : nullptr;
  auto offer_incumbent = [&](std::unordered_map<std::string, double> x, double objective) {
    if (!better_incumbent(sense, objective, incumbent, has_incumbent)) return false;
    polish_incumbent(model, x, objective, lp_opt, debug_ptr);
    if (!better_incumbent(sense, objective, incumbent, has_incumbent)) return false;
    has_incumbent = true;
    incumbent = objective;
    incumbent_x = std::move(x);
    if (result.mip_diagnostics.time_to_first_incumbent < 0.0) {
      result.mip_diagnostics.time_to_first_incumbent = now_seconds() - t_start;
    }
    return true;
  };
  std::int64_t nodes = 0;
  std::int64_t lp_iterations = 0;
  std::int64_t next_id = 1;
  // The global bound is the weakest bound of every node not yet resolved: the
  // open ones, plus those closed without being solved (pruned against the
  // incumbent within mip_gap, or dropped because their LP failed).
  const double no_bound = (sense == Sense::Minimize) ? std::numeric_limits<double>::infinity()
                                                     : -std::numeric_limits<double>::infinity();
  auto weaker = [&](double a, double b) {
    return sense == Sense::Minimize ? std::min(a, b) : std::max(a, b);
  };
  double closed_bound = no_bound;
  auto close_node = [&](double bound, const SearchNode& node,
                        const OptimizationModel& node_model,
                        const std::string& component) {
    closed_bound = weaker(closed_bound, bound);
    if (debug_ptr != nullptr) {
      debug_ptr->check_bound(sense, bound, debug_ptr->objective_cache,
                             debug_ptr->contains(node_model), node.id, node.depth, component);
    }
  };
  std::vector<std::string> warnings;
  std::vector<std::string> cut_rejections;
  std::size_t cut_rejection_count = 0;
  std::int64_t dive_lps = 0;
  std::int64_t tree_cut_nodes = 0;
  std::int64_t tree_cuts = 0;
  auto note_cut_rejection = [&](std::string reason) {
    ++cut_rejection_count;
    if (cut_rejections.size() < 20) cut_rejections.push_back(std::move(reason));
  };
  std::unordered_map<std::string, PseudoCostStats> pseudo;
  bool any_node_lp_error = false;
  bool hit_node_limit = false;
  double t_lp = 0.0, t_cuts = 0.0, t_heur = 0.0, t_branch = 0.0;
  const bool log_progress = std::getenv("SOVEREIGN_BB_LOG") != nullptr;

  std::priority_queue<SearchNode, std::vector<SearchNode>, BestBoundCompareMin> pq_min;
  std::priority_queue<SearchNode, std::vector<SearchNode>, BestBoundCompareMax> pq_max;

  // The child chosen for plunging waits here instead of in the queue. It is
  // still an open node: it counts for emptiness and is always popped next.
  std::vector<SearchNode> plunge;

  auto push_node = [&](SearchNode node) {
    if (sense == Sense::Minimize) pq_min.push(std::move(node));
    else pq_max.push(std::move(node));
  };
  auto empty_queue = [&]() {
    return plunge.empty() && (sense == Sense::Minimize ? pq_min.empty() : pq_max.empty());
  };
  auto global_bound = [&]() {
    double b = closed_bound;
    for (const auto& n : plunge) b = weaker(b, n.bound);
    if (sense == Sense::Minimize && !pq_min.empty()) b = weaker(b, pq_min.top().bound);
    if (sense == Sense::Maximize && !pq_max.empty()) b = weaker(b, pq_max.top().bound);
    return b;
  };
  auto record_presolve = [&](const SolverResult& lp_result) {
    result.mip_diagnostics.presolve_fixed_variables += lp_result.presolve_fixed_variables;
    result.mip_diagnostics.presolve_substituted_variables += lp_result.presolve_substituted_variables;
    result.mip_diagnostics.presolve_removed_constraints += lp_result.presolve_removed_constraints;
    result.mip_diagnostics.presolve_tightened_bounds += lp_result.presolve_tightened_bounds;
    result.mip_diagnostics.presolve_passes += lp_result.presolve_passes;
  };
  auto pop_node = [&]() {
    SearchNode n;
    if (!plunge.empty()) {
      n = std::move(plunge.back());
      plunge.pop_back();
      return n;
    }
    if (sense == Sense::Minimize) {
      n = pq_min.top();
      pq_min.pop();
    } else {
      n = pq_max.top();
      pq_max.pop();
    }
    return n;
  };

  SearchNode root;
  {
    auto root_model = std::make_shared<OptimizationModel>(model);
    root_model->problem_type = ProblemType::MILP;
    root.model = std::move(root_model);
  }
  root.depth = 0;
  root.id = next_id++;
  root.bound = (sense == Sense::Minimize) ? -std::numeric_limits<double>::infinity()
                                          : std::numeric_limits<double>::infinity();
  push_node(std::move(root));

  while (!empty_queue()) {
    if (nodes >= options_.max_nodes) {
      std::ostringstream oss;
      oss << "Branch-and-bound stopped at the node limit (" << options_.max_nodes
          << " nodes explored). ";
      if (has_incumbent) {
        oss << "Reporting the incumbent as FEASIBLE; optimality is NOT proven. "
            << "Open nodes remain in the queue.";
      } else {
        oss << "No integer-feasible point was found, so infeasibility cannot be "
            << "certified either.";
      }
      result.status = has_incumbent ? SolverStatus::Feasible : SolverStatus::IterationLimit;
      result.message = oss.str();
      hit_node_limit = true;
      break;
    }
    if (options_.time_limit_seconds > 0.0 &&
        now_seconds() - t_start >= options_.time_limit_seconds) {
      std::ostringstream oss;
      oss << "Branch-and-bound stopped at the time limit (" << options_.time_limit_seconds
          << " s, " << nodes << " nodes explored). ";
      if (has_incumbent) {
        oss << "Reporting the incumbent as FEASIBLE; optimality is NOT proven. "
            << "Open nodes remain in the queue.";
      } else {
        oss << "No integer-feasible point was found, so infeasibility cannot be "
            << "certified either.";
      }
      result.status = has_incumbent ? SolverStatus::Feasible : SolverStatus::TimeLimit;
      result.message = oss.str();
      hit_node_limit = true;
      break;
    }

    SearchNode node = pop_node();
    ++nodes;
    if (log_progress && (nodes <= 20 || nodes % 500 == 0)) {
      const double progress_gap =
          has_incumbent && std::isfinite(node.bound)
              ? std::max(0.0, relative_gap(sense, node.bound, incumbent))
              : -1.0;
      std::cerr << "[bb] nodes=" << nodes << " open=" << (pq_min.size() + pq_max.size() + plunge.size())
                << " incumbent=" << (has_incumbent ? std::to_string(incumbent) : "-")
                << " node_bound=" << node.bound << " depth=" << node.depth
                << " gap=" << progress_gap
                << " t=" << (now_seconds() - t_start) << "s (lp " << t_lp << " cuts " << t_cuts
                << " heur " << t_heur << " branch " << t_branch << ")\n";
    }
    OptimizationModel node_model = *node.model;
    for (const BoundChange& bc : node.bounds) {
      Variable& v = node_model.variables[static_cast<std::size_t>(bc.var)];
      v.lower_bound = bc.lower;
      v.upper_bound = bc.upper;
    }
    const bool reference_in_node =
        debug_ptr != nullptr && debug_ptr->contains(node_model);
    if (debug_ptr != nullptr) {
      debug_ptr->check_bound(sense, node.bound, debug_ptr->objective_cache,
                             reference_in_node, node.id, node.depth, "node bound");
    }
    const bool protect_node_reference =
        debug_ptr != nullptr && debug_ptr->protect_reference_from_cutoff(reference_in_node);
    if (!protect_node_reference &&
        can_prune_by_bound(sense, node.bound, incumbent, has_incumbent, options_.mip_gap)) {
      if (debug_ptr != nullptr) {
        debug_ptr->check_cutoff(reference_in_node, node.id, node.depth, sense,
                                node.bound, incumbent);
      }
      close_node(node.bound, node, node_model, "objective cutoff");
      continue;
    }

    LpBasis node_basis;
    double t0 = now_seconds();
    SolverResult lp = solve_node_lp(node_model, lp_opt, node.basis.get(), &node_basis,
                                    debug_ptr, node.id, node.depth, "node LP");
    t_lp += now_seconds() - t0;
    lp_iterations += lp.iterations;
    record_presolve(lp);

    if (lp.status == SolverStatus::Infeasible) continue;
    if (lp.status == SolverStatus::Optimal && lp.has_objective_value && node.branch_var >= 0 &&
        node.branch_distance > options_.integer_tol) {
      auto& st = pseudo[node_model.variables[static_cast<std::size_t>(node.branch_var)].name];
      const double unit = std::abs(lp.objective_value - node.parent_obj) / node.branch_distance;
      if (node.branch_up) {
        st.up_sum += unit;
        ++st.up_count;
      } else {
        st.down_sum += unit;
        ++st.down_count;
      }
    }
    if (lp.status == SolverStatus::Unbounded) {
      ++result.mip_diagnostics.unbounded_nodes;
      result.status = SolverStatus::Unbounded;
      result.message = "MILP relaxation unbounded.";
      result.nodes = nodes;
      result.iterations = lp_iterations;
      attach_debug();
      return result;
    }
    // A node LP without a certified objective has no valid bound either, so it
    // is a failure too — skipping it would drop the subtree without a trace.
    if ((lp.status != SolverStatus::Optimal && lp.status != SolverStatus::Feasible) ||
        !lp.has_objective_value) {
      // A node LP failed to solve. This subtree cannot be soundly pruned or
      // explored further — record the failure and downgrade the final status so
      // we never silently report OPTIMAL (or INFEASIBLE) with a dropped branch.
      // Distinguish the reason, because "ran out of iterations" and "the basis
      // went singular" call for completely different responses.
      any_node_lp_error = true;
      ++result.mip_diagnostics.node_lp_failures;
      ++result.mip_diagnostics.dropped_subtrees;
      if (lp.status == SolverStatus::NumericalError) {
        ++result.mip_diagnostics.numerical_error_nodes;
      } else if (lp.status == SolverStatus::IterationLimit) {
        ++result.mip_diagnostics.iteration_limit_nodes;
      }
      close_node(node.bound, node, node_model, "LP failure");
      std::ostringstream oss;
      oss << "Node LP returned " << to_string(lp.status)
          << " (subtree dropped, optimality not certified)";
      if (!lp.message.empty()) oss << ": " << lp.message;
      if (lp.duality_gap > 0.0) {
        oss << " [relative duality gap " << lp.duality_gap << "]";
      }
      warnings.push_back(oss.str());
      if (std::getenv("SOVEREIGN_DUMP_FAILING_NODE")) {
        std::ofstream out(std::getenv("SOVEREIGN_DUMP_FAILING_NODE"));
        if (out) out << model_to_json_string(make_lp_relaxation(node_model)) << '\n';
      }
      continue;
    }

    // Tree-wide branch-and-cut
    const bool cut_here = options_.enable_cuts &&
                          (node.depth == 0 || options_.cut_frequency <= 1 ||
                           (node.depth % options_.cut_frequency) == 0);
    t0 = now_seconds();
    int cuts_added_here = 0;
    if (cut_here) {
      // Any incumbent we already have is a proven-feasible integer point, so it
      // is exactly the evidence the validity gate needs.
      std::vector<std::unordered_map<std::string, double>> reference_points;
      if (has_incumbent) {
        reference_points.push_back(incumbent_x);
      }
      if (debug_ptr != nullptr && debug_ptr->contains(node_model)) {
        reference_points.push_back(debug_ptr->values);
      }
      const int rounds = (node.depth == 0) ? options_.max_cut_rounds : 1;
      for (int round = 0; round < rounds; ++round) {
        std::vector<std::string> rejections;
        const int added = apply_cuts(node_model, lp.primal, options_.integer_tol,
                                     options_.max_cuts_per_node, reference_points,
                                     &rejections, model.constraints.size(),
                                     options_.cmir_max_depth >= 0 &&
                                         node.depth <= options_.cmir_max_depth,
                                     &result.mip_diagnostics.cuts_by_family, debug_ptr,
                                     node.id, node.depth);
        for (auto& r : rejections) note_cut_rejection(std::move(r));
        if (added == 0) break;
        cuts_added_here += added;
        LpBasis cut_basis;
        SolverResult cut_lp = solve_node_lp(node_model, lp_opt,
                                            node_basis.empty() ? nullptr : &node_basis,
                                            &cut_basis, debug_ptr, node.id, node.depth,
                                            "cut-loop LP");
        lp_iterations += cut_lp.iterations;
        record_presolve(cut_lp);
        // Keep the pre-cut LP on failure: cuts are valid inequalities, so its
        // objective is still a sound (weaker) bound for the cut-augmented node.
        if ((cut_lp.status != SolverStatus::Optimal &&
             cut_lp.status != SolverStatus::Feasible) ||
            !cut_lp.has_objective_value) {
          std::ostringstream oss;
          oss << "Cut loop stopped at depth " << node.depth << " round " << round
              << ": node LP became " << to_string(cut_lp.status);
          if (!cut_lp.message.empty()) oss << " (" << cut_lp.message << ")";
          note_cut_rejection(oss.str());
          break;
        }
        lp = std::move(cut_lp);
        node_basis = std::move(cut_basis);
      }
      if (cuts_added_here > 0 && node.depth == 0) {
        std::ostringstream oss;
        oss << "Added " << cuts_added_here << " validated cut"
            << (cuts_added_here == 1 ? "" : "s") << " at the root";
        warnings.push_back(oss.str());
      } else if (cuts_added_here > 0) {
        ++tree_cut_nodes;
        tree_cuts += cuts_added_here;
      }
    }
    t_cuts += now_seconds() - t0;

    std::shared_ptr<const LpBasis> child_basis;
    if (!node_basis.empty()) child_basis = std::make_shared<const LpBasis>(std::move(node_basis));

    t0 = now_seconds();
    if (options_.enable_heuristics) {
      HeuristicResult h = rounding_heuristic(node_model, lp.primal, options_.integer_tol);
      const int freq = has_incumbent ? options_.dive_frequency
                                     : std::max(1, options_.dive_frequency / 5);
      // Diving LPs are capped at a share of the node LPs (more while there is
      // no incumbent) so the heuristic cannot crowd out the tree search.
      const double dive_share = has_incumbent ? 0.2 : 0.5;
      const bool within_budget =
          static_cast<double>(dive_lps) <= dive_share * static_cast<double>(nodes) + 1000.0;
      const bool dive_now =
          node.depth == 0 ||
          (options_.dive_frequency > 0 && nodes % freq == 0 && within_budget);
      if (!h.found && dive_now) {
        h = diving_heuristic(node_model, lp.primal, options_.dive_max_depth,
                             options_.integer_tol, child_basis.get(),
                             has_incumbent ? &incumbent : nullptr);
        dive_lps += h.lp_solves;
      }
      if (h.found && offer_incumbent(h.primal, h.objective)) {
        warnings.push_back("Heuristic incumbent via " + h.method);
      }
    }
    t_heur += now_seconds() - t0;

    node.bound = lp.objective_value;
    if (debug_ptr != nullptr) {
      debug_ptr->check_bound(sense, node.bound, debug_ptr->objective_cache,
                             debug_ptr->contains(node_model), node.id, node.depth,
                             "node LP bound");
    }

    // Accept integer-feasible nodes BEFORE bound pruning. Pruning on
    // relative_gap <= mip_gap when the LP objective is within mip_gap of the
    // incumbent would otherwise skip recording an integer solution whose
    // objective is equal (or only epsilon-better) than the incumbent — a
    // sibling of the "equality/tolerance" class of bugs. Integrality is the
    // authority for accepting; bound pruning only applies to fractional nodes.
    if (is_integer_feasible(node_model, lp.primal, options_.integer_tol)) {
      auto x = lp.primal;
      snap_integer_primal(node_model, x, options_.integer_tol);
      offer_incumbent(std::move(x), lp.objective_value);
      continue;
    }

    const bool protect_lp_reference =
        debug_ptr != nullptr &&
        debug_ptr->protect_reference_from_cutoff(debug_ptr->contains(node_model));
    if (!protect_lp_reference &&
        can_prune_by_bound(sense, lp.objective_value, incumbent, has_incumbent,
                           options_.mip_gap)) {
      if (debug_ptr != nullptr) {
        debug_ptr->check_cutoff(debug_ptr->contains(node_model), node.id, node.depth,
                                sense, lp.objective_value, incumbent);
      }
      close_node(lp.objective_value, node, node_model, "objective cutoff");
      continue;
    }

    int bvar = -1;
    t0 = now_seconds();
    if (options_.branch_rule == BranchRule::StrongBranching) {
      bvar = pick_reliability_branch(node_model, lp.primal, lp.objective_value,
                                     options_.integer_tol, options_.reliability_threshold,
                                     options_.max_strong_per_node, lp_opt, child_basis.get(),
                                     parallel_strong, pseudo, debug_ptr, node.id, node.depth);
    } else if (options_.branch_rule == BranchRule::FullStrong) {
      bvar = pick_strong_branch(node_model, lp.primal, lp.objective_value,
                                options_.integer_tol, options_.strong_branch_candidates,
                                lp_opt, child_basis.get(), parallel_strong, &pseudo,
                                debug_ptr, node.id, node.depth);
    } else if (options_.branch_rule == BranchRule::PseudoCost) {
      bvar = pick_pseudo_cost(node_model, lp.primal, options_.integer_tol, pseudo);
    } else {
      bvar = pick_most_fractional(node_model, lp.primal, options_.integer_tol);
    }
    t_branch += now_seconds() - t0;

    if (bvar < 0) {
      auto x = lp.primal;
      snap_integer_primal(node_model, x, options_.integer_tol);
      offer_incumbent(std::move(x), lp.objective_value);
      continue;
    }

    const Variable& bv = node_model.variables[static_cast<std::size_t>(bvar)];
    const double val = lp.primal.at(bv.name);
    const double floor_v = std::floor(val);
    const double ceil_v = std::ceil(val);
    // Plunge toward the nearer integer; the sibling waits in the queue.
    const bool prefer_up = val - floor_v >= 0.5;
    // Cuts added here are part of this subtree's model, so the children need a
    // new shared base; otherwise they keep the parent's base and bound list.
    std::shared_ptr<const OptimizationModel> child_model = node.model;
    std::vector<BoundChange> child_bounds = std::move(node.bounds);
    if (cuts_added_here > 0) {
      child_model = std::make_shared<const OptimizationModel>(node_model);
      child_bounds.clear();
    }
    auto make_child = [&](bool up, double lower, double upper) {
      SearchNode child;
      child.model = child_model;
      child.bounds.reserve(child_bounds.size() + 1);
      child.bounds = child_bounds;
      auto same_var = std::find_if(child.bounds.begin(), child.bounds.end(),
                                   [&](const BoundChange& bc) { return bc.var == bvar; });
      if (same_var != child.bounds.end()) *same_var = BoundChange{bvar, lower, upper};
      else child.bounds.push_back(BoundChange{bvar, lower, upper});
      child.id = next_id++;
      child.depth = node.depth + 1;
      child.bound = lp.objective_value;
      child.basis = child_basis;
      child.branch_var = bvar;
      child.branch_up = up;
      child.branch_distance = up ? ceil_v - val : val - floor_v;
      child.parent_obj = lp.objective_value;
      return child;
    };
    for (const bool up : {prefer_up, !prefer_up}) {
      const double lower = up ? std::max(bv.lower_bound, ceil_v) : bv.lower_bound;
      const double upper = up ? bv.upper_bound : std::min(bv.upper_bound, floor_v);
      if (lower > upper + 1e-12) continue;
      SearchNode child = make_child(up, lower, upper);
      if (options_.plunging && up == prefer_up) plunge.push_back(std::move(child));
      else push_node(std::move(child));
    }
  }

  result.nodes = nodes;
  result.iterations = lp_iterations;
  result.warnings = warnings;
  // Once the tree is exhausted, all remaining nodes may have been closed by
  // an incumbent cutoff. Their LP bounds can be strictly weaker in the
  // opposite direction than the incumbent (for example, all remaining
  // minimization-node bounds can be above the incumbent). The incumbent is a
  // valid attained objective and must therefore be included in the reported
  // dual bound; otherwise an optimal solve can publish a bound outside the
  // original model's feasible optimum.
  const double raw_global_bound = global_bound();
  const double final_bound =
      has_incumbent && std::isfinite(raw_global_bound)
          ? weaker(raw_global_bound, incumbent)
          : raw_global_bound;
  if (debug_ptr != nullptr) {
    debug_ptr->check_bound(sense, final_bound, debug_ptr->objective_cache, true, -1, -1,
                           "final best bound");
  }
  if (std::isfinite(final_bound)) {
    result.mip_diagnostics.has_best_bound = true;
    result.mip_diagnostics.best_bound = final_bound;
  }
  result.mip_diagnostics.cut_validity_rejections =
      static_cast<std::int64_t>(cut_rejection_count);

  if (has_incumbent) {
    const double bound = weaker(final_bound, incumbent);
    if (std::isfinite(bound)) {
      result.optimality_gap = std::max(0.0, relative_gap(sense, bound, incumbent));
    }
    if (hit_node_limit) {
      // message was already set at the point we broke out of the loop.
      result.status = SolverStatus::Feasible;
    } else if (any_node_lp_error) {
      // A subtree LP failed; we cannot certify optimality even though the
      // remaining tree was exhausted. Say which subtrees and why.
      result.status = SolverStatus::Feasible;
      std::ostringstream oss;
      oss << "Integer feasible solution found, but the search is INCOMPLETE: one or more "
          << "node LPs failed, so their subtrees were dropped without being explored or "
          << "proved infeasible. Optimality is NOT certified. See warnings for the "
          << "per-node reason.";
      result.message = oss.str();
    } else if (result.optimality_gap <= options_.mip_gap || empty_queue()) {
      result.status = SolverStatus::Optimal;
      std::ostringstream oss;
      oss << "Optimal integer solution found by branch-and-cut. ";
      oss << "Tree exhausted with no dropped subtrees, so the bound is proven.";
      result.message = oss.str();
      result.optimality_gap = 0.0;
    } else {
      result.status = SolverStatus::Feasible;
      std::ostringstream oss;
      oss << "Integer feasible solution found, but the relative gap " << result.optimality_gap
          << " still exceeds the requested mip_gap " << options_.mip_gap
          << ". Nodes remain open, so this is not proven optimal.";
      result.message = oss.str();
    }
    result.has_objective_value = true;
    result.objective_value = incumbent;
    result.primal = incumbent_x;
  } else if (any_node_lp_error) {
    // No incumbent AND some subtree was dropped due to LP failure: we must
    // not claim INFEASIBLE, since the failure may have hidden the only
    // feasible region.
    result.status = SolverStatus::NumericalError;
    result.message =
        "MILP search INCONCLUSIVE: no integer-feasible point was found, and one or more "
        "node LPs failed, so their subtrees were never explored. Infeasibility cannot be "
        "certified and no answer is available. See warnings for the per-node reason.";
  } else if (hit_node_limit) {
    // Set in the loop; keep whatever reason we recorded there.
  } else if (result.message.empty()) {
    result.status = SolverStatus::Infeasible;
    result.message =
        "MILP is infeasible: the node queue emptied with every node proved infeasible or "
        "pruned by bound, and no node LP failed.";
  }

  std::ostringstream oss;
  oss << "nodes=" << nodes << " lp_iters=" << lp_iterations << " branch_rule="
      << (options_.branch_rule == BranchRule::StrongBranching
              ? "strong"              : (options_.branch_rule == BranchRule::PseudoCost ? "pseudocost"
                                                                : "most_fractional"))
      << " parallel_strong_lp=" << (parallel_strong ? "on" : "off");
  result.warnings.push_back(oss.str());
  std::ostringstream profile;
  profile << "search_profile total=" << (now_seconds() - t_start) << "s node_lp=" << t_lp
          << "s cuts=" << t_cuts << "s heuristics=" << t_heur << "s branching=" << t_branch
          << "s dive_lps=" << dive_lps;
  result.warnings.push_back(profile.str());
  if (tree_cut_nodes > 0) {
    result.warnings.push_back("Added " + std::to_string(tree_cuts) + " validated cuts at " +
                              std::to_string(tree_cut_nodes) + " non-root nodes");
  }

  attach_debug();
  // Rejected cuts are never silent. A generator that emits invalid cuts is a
  // correctness problem, so the count goes into the summary and the individual
  // reasons follow (bounded, so a pathological generator cannot exhaust memory).
  if (cut_rejection_count > 0) {
    std::ostringstream cut_oss;
    cut_oss << "cut_validity_gate rejections=" << cut_rejection_count;
    result.warnings.push_back(cut_oss.str());
    for (const auto& r : cut_rejections) {
      result.warnings.push_back("cut_rejected: " + r);
    }
    if (cut_rejection_count > cut_rejections.size()) {
      result.warnings.push_back("cut_rejected: ... " +
                                std::to_string(cut_rejection_count - cut_rejections.size()) +
                                " further rejections suppressed");
    }
  }

  return result;
}

}  // namespace sovereign
