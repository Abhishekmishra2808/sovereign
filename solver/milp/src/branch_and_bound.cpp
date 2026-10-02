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
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

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

SolverResult solve_node_lp(const OptimizationModel& node_model,
                           const RevisedSimplexOptions& /*lp_opt*/,
                           const LpBasis* warm = nullptr, LpBasis* basis_out = nullptr) {
  OptimizationModel relax = make_lp_relaxation(node_model);
  if (basis_out != nullptr) *basis_out = LpBasis{};
  if (node_dual_simplex_enabled()) {
    SolverResult dual = solve_lp_dual_simplex(relax, warm, basis_out);
    if (dual.status == SolverStatus::Optimal || dual.status == SolverStatus::Infeasible) {
      return dual;
    }
    if (basis_out != nullptr) *basis_out = LpBasis{};
  }
  // Branch bounds change at every node. The engine's one-time root presolve
  // cannot detect contradictions introduced by a later branch, and sending
  // those infeasible nodes to Phase I can make simplex cycle for 100k pivots.
  Presolver presolver;
  const PresolveResult prep = presolver.run(relax);
  if (prep.infeasible || prep.unbounded) {
    SolverResult r;
    r.status = prep.infeasible ? SolverStatus::Infeasible : SolverStatus::Unbounded;
    r.message = prep.message;
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
    return r;
  }
  // Route through LpSolver so SOVEREIGN_LP_ALGORITHM=auto|ipm|simplex applies
  // to MILP node relaxations (not just standalone LPs).
  SolverResult result = presolver.recover(LpSolver().solve(prep.reduced), prep, relax.sense);

  // Retry with pure simplex if auto mode returns NUMERICAL_ERROR
  // Sometimes IPM warmstart corrupts the simplex solve; pure simplex can succeed
  if (result.status == SolverStatus::NumericalError) {
    SolverResult simplex_result = presolver.recover(
        LpSolver().solve(prep.reduced, "simplex"), prep, relax.sense);
    if (simplex_result.status == SolverStatus::Optimal &&
        simplex_result.has_objective_value) {
      return simplex_result;
    }
  }

  return result;
}

// An integer-feasible node LP point satisfies the node's rows (cuts included)
// only to the LP tolerance, and rounding its integers moves the original rows
// further. Re-solving the continuous part with the integers fixed makes the
// incumbent satisfy the original rows. If that LP fails, the point is kept.
void polish_incumbent(const OptimizationModel& milp, std::unordered_map<std::string, double>& x,
                      double& objective, const RevisedSimplexOptions& lp_opt) {
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
  const SolverResult r = solve_node_lp(fixed, lp_opt);
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
  SolverResult result;
};

DWORD WINAPI parallel_lp_thread(LPVOID param) {
  auto* job = reinterpret_cast<ParallelLpJob*>(param);
  job->result = solve_node_lp(*job->model, *job->opt, job->warm);
  return 0;
}

void solve_two_lps_parallel(const OptimizationModel& down, const OptimizationModel& up,
                            const RevisedSimplexOptions& lp_opt, const LpBasis* warm,
                            SolverResult& down_r, SolverResult& up_r, bool enable_parallel) {
  if (!enable_parallel) {
    down_r = solve_node_lp(down, lp_opt, warm);
    up_r = solve_node_lp(up, lp_opt, warm);
    return;
  }
  ParallelLpJob jobs[2];
  jobs[0].model = &down;
  jobs[0].opt = &lp_opt;
  jobs[0].warm = warm;
  jobs[1].model = &up;
  jobs[1].opt = &lp_opt;
  jobs[1].warm = warm;
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
      up_r = solve_node_lp(up, lp_opt, warm);
    } else if (h1) {
      WaitForSingleObject(h1, INFINITE);
      CloseHandle(h1);
      down_r = solve_node_lp(down, lp_opt, warm);
      up_r = jobs[1].result;
    } else {
      down_r = solve_node_lp(down, lp_opt, warm);
      up_r = solve_node_lp(up, lp_opt, warm);
    }
  }
}
#else
void solve_two_lps_parallel(const OptimizationModel& down, const OptimizationModel& up,
                            const RevisedSimplexOptions& lp_opt, const LpBasis* warm,
                            SolverResult& down_r, SolverResult& up_r, bool) {
  down_r = solve_node_lp(down, lp_opt, warm);
  up_r = solve_node_lp(up, lp_opt, warm);
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
               std::vector<std::string>* rejections, std::size_t base_rows, bool with_cmir) {
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
    const std::string why = check_cut_validity(model, cut.constraint, reference_points,
                                               std::max(integer_tol, 1e-6));
    if (!why.empty()) {
      ++rejected;
      if (rejections != nullptr) rejections->push_back(why);
      continue;
    }
    model.constraints.push_back(cut.constraint);
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
                       std::unordered_map<std::string, PseudoCostStats>* stats) {
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
    solve_two_lps_parallel(down, up, lp_opt, warm, rd, ru, parallel_lps);

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
                            std::unordered_map<std::string, PseudoCostStats>& stats) {
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
      solve_two_lps_parallel(down, up, lp_opt, warm, rd, ru, parallel_lps);
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

  bool has_incumbent = false;
  double incumbent = 0.0;
  std::unordered_map<std::string, double> incumbent_x;
  auto offer_incumbent = [&](std::unordered_map<std::string, double> x, double objective) {
    if (!better_incumbent(sense, objective, incumbent, has_incumbent)) return false;
    polish_incumbent(model, x, objective, lp_opt);
    if (!better_incumbent(sense, objective, incumbent, has_incumbent)) return false;
    has_incumbent = true;
    incumbent = objective;
    incumbent_x = std::move(x);
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
  auto close_node = [&](double bound) { closed_bound = weaker(closed_bound, bound); };
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
  const double t_start = now_seconds();
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
      std::cerr << "[bb] nodes=" << nodes << " open=" << (pq_min.size() + pq_max.size() + plunge.size())
                << " incumbent=" << (has_incumbent ? std::to_string(incumbent) : "-")
                << " node_bound=" << node.bound << " depth=" << node.depth
                << " t=" << (now_seconds() - t_start) << "s (lp " << t_lp << " cuts " << t_cuts
                << " heur " << t_heur << " branch " << t_branch << ")\n";
    }
    if (can_prune_by_bound(sense, node.bound, incumbent, has_incumbent, options_.mip_gap)) {
      close_node(node.bound);
      continue;
    }

    OptimizationModel node_model = *node.model;
    for (const BoundChange& bc : node.bounds) {
      Variable& v = node_model.variables[static_cast<std::size_t>(bc.var)];
      v.lower_bound = bc.lower;
      v.upper_bound = bc.upper;
    }

    LpBasis node_basis;
    double t0 = now_seconds();
    SolverResult lp = solve_node_lp(node_model, lp_opt, node.basis.get(), &node_basis);
    t_lp += now_seconds() - t0;
    lp_iterations += lp.iterations;

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
      result.status = SolverStatus::Unbounded;
      result.message = "MILP relaxation unbounded.";
      result.nodes = nodes;
      result.iterations = lp_iterations;
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
      close_node(node.bound);
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
      const int rounds = (node.depth == 0) ? options_.max_cut_rounds : 1;
      for (int round = 0; round < rounds; ++round) {
        std::vector<std::string> rejections;
        const int added = apply_cuts(node_model, lp.primal, options_.integer_tol,
                                     options_.max_cuts_per_node, reference_points,
                                     &rejections, model.constraints.size(),
                                     node.depth <= options_.cmir_max_depth);
        for (auto& r : rejections) note_cut_rejection(std::move(r));
        if (added == 0) break;
        cuts_added_here += added;
        LpBasis cut_basis;
        SolverResult cut_lp = solve_node_lp(node_model, lp_opt,
                                            node_basis.empty() ? nullptr : &node_basis,
                                            &cut_basis);
        lp_iterations += cut_lp.iterations;
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
        h = diving_heuristic(node_model, lp.primal, 0, options_.integer_tol, child_basis.get(),
                             has_incumbent ? &incumbent : nullptr);
        dive_lps += h.lp_solves;
      }
      if (h.found && offer_incumbent(h.primal, h.objective)) {
        warnings.push_back("Heuristic incumbent via " + h.method);
      }
    }
    t_heur += now_seconds() - t0;

    node.bound = lp.objective_value;

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

    if (can_prune_by_bound(sense, lp.objective_value, incumbent, has_incumbent,
                           options_.mip_gap)) {
      close_node(lp.objective_value);
      continue;
    }

    int bvar = -1;
    t0 = now_seconds();
    if (options_.branch_rule == BranchRule::StrongBranching) {
      bvar = pick_reliability_branch(node_model, lp.primal, lp.objective_value,
                                     options_.integer_tol, options_.reliability_threshold,
                                     options_.max_strong_per_node, lp_opt, child_basis.get(),
                                     parallel_strong, pseudo);
    } else if (options_.branch_rule == BranchRule::FullStrong) {
      bvar = pick_strong_branch(node_model, lp.primal, lp.objective_value,
                                options_.integer_tol, options_.strong_branch_candidates,
                                lp_opt, child_basis.get(), parallel_strong, &pseudo);
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

  if (has_incumbent) {
    const double bound = weaker(global_bound(), incumbent);
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
