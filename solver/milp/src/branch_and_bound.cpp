#include "sovereign/branch_and_bound.hpp"

#include "sovereign/cuts.hpp"
#include "sovereign/heuristics.hpp"
#include "sovereign/lp_solver.hpp"
#include "sovereign/revised_simplex.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <limits>
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

struct SearchNode {
  OptimizationModel model;
  double bound = 0.0;
  int depth = 0;
  std::int64_t id = 0;
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

SolverResult solve_node_lp(const OptimizationModel& node_model,
                           const RevisedSimplexOptions& /*lp_opt*/) {
  OptimizationModel relax = make_lp_relaxation(node_model);
  // Route through LpSolver so SOVEREIGN_LP_ALGORITHM=auto|ipm|simplex applies
  // to MILP node relaxations (not just standalone LPs).
  return LpSolver().solve(relax);
}

#if defined(_WIN32)
struct ParallelLpJob {
  const OptimizationModel* model = nullptr;
  const RevisedSimplexOptions* opt = nullptr;
  SolverResult result;
};

DWORD WINAPI parallel_lp_thread(LPVOID param) {
  auto* job = reinterpret_cast<ParallelLpJob*>(param);
  job->result = solve_node_lp(*job->model, *job->opt);
  return 0;
}

void solve_two_lps_parallel(const OptimizationModel& down, const OptimizationModel& up,
                            const RevisedSimplexOptions& lp_opt, SolverResult& down_r,
                            SolverResult& up_r, bool enable_parallel) {
  if (!enable_parallel) {
    down_r = solve_node_lp(down, lp_opt);
    up_r = solve_node_lp(up, lp_opt);
    return;
  }
  ParallelLpJob jobs[2];
  jobs[0].model = &down;
  jobs[0].opt = &lp_opt;
  jobs[1].model = &up;
  jobs[1].opt = &lp_opt;
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
      up_r = solve_node_lp(up, lp_opt);
    } else if (h1) {
      WaitForSingleObject(h1, INFINITE);
      CloseHandle(h1);
      down_r = solve_node_lp(down, lp_opt);
      up_r = jobs[1].result;
    } else {
      down_r = solve_node_lp(down, lp_opt);
      up_r = solve_node_lp(up, lp_opt);
    }
  }
}
#else
void solve_two_lps_parallel(const OptimizationModel& down, const OptimizationModel& up,
                            const RevisedSimplexOptions& lp_opt, SolverResult& down_r,
                            SolverResult& up_r, bool) {
  down_r = solve_node_lp(down, lp_opt);
  up_r = solve_node_lp(up, lp_opt);
}
#endif

int apply_cuts(OptimizationModel& model, const std::unordered_map<std::string, double>& x,
               double integer_tol, int max_cuts) {
  auto covers = generate_cover_cuts(model, x, integer_tol);
  auto mirs = generate_mir_cuts(model, x, integer_tol);
  int added = 0;
  for (const auto& cut : covers) {
    if (added >= max_cuts) break;
    model.constraints.push_back(cut.constraint);
    ++added;
  }
  for (const auto& cut : mirs) {
    if (added >= max_cuts) break;
    model.constraints.push_back(cut.constraint);
    ++added;
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
                       const RevisedSimplexOptions& lp_opt, bool parallel_lps,
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
    solve_two_lps_parallel(down, up, lp_opt, rd, ru, parallel_lps);

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

  const Sense sense = model.sense;
  const bool parallel_strong = options_.parallel_workers != 1;

  bool has_incumbent = false;
  double incumbent = 0.0;
  std::unordered_map<std::string, double> incumbent_x;
  std::int64_t nodes = 0;
  std::int64_t lp_iterations = 0;
  std::int64_t next_id = 1;
  double best_bound = (sense == Sense::Minimize) ? -std::numeric_limits<double>::infinity()
                                                 : std::numeric_limits<double>::infinity();
  bool found_finite_bound = false;
  std::vector<std::string> warnings;
  std::unordered_map<std::string, PseudoCostStats> pseudo;
  bool any_node_lp_error = false;

  std::priority_queue<SearchNode, std::vector<SearchNode>, BestBoundCompareMin> pq_min;
  std::priority_queue<SearchNode, std::vector<SearchNode>, BestBoundCompareMax> pq_max;

  auto push_node = [&](SearchNode node) {
    if (sense == Sense::Minimize) pq_min.push(std::move(node));
    else pq_max.push(std::move(node));
  };
  auto empty_queue = [&]() {
    return sense == Sense::Minimize ? pq_min.empty() : pq_max.empty();
  };
  auto pop_node = [&]() {
    SearchNode n;
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
  root.model = model;
  root.model.problem_type = ProblemType::MILP;
  root.depth = 0;
  root.id = next_id++;
  root.bound = (sense == Sense::Minimize) ? -std::numeric_limits<double>::infinity()
                                          : std::numeric_limits<double>::infinity();
  push_node(std::move(root));

  while (!empty_queue()) {
    if (nodes >= options_.max_nodes) {
      result.status = has_incumbent ? SolverStatus::Feasible : SolverStatus::Error;
      result.message = "MILP node limit reached.";
      break;
    }

    SearchNode node = pop_node();
    ++nodes;
    if (can_prune_by_bound(sense, node.bound, incumbent, has_incumbent, options_.mip_gap)) {
      continue;
    }

    SolverResult lp = solve_node_lp(node.model, lp_opt);
    lp_iterations += lp.iterations;

    if (lp.status == SolverStatus::Infeasible) continue;
    if (lp.status == SolverStatus::Unbounded) {
      result.status = SolverStatus::Unbounded;
      result.message = "MILP relaxation unbounded.";
      result.nodes = nodes;
      result.iterations = lp_iterations;
      return result;
    }
    if (lp.status != SolverStatus::Optimal && lp.status != SolverStatus::Feasible) {
      // A node LP failed to solve (e.g. iteration limit, singular basis).
      // This subtree cannot be soundly pruned or explored further — record
      // the failure and downgrade the final status so we never silently
      // report OPTIMAL (or INFEASIBLE) with a dropped branch.
      any_node_lp_error = true;
      warnings.push_back("Node LP failed (subtree unsound, dropped): " + lp.message);
      if (std::getenv("SOVEREIGN_DUMP_FAILING_NODE")) {
        std::ofstream out(std::getenv("SOVEREIGN_DUMP_FAILING_NODE"));
        out << "{\n  \"problem_type\": \"LP\",\n  \"sense\": \""
            << (node.model.sense == Sense::Maximize ? "maximize" : "minimize")
            << "\",\n  \"variables\": [\n";
        for (std::size_t i = 0; i < node.model.variables.size(); ++i) {
          const auto& v = node.model.variables[i];
          out << "    {\"name\": \"" << v.name << "\", \"type\": \"continuous\", \"lower_bound\": "
              << v.lower_bound << ", \"upper_bound\": " << v.upper_bound << "}"
              << (i + 1 < node.model.variables.size() ? "," : "") << "\n";
        }
        out << "  ],\n  \"objective\": {\"linear\": {";
        bool first = true;
        for (const auto& kv : node.model.objective.linear) {
          if (!first) out << ", ";
          out << "\"" << kv.first << "\": " << kv.second;
          first = false;
        }
        out << "}},\n  \"constraints\": [\n";
        for (std::size_t ci = 0; ci < node.model.constraints.size(); ++ci) {
          const auto& c = node.model.constraints[ci];
          out << "    {\"name\": \"" << c.name << "\", \"linear\": {";
          bool f2 = true;
          for (const auto& kv : c.linear) {
            if (!f2) out << ", ";
            out << "\"" << kv.first << "\": " << kv.second;
            f2 = false;
          }
          const char* sense_str = c.sense == ConstraintSense::Le ? "<=" :
                                   c.sense == ConstraintSense::Ge ? ">=" : "=";
          out << "}, \"sense\": \"" << sense_str << "\", \"rhs\": " << c.rhs << "}"
              << (ci + 1 < node.model.constraints.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        out.close();
      }
      continue;
    }
    if (!lp.has_objective_value) continue;

    // Tree-wide branch-and-cut
    const bool cut_here = options_.enable_cuts &&
                          (node.depth == 0 || options_.cut_frequency <= 1 ||
                           (node.depth % options_.cut_frequency) == 0);
    if (cut_here) {
      const int rounds = (node.depth == 0) ? options_.max_cut_rounds : 1;
      for (int round = 0; round < rounds; ++round) {
        const int added =
            apply_cuts(node.model, lp.primal, options_.integer_tol, options_.max_cuts_per_node);
        if (added == 0) break;
        warnings.push_back("Added " + std::to_string(added) + " cuts at depth " +
                           std::to_string(node.depth) + " round " + std::to_string(round));
        lp = solve_node_lp(node.model, lp_opt);
        lp_iterations += lp.iterations;
        if (lp.status != SolverStatus::Optimal && lp.status != SolverStatus::Feasible) break;
        if (!lp.has_objective_value) break;
      }
    }

    if (options_.enable_heuristics) {
      HeuristicResult h = rounding_heuristic(node.model, lp.primal, options_.integer_tol);
      if (!h.found) h = diving_heuristic(node.model, lp.primal, 12, options_.integer_tol);
      if (h.found && better_incumbent(sense, h.objective, incumbent, has_incumbent)) {
        has_incumbent = true;
        incumbent = h.objective;
        incumbent_x = h.primal;
        warnings.push_back("Heuristic incumbent via " + h.method);
      }
    }

    node.bound = lp.objective_value;
    if (!found_finite_bound) {
      best_bound = lp.objective_value;
      found_finite_bound = true;
    } else if (sense == Sense::Minimize) {
      best_bound = std::min(best_bound, lp.objective_value);
    } else {
      best_bound = std::max(best_bound, lp.objective_value);
    }
    if (sense == Sense::Minimize && !pq_min.empty()) {
      best_bound = std::min(best_bound, pq_min.top().bound);
    } else if (sense == Sense::Maximize && !pq_max.empty()) {
      best_bound = std::max(best_bound, pq_max.top().bound);
    }
    best_bound = (sense == Sense::Minimize) ? std::min(best_bound, lp.objective_value)
                                            : std::max(best_bound, lp.objective_value);

    // Accept integer-feasible nodes BEFORE bound pruning. Pruning on
    // relative_gap <= mip_gap when the LP objective is within mip_gap of the
    // incumbent would otherwise skip recording an integer solution whose
    // objective is equal (or only epsilon-better) than the incumbent — a
    // sibling of the "equality/tolerance" class of bugs. Integrality is the
    // authority for accepting; bound pruning only applies to fractional nodes.
    if (is_integer_feasible(node.model, lp.primal, options_.integer_tol)) {
      auto x = lp.primal;
      snap_integer_primal(node.model, x, options_.integer_tol);
      if (better_incumbent(sense, lp.objective_value, incumbent, has_incumbent)) {
        has_incumbent = true;
        incumbent = lp.objective_value;
        incumbent_x = x;
      }
      continue;
    }

    if (can_prune_by_bound(sense, lp.objective_value, incumbent, has_incumbent,
                           options_.mip_gap)) {
      continue;
    }

    int bvar = -1;
    if (options_.branch_rule == BranchRule::StrongBranching) {
      bvar = pick_strong_branch(node.model, lp.primal, lp.objective_value,
                                options_.integer_tol, options_.strong_branch_candidates,
                                lp_opt, parallel_strong, &pseudo);
    } else if (options_.branch_rule == BranchRule::PseudoCost) {
      bvar = pick_pseudo_cost(node.model, lp.primal, options_.integer_tol, pseudo);
    } else {
      bvar = pick_most_fractional(node.model, lp.primal, options_.integer_tol);
    }

    if (bvar < 0) {
      auto x = lp.primal;
      snap_integer_primal(node.model, x, options_.integer_tol);
      if (better_incumbent(sense, lp.objective_value, incumbent, has_incumbent)) {
        has_incumbent = true;
        incumbent = lp.objective_value;
        incumbent_x = x;
      }
      continue;
    }

    const Variable& bv = node.model.variables[static_cast<std::size_t>(bvar)];
    const double val = lp.primal.at(bv.name);
    const double floor_v = std::floor(val);
    const double ceil_v = std::ceil(val);

    if (floor_v >= bv.lower_bound - 1e-12) {
      SearchNode down = node;
      down.id = next_id++;
      down.depth = node.depth + 1;
      down.bound = lp.objective_value;
      Variable& dv = down.model.variables[static_cast<std::size_t>(bvar)];
      dv.upper_bound = std::min(dv.upper_bound, floor_v);
      if (dv.lower_bound <= dv.upper_bound + 1e-12) push_node(std::move(down));
    }
    if (ceil_v <= bv.upper_bound + 1e-12) {
      SearchNode up = node;
      up.id = next_id++;
      up.depth = node.depth + 1;
      up.bound = lp.objective_value;
      Variable& uv = up.model.variables[static_cast<std::size_t>(bvar)];
      uv.lower_bound = std::max(uv.lower_bound, ceil_v);
      if (uv.lower_bound <= uv.upper_bound + 1e-12) push_node(std::move(up));
    }
  }

  result.nodes = nodes;
  result.iterations = lp_iterations;
  result.warnings = warnings;

  if (has_incumbent) {
    if (found_finite_bound) {
      result.optimality_gap = std::max(0.0, relative_gap(sense, best_bound, incumbent));
    }
    if (!empty_queue() && nodes >= options_.max_nodes) {
      result.status = SolverStatus::Feasible;
      result.message = "Integer feasible solution found (node limit).";
    } else if (any_node_lp_error) {
      // A subtree LP failed; we cannot certify optimality even though the
      // remaining tree was exhausted.
      result.status = SolverStatus::Feasible;
      result.message =
          "Integer feasible solution found, but one or more node LPs failed "
          "(optimality not certified).";
    } else if (result.optimality_gap <= options_.mip_gap || empty_queue()) {
      result.status = SolverStatus::Optimal;
      result.message =
          "Optimal integer solution found by branch-and-cut (strong/pseudo branching).";
      result.optimality_gap = 0.0;
    } else {
      result.status = SolverStatus::Feasible;
      result.message = "Integer feasible solution found.";
    }
    result.has_objective_value = true;
    result.objective_value = incumbent;
    result.primal = incumbent_x;
  } else if (any_node_lp_error) {
    // No incumbent AND some subtree was dropped due to LP failure: we must
    // not claim INFEASIBLE, since the failure may have hidden the only
    // feasible region.
    result.status = SolverStatus::Error;
    result.message =
        "MILP search inconclusive: no incumbent found and one or more node "
        "LPs failed, so infeasibility cannot be certified.";
  } else if (result.message.empty()) {
    result.status = SolverStatus::Infeasible;
    result.message = "MILP is infeasible.";
  }

  std::ostringstream oss;
  oss << "nodes=" << nodes << " lp_iters=" << lp_iterations << " branch_rule="
      << (options_.branch_rule == BranchRule::StrongBranching
              ? "strong"
              : (options_.branch_rule == BranchRule::PseudoCost ? "pseudocost"
                                                                : "most_fractional"))
      << " parallel_strong_lp=" << (parallel_strong ? "on" : "off");
  result.warnings.push_back(oss.str());
  return result;
}

}  // namespace sovereign
