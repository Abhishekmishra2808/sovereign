#include "sovereign/milp_solver.hpp"

#include "sovereign/branch_and_bound.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <string>

namespace sovereign {
namespace {

BranchRule parse_branch_rule(const char* s) {
  if (!s) return BranchRule::StrongBranching;
  const std::string v(s);
  if (v == "most_fractional" || v == "fractional") return BranchRule::MostFractional;
  if (v == "pseudocost" || v == "pseudo") return BranchRule::PseudoCost;
  if (v == "full_strong") return BranchRule::FullStrong;
  return BranchRule::StrongBranching;
}

bool env_flag_true(const char* name, bool default_value) {
  const char* s = std::getenv(name);
  if (!s) return default_value;
  return std::strcmp(s, "0") != 0 && std::strcmp(s, "false") != 0 &&
         std::strcmp(s, "off") != 0;
}

}  // namespace

SolverResult MilpSolver::solve(const OptimizationModel& model) const {
  BranchAndBoundOptions opt;
  opt.branch_rule = parse_branch_rule(std::getenv("SOVEREIGN_BRANCH_RULE"));
  opt.enable_cuts = env_flag_true("SOVEREIGN_ENABLE_CUTS", true);
  opt.enable_heuristics = env_flag_true("SOVEREIGN_ENABLE_HEURISTICS", true);
  if (const char* pw = std::getenv("SOVEREIGN_PARALLEL_WORKERS")) {
    opt.parallel_workers = std::atoi(pw);
  }
  if (const char* cf = std::getenv("SOVEREIGN_CUT_FREQUENCY")) {
    opt.cut_frequency = std::atoi(cf);
  } else if (model.variables.size() > 1000) {
    // Tree cuts are useful, but separating and re-solving at every node can
    // dominate wide MIPs after the root relaxation is established.
    opt.cut_frequency = 5;
  }
  if (const char* cr = std::getenv("SOVEREIGN_MAX_CUT_ROUNDS")) {
    opt.max_cut_rounds = std::max(0, std::atoi(cr));
  } else {
    // Re-solving after every root cut is useful on small models, but can
    // consume the entire time budget on sparse, wide MIPs before branching
    // begins. Keep one separation pass for larger models; callers can still
    // explicitly request more rounds through the environment.
    const std::size_t scale =
        model.variables.size() * std::max<std::size_t>(1, model.constraints.size());
    if (scale > 500000 || model.variables.size() > 4000) {
      opt.max_cut_rounds = 1;
    } else if (scale > 100000) {
      opt.max_cut_rounds = 2;
    }
  }
  if (const char* mc = std::getenv("SOVEREIGN_MAX_CUTS_PER_NODE")) {
    opt.max_cuts_per_node = std::max(0, std::atoi(mc));
  }
  if (const char* ms = std::getenv("SOVEREIGN_MAX_STRONG_PER_NODE")) {
    opt.max_strong_per_node = std::max(0, std::atoi(ms));
  } else if (model.variables.size() > 1000) {
    // Reliability branching is valuable, but probing many candidates can
    // dominate wide models. Preserve a small amount of look-ahead and let
    // pseudo-costs take over quickly.
    opt.max_strong_per_node = 2;
  }
  if (const char* rt = std::getenv("SOVEREIGN_RELIABILITY_THRESHOLD")) {
    opt.reliability_threshold = std::max(0, std::atoi(rt));
  } else if (model.variables.size() > 1000) {
    opt.reliability_threshold = 2;
  }
  if (const char* cd = std::getenv("SOVEREIGN_CMIR_MAX_DEPTH")) {
    opt.cmir_max_depth = std::atoi(cd);
  }
  if (const char* mn = std::getenv("SOVEREIGN_MAX_NODES")) {
    opt.max_nodes = std::atoi(mn);
  }
  if (const char* tl = std::getenv("SOVEREIGN_TIME_LIMIT")) {
    opt.time_limit_seconds = std::atof(tl);
  }
  if (const char* df = std::getenv("SOVEREIGN_DIVE_FREQUENCY")) {
    opt.dive_frequency = std::atoi(df);
  }
  if (const char* dd = std::getenv("SOVEREIGN_DIVE_MAX_DEPTH")) {
    opt.dive_max_depth = std::max(0, std::atoi(dd));
  } else if (model.variables.size() > 1000) {
    // A full dive over thousands of integer variables can dominate the solve
    // before branching gets a chance to produce a useful incumbent.
    opt.dive_max_depth = 50;
  }
  opt.plunging = env_flag_true("SOVEREIGN_PLUNGING", true);
  return BranchAndBoundSolver(opt).solve(model);
}

}  // namespace sovereign
