#include "sovereign/milp_solver.hpp"

#include "sovereign/branch_and_bound.hpp"

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
  opt.plunging = env_flag_true("SOVEREIGN_PLUNGING", true);
  return BranchAndBoundSolver(opt).solve(model);
}

}  // namespace sovereign
