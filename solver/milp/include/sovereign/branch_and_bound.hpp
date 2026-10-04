#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

enum class BranchRule {
  MostFractional = 0,
  // Reliability branching: strong branching only on candidates whose
  // pseudocosts have fewer than `reliability_threshold` observations per
  // direction, pseudocost scores for the rest.
  StrongBranching = 1,
  PseudoCost = 2,
  // Strong branching on the top `strong_branch_candidates` at every node.
  FullStrong = 3,
};

struct BranchAndBoundOptions {
  int max_nodes = 100000;
  // Wall-clock budget for the tree search; 0 = none. On expiry the incumbent is
  // reported as FEASIBLE (or TIME_LIMIT without one), like the node limit.
  double time_limit_seconds = 0.0;
  double mip_gap = 1e-6;
  double integer_tol = 1e-6;
  double feasibility_tol = 1e-8;
  bool prefer_best_bound = true;
  bool enable_cuts = true;
  bool enable_heuristics = true;
  int max_cut_rounds = 5;
  // Tree-wide branch-and-cut: generate cuts at nodes with depth % cut_frequency == 0
  // (0 = every node; 1 = every node; k = every k depths). Root always eligible.
  int cut_frequency = 1;
  int max_cuts_per_node = 20;
  // MIR separation costs an LP re-solve whenever it finds a cut, so below this
  // depth the tree keeps only the cheap cover and rounding cuts.
  // CMIR candidates remain available for explicit ablations, but the default
  // path keeps them disabled until each candidate can be certified
  // independently of a known incumbent. A cut that is merely plausible can
  // invalidate a global MILP bound before an incumbent exists.
  int cmir_max_depth = -1;
  BranchRule branch_rule = BranchRule::StrongBranching;
  int strong_branch_candidates = 4;
  int reliability_threshold = 4;
  int max_strong_per_node = 8;
  // Diving runs at the root, then every `dive_frequency` nodes (5x more often
  // while no incumbent exists). 0 disables tree dives.
  int dive_frequency = 50;
  // Maximum integer fixings attempted by one diving heuristic. 0 means all
  // integer variables (legacy behavior); the MILP entry point supplies an
  // adaptive cap for wide models.
  int dive_max_depth = 0;
  // After branching, continue with the preferred child before returning to the
  // best-bound queue.
  bool plunging = true;
  // Multi-core parallel node processing (1 = serial)
  // Multi-core: parallel strong-branch child LP solves when != 1 (Win32 threads).
  // 0 => enable parallel strong branching; 1 => serial.
  int parallel_workers = 0;
};

class BranchAndBoundSolver {
 public:
  explicit BranchAndBoundSolver(BranchAndBoundOptions options = {});

  SolverResult solve(const OptimizationModel& model) const;

 private:
  BranchAndBoundOptions options_;
};

}  // namespace sovereign
