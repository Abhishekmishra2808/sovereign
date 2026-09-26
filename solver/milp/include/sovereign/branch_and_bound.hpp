#pragma once

#include "sovereign/types.hpp"

namespace sovereign {

enum class BranchRule {
  MostFractional = 0,
  StrongBranching = 1,
  PseudoCost = 2,
};

struct BranchAndBoundOptions {
  int max_nodes = 100000;
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
  BranchRule branch_rule = BranchRule::StrongBranching;
  int strong_branch_candidates = 4;
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
