#pragma once

#include "sovereign/types.hpp"

#include <cstdint>
#include <vector>

namespace sovereign {

// Status of one column of the computational form [A | -I] z = 0: structural
// columns first (model.variables order), then one logical per constraint row
// (model.constraints order).
enum class BasisStatus : std::uint8_t { Basic, AtLower, AtUpper, Free };

// A simplex basis that can be handed from a branch-and-bound parent to its
// children. Children differ only in variable bounds and in cut rows appended
// after the parent's rows, so a saved basis stays structurally valid: rows the
// basis does not know about start with their logical basic.
struct LpBasis {
  std::vector<BasisStatus> cols;
  std::vector<BasisStatus> rows;
  bool empty() const { return cols.empty(); }
};

struct DualSimplexOptions {
  int max_iterations = 0;       // 0 = automatic, scaled with problem size
  int refactor_every = 50;
  double primal_tol = 1e-7;     // in the scaled problem
  double dual_tol = 1e-7;
  double pivot_tol = 1e-9;
  // Dual steepest-edge pricing (weights start at 1 and are updated exactly
  // from then on); false = largest primal infeasibility.
  bool steepest_edge = true;
};

// Bounded-variable dual simplex for an LP (variable types are ignored).
//
// Variable bounds are handled implicitly instead of as extra rows, and `warm`
// (optional) restarts from a previous optimal basis, which is dual feasible for
// any tightening of bounds or appended rows — exactly what branching and
// cutting produce. The result is OPTIMAL or INFEASIBLE only when certified
// (INFEASIBLE via an explicit Farkas-style row check); every other status means
// "use another method", and the caller is expected to fall back.
SolverResult solve_lp_dual_simplex(const OptimizationModel& lp, const LpBasis* warm,
                                   LpBasis* basis_out,
                                   const DualSimplexOptions& options = {});

}  // namespace sovereign
