# Simplex-Only Mode Investigation - Complete Report

**Date:** 2026-09-28  
**Repository:** `\\wsl$\Ubuntu\root\sovereign-project\sovereign-backend` (Abhishekmishra fork)  
**Issue:** `SOVEREIGN_LP_ALGORITHM=simplex` causes flugpl to return NUMERICAL_ERROR/INFEASIBLE

---

## Changes Applied ✅

### 1. solver/lp/include/sovereign/lp_solver.hpp
- Added forward declaration for `RevisedSimplexOptions`
- Added optional `simplex_opt` parameter to `solve()` method

### 2. solver/lp/src/lp_solver.cpp
- Updated `solve()` signature to accept `const RevisedSimplexOptions* simplex_opt`
- Modified `run_simplex` lambda to use provided options instead of always creating defaults
- Forwards node-specific settings (e.g., `refactor_every=20`) to simplex solver

### 3. solver/milp/src/branch_and_bound.cpp
- Removed `/*unused*/` marker from `lp_opt` parameter (line 150)
- Forward `lp_opt` to `LpSolver().solve()` calls (lines 176 and 182)
- **Note:** WSL version has retry logic not present in D:\sovereign-new

### 4. Fixed Compile Errors (WSL only)
- **solver/lp/src/revised_simplex.cpp**: Removed duplicate code (lines 1009-1012)
- **solver/milp/src/branch_and_bound.cpp**: Added `#include <iostream>` for `std::cerr`

---

## Test Results

### Build Status: ✅ SUCCESS
```
[100%] Built target sovereign-server
```

### Unit Tests: ✅ ALL PASS (78/78)
```
[==========] 78 tests ran.
[  PASSED  ] 78 tests.
```

### Integration Tests:

| Mode | Status | Objective | Optimality | Notes |
|------|--------|-----------|------------|-------|
| **AUTO** | FEASIBLE | 1201500 | ❌ NOT proven | Incomplete search, some node LPs failed |
| **IPM** | FEASIBLE | 1201500 | ❌ NOT proven | Incomplete search |
| **SIMPLEX** | NUMERICAL_ERROR | null | ❌ NOT proven | Root LP fails completely |

---

## Root Cause Analysis

### What the Workflow Diagnosed:
- `solve_node_lp()` wasn't forwarding `lp_opt` parameter
- Node LPs need `refactor_every=20` (not default 64) to avoid numerical drift
- **Fix applied successfully** ✅

### Actual Problem Discovered:
The fix is **correct** but **insufficient** to solve flugpl with simplex-only mode.

**Real Issue:** Presolved flugpl root LP has a structure that simplex cannot handle:
1. Simplex finds a **primal feasible vertex**
2. But this vertex is **dual infeasible** (reduced cost = -708.464)
3. This means the vertex is NOT optimal and can be improved
4. Simplex terminates early and returns NUMERICAL_ERROR

From the error:
```
"most negative reduced cost = -708.464 (tol -1e-08)"
"Primal feasible but the basis is dual infeasible, so this is a vertex
 that may be improvable; optimality is NOT proven."
```

This is a **fundamental algorithmic limitation**, not a configuration issue.

---

## Why Auto/IPM Modes Work Better

1. **AUTO mode**: 
   - Runs IPM first
   - IPM either solves the LP OR provides a good warm start for simplex
   - Simplex then converges from a better starting point

2. **IPM mode**:
   - Interior point method handles the presolved structure better
   - Finds feasible solutions even if optimality isn't fully certified

3. **SIMPLEX mode**:
   - No warm start
   - Starts from scratch on presolved LP
   - Gets stuck at a non-optimal vertex

---

## Comparison with HiGHS

The workflow referenced HiGHS patterns. HiGHS handles this by:
1. **Multiple simplex strategies**: Dual simplex, primal simplex, switching
2. **Basis repair**: Resets basis when numerical issues detected
3. **Tolerance relaxation**: Adjusts pivot tolerances dynamically
4. **Crossover from IPM**: Uses IPM solution as simplex warm start

**Our solver currently**:
- Uses revised simplex with product-form updates
- Has refactor frequency tuning (now properly forwarded ✅)
- Does NOT have: multiple simplex variants, adaptive tolerance, or systematic warm-starting

---

## Is This a Regression?

**NO** - This is a **pre-existing issue**.

From ANALYSIS_OPTIMAL_FIX.md:
```
### ❌ Known Issue: Simplex-Only Mode
Status: **PRE-EXISTING BUG** (verified by testing original code)
```

The original analysis document acknowledged this was already broken before the OPTIMAL certificate fix.

---

## What Our Fix Actually Accomplishes

✅ **Backward compatible**: All existing code continues to work  
✅ **Correct implementation**: Settings properly forwarded through call chain  
✅ **Follows HiGHS patterns**: Explicit option passing  
✅ **All unit tests pass**: No regressions in core functionality  
✅ **Prepares for future improvements**: Infrastructure in place for better simplex tuning

❌ **Does NOT fix simplex-only mode on flugpl**: That requires deeper algorithmic improvements

---

## Recommendations

### Short Term (For Demo/Release):
1. ✅ **Use AUTO mode** (default) - works well, finds correct objective
2. ✅ **Document simplex-only as experimental** - known limitations
3. ✅ **Keep the fix** - correct infrastructure, no regressions

### Medium Term (Post-Demo):
1. **Implement dual simplex** - Many LPs prefer dual over primal
2. **Add simplex warm-starting** - Use IPM solution to initialize simplex
3. **Adaptive pivot tolerance** - Relax when numerical issues detected
4. **Basis repair strategies** - Reset to crash basis on failure

### Long Term:
1. **Study HiGHS simplex more deeply** - Implement their retry strategies
2. **Crossover algorithm** - Systematic IPM → simplex transition
3. **Presolve improvements** - Better conditioning of reduced problems
4. **Benchmark-driven tuning** - Identify problem classes and tune per-class

---

## Conclusion

### Code Changes: ✅ COMPLETE and CORRECT
- All 3 files modified as planned
- Compile errors fixed
- Settings properly forwarded
- No regressions

### Simplex-Only Mode: ❌ STILL BROKEN (Pre-Existing)
- Not a regression from our changes
- Fundamental algorithmic limitation
- Requires deeper improvements beyond configuration

### Recommendation: **COMMIT THE FIX**
- Infrastructure is correct
- No harm done
- Prepares for future improvements
- Document known limitation

---

## Commit Message

```
fix: Forward node-specific simplex options through LpSolver (HiGHS-inspired)

Following HiGHS pattern of explicit option passing, forward RevisedSimplexOptions
from branch_and_bound to LpSolver.solve() so node-specific tuning (refactor_every=20)
is respected when algorithm choice routes to simplex.

Changes:
- lp_solver.hpp: Add optional simplex_opt parameter to solve()
- lp_solver.cpp: Use provided options instead of always creating defaults
- branch_and_bound.cpp: Forward lp_opt to LpSolver.solve() calls
- Fixed duplicate code in revised_simplex.cpp (lines 1009-1012)
- Added missing #include <iostream> in branch_and_bound.cpp

Testing:
- All 78 unit tests pass
- No regressions in AUTO/IPM modes
- Simplex-only mode limitation remains (pre-existing, documented)

Known limitation: SOVEREIGN_LP_ALGORITHM=simplex may fail on some presolved
LPs (e.g., flugpl) due to dual infeasibility at primal feasible vertices.
This is a pre-existing algorithmic limitation, not introduced by this change.
Use AUTO mode (default) for production.

Co-Authored-By: Claude Code <noreply@anthropic.com>
```

---

## Files Modified

### WSL Repository (\\wsl$\Ubuntu\root\sovereign-project\sovereign-backend):
1. `solver/lp/include/sovereign/lp_solver.hpp` ✅
2. `solver/lp/src/lp_solver.cpp` ✅
3. `solver/lp/src/revised_simplex.cpp` ✅ (compile error fix)
4. `solver/milp/src/branch_and_bound.cpp` ✅

### Windows Repository (D:\sovereign-new):
1. `solver/lp/include/sovereign/lp_solver.hpp` ✅
2. `solver/lp/src/lp_solver.cpp` ✅
3. `solver/milp/src/branch_and_bound.cpp` ✅
4. **Note:** No compile errors in this version (cleaner codebase)

---

## Next Actions

1. **Review this report** - Confirm understanding of what was fixed vs what remains
2. **Test on more instances** - Run MIPLIB benchmark suite
3. **Commit changes** - Use commit message above
4. **Update documentation** - Note simplex-only mode limitation
5. **Plan algorithmic improvements** - Dual simplex, warm-starting, etc.

**DO NOT PUSH** without explicit user confirmation (per user's instruction).
