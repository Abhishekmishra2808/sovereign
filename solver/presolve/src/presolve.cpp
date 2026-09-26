#include "sovereign/presolve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace sovereign {
namespace {

bool nearly_equal(double a, double b, double tol) {
  return std::abs(a - b) <= tol;
}

bool is_fixed(const Variable& v, double tol) {
  return nearly_equal(v.lower_bound, v.upper_bound, tol);
}

int find_var(OptimizationModel& model, const std::string& name) {
  for (std::size_t i = 0; i < model.variables.size(); ++i) {
    if (model.variables[i].name == name) return static_cast<int>(i);
  }
  return -1;
}

void remove_variable_at(OptimizationModel& model, int idx) {
  const std::string name = model.variables[static_cast<std::size_t>(idx)].name;
  model.variables.erase(model.variables.begin() + idx);
  model.objective.linear.erase(name);
  for (auto it = model.objective.quadratic.begin();
       it != model.objective.quadratic.end();) {
    if (it->first == name) {
      it = model.objective.quadratic.erase(it);
      continue;
    }
    it->second.erase(name);
    if (it->second.empty()) {
      it = model.objective.quadratic.erase(it);
    } else {
      ++it;
    }
  }
  for (auto& c : model.constraints) {
    c.linear.erase(name);
  }
}

double obj_coef(const OptimizationModel& model, const std::string& name) {
  auto it = model.objective.linear.find(name);
  return it == model.objective.linear.end() ? 0.0 : it->second;
}

bool tighten_bound(Variable& v, double lb, double ub, double tol, int* tightened) {
  bool changed = false;
  const double INF_BOUND = 1e29;
  if (lb > v.lower_bound + tol) {
    v.lower_bound = lb;
    changed = true;
    if (tightened) ++(*tightened);
  }
  // Do not invent finite upper bounds from +inf: explicit UB rows in standard
  // form would explode the basis (one row per variable) and destroy simplex scaling.
  if (ub < v.upper_bound - tol) {
    if (v.upper_bound >= INF_BOUND && ub < INF_BOUND) {
      // skip +inf -> finite UB tightening
    } else {
      v.upper_bound = ub;
      changed = true;
      if (tightened) ++(*tightened);
    }
  }
  return changed;
}

// Substitute name = coeff * other + offset into the model (name still present until removed).
void apply_affine_substitution(OptimizationModel& model, const std::string& name,
                               const std::string& other, double coeff, double offset,
                               double* objective_offset, Sense sense) {
  // Objective: c_name * (coeff*other + offset) + ...
  const double cn = obj_coef(model, name);
  if (cn != 0.0) {
    model.objective.linear[other] += cn * coeff;
    // Constant term: for reporting in original sense, accumulate cn*offset
    if (objective_offset) {
      *objective_offset += cn * offset;
    }
    (void)sense;
    model.objective.linear.erase(name);
  }

  for (auto& c : model.constraints) {
    auto it = c.linear.find(name);
    if (it == c.linear.end()) continue;
    const double a = it->second;
    c.linear.erase(it);
    if (std::abs(a * coeff) > 0.0) {
      c.linear[other] += a * coeff;
    }
    c.rhs -= a * offset;
  }
}

bool remove_empty_and_redundant(OptimizationModel& model, PresolveResult& out,
                                double tol, int* removed) {
  bool changed = false;
  std::vector<Constraint> kept;
  kept.reserve(model.constraints.size());

  for (const auto& c : model.constraints) {
    double min_act = 0.0;
    double max_act = 0.0;
    bool has_terms = false;
    for (const auto& kv : c.linear) {
      if (std::abs(kv.second) <= tol) continue;
      has_terms = true;
      const int vi = find_var(model, kv.first);
      if (vi < 0) continue;
      const Variable& v = model.variables[static_cast<std::size_t>(vi)];
      const double a = kv.second;
      if (a > 0) {
        min_act += a * v.lower_bound;
        max_act += a * v.upper_bound;
      } else {
        min_act += a * v.upper_bound;
        max_act += a * v.lower_bound;
      }
    }

    if (!has_terms) {
      // 0 ? rhs
      bool ok = true;
      if (c.sense == ConstraintSense::Le) ok = (0.0 <= c.rhs + tol);
      else if (c.sense == ConstraintSense::Ge) ok = (0.0 >= c.rhs - tol);
      else ok = nearly_equal(0.0, c.rhs, tol);
      if (!ok) {
        out.infeasible = true;
        out.message = "Presolve detected infeasible empty constraint: " + c.name;
        return true;
      }
      PresolveAction act;
      act.type = PresolveActionType::DropConstraint;
      act.name = c.name;
      act.note = "empty redundant";
      out.actions.push_back(act);
      if (removed) ++(*removed);
      changed = true;
      continue;
    }

    // Redundancy / infeasibility vs implied activity range
    if (c.sense == ConstraintSense::Le) {
      if (min_act > c.rhs + tol) {
        out.infeasible = true;
        out.message = "Presolve detected infeasible constraint: " + c.name;
        return true;
      }
      if (max_act <= c.rhs + tol) {
        PresolveAction act;
        act.type = PresolveActionType::DropConstraint;
        act.name = c.name;
        act.note = "redundant <=";
        out.actions.push_back(act);
        if (removed) ++(*removed);
        changed = true;
        continue;
      }
    } else if (c.sense == ConstraintSense::Ge) {
      if (max_act < c.rhs - tol) {
        out.infeasible = true;
        out.message = "Presolve detected infeasible constraint: " + c.name;
        return true;
      }
      if (min_act >= c.rhs - tol) {
        PresolveAction act;
        act.type = PresolveActionType::DropConstraint;
        act.name = c.name;
        act.note = "redundant >=";
        out.actions.push_back(act);
        if (removed) ++(*removed);
        changed = true;
        continue;
      }
    } else {  // Eq
      if (min_act > c.rhs + tol || max_act < c.rhs - tol) {
        out.infeasible = true;
        out.message = "Presolve detected infeasible equality: " + c.name;
        return true;
      }
      if (nearly_equal(min_act, max_act, tol) && nearly_equal(min_act, c.rhs, tol)) {
        PresolveAction act;
        act.type = PresolveActionType::DropConstraint;
        act.name = c.name;
        act.note = "redundant =";
        out.actions.push_back(act);
        if (removed) ++(*removed);
        changed = true;
        continue;
      }
    }

    kept.push_back(c);
  }

  model.constraints.swap(kept);
  return changed;
}

bool fix_fixed_variables(OptimizationModel& model, PresolveResult& out, double tol) {
  bool changed = false;
  for (;;) {
    int idx = -1;
    for (std::size_t i = 0; i < model.variables.size(); ++i) {
      if (is_fixed(model.variables[i], tol)) {
        idx = static_cast<int>(i);
        break;
      }
    }
    if (idx < 0) break;

    Variable v = model.variables[static_cast<std::size_t>(idx)];
    if (v.lower_bound > v.upper_bound + tol) {
      out.infeasible = true;
      out.message = "Presolve detected inconsistent bounds on " + v.name;
      return true;
    }
    const double val = 0.5 * (v.lower_bound + v.upper_bound);

    // Fold into objective constant and constraints, then remove.
    const double cn = obj_coef(model, v.name);
    out.objective_offset += cn * val;

    for (auto& c : model.constraints) {
      auto it = c.linear.find(v.name);
      if (it == c.linear.end()) continue;
      c.rhs -= it->second * val;
      c.linear.erase(it);
    }

    PresolveAction act;
    act.type = PresolveActionType::FixVariable;
    act.name = v.name;
    act.value = val;
    act.note = "fixed bounds";
    out.actions.push_back(act);
    ++out.stats.fixed_variables;

    remove_variable_at(model, idx);
    changed = true;
  }
  return changed;
}

bool singleton_rows(OptimizationModel& model, PresolveResult& out, double tol,
                    bool enable_substitution) {
  bool changed = false;

  for (;;) {
    bool progress = false;
    for (std::size_t ci = 0; ci < model.constraints.size(); ++ci) {
      Constraint& c = model.constraints[ci];
      std::vector<std::pair<std::string, double>> terms;
      for (const auto& kv : c.linear) {
        if (std::abs(kv.second) > tol) terms.push_back(kv);
      }
      if (terms.size() != 1) continue;

      const std::string name = terms[0].first;
      const double a = terms[0].second;
      const int vi = find_var(model, name);
      if (vi < 0) continue;
      Variable& v = model.variables[static_cast<std::size_t>(vi)];

      if (c.sense == ConstraintSense::Eq) {
        if (std::abs(a) <= tol) continue;
        const double val = c.rhs / a;
        if (val < v.lower_bound - tol || val > v.upper_bound + tol) {
          out.infeasible = true;
          out.message = "Singleton equality conflicts bounds on " + name;
          return true;
        }
        v.lower_bound = val;
        v.upper_bound = val;
        // Drop constraint; fixed-var pass will remove variable.
        PresolveAction drop;
        drop.type = PresolveActionType::DropConstraint;
        drop.name = c.name;
        drop.note = "singleton equality";
        out.actions.push_back(drop);
        model.constraints.erase(model.constraints.begin() +
                                static_cast<std::ptrdiff_t>(ci));
        ++out.stats.removed_constraints;
        progress = true;
        break;
      }

      // Inequality singleton => bound tightening
      // a x <= rhs  or  a x >= rhs
      bool absorbed = false;
      if (a > 0) {
        if (c.sense == ConstraintSense::Le) {
          absorbed = tighten_bound(v, v.lower_bound, c.rhs / a, tol,
                                   &out.stats.tightened_bounds);
          if (!absorbed && v.upper_bound <= c.rhs / a + tol) absorbed = true;
        } else {
          absorbed = tighten_bound(v, c.rhs / a, v.upper_bound, tol,
                                   &out.stats.tightened_bounds);
          if (!absorbed && v.lower_bound >= c.rhs / a - tol) absorbed = true;
        }
      } else {
        if (c.sense == ConstraintSense::Le) {
          absorbed = tighten_bound(v, c.rhs / a, v.upper_bound, tol,
                                   &out.stats.tightened_bounds);
          if (!absorbed && v.lower_bound >= c.rhs / a - tol) absorbed = true;
        } else {
          absorbed = tighten_bound(v, v.lower_bound, c.rhs / a, tol,
                                   &out.stats.tightened_bounds);
          if (!absorbed && v.upper_bound <= c.rhs / a + tol) absorbed = true;
        }
      }

      if (v.lower_bound > v.upper_bound + tol) {
        out.infeasible = true;
        out.message = "Singleton bound tightening proved infeasible on " + name;
        return true;
      }

      // Only drop if the inequality was actually absorbed into bounds.
      // (Skipping +inf→finite UB must leave the row in the model.)
      if (!absorbed) {
        continue;
      }

      PresolveAction drop;
      drop.type = PresolveActionType::DropConstraint;
      drop.name = c.name;
      drop.note = "singleton inequality absorbed into bounds";
      out.actions.push_back(drop);
      model.constraints.erase(model.constraints.begin() +
                              static_cast<std::ptrdiff_t>(ci));
      ++out.stats.removed_constraints;
      progress = true;
      break;
    }

    // Two-variable equality substitution: a x + b y = rhs => x = (-b/a) y + rhs/a
    if (enable_substitution && !progress) {
      for (std::size_t ci = 0; ci < model.constraints.size(); ++ci) {
        Constraint& c = model.constraints[ci];
        if (c.sense != ConstraintSense::Eq) continue;
        std::vector<std::pair<std::string, double>> terms;
        for (const auto& kv : c.linear) {
          if (std::abs(kv.second) > tol) terms.push_back(kv);
        }
        if (terms.size() != 2) continue;
        // Prefer substituting the variable with smaller |obj| impact / abs coeff
        std::string elim = terms[0].first;
        double a_elim = terms[0].second;
        std::string keep = terms[1].first;
        double a_keep = terms[1].second;
        if (std::abs(a_elim) < std::abs(a_keep)) {
          std::swap(elim, keep);
          std::swap(a_elim, a_keep);
        }
        if (std::abs(a_elim) <= tol) continue;

        const double coeff = -a_keep / a_elim;
        const double offset = c.rhs / a_elim;

        // Transfer bounds of eliminated variable onto the kept variable.
        const int eidx_bounds = find_var(model, elim);
        const int kidx_bounds = find_var(model, keep);
        if (eidx_bounds >= 0 && kidx_bounds >= 0) {
          const Variable& ev = model.variables[static_cast<std::size_t>(eidx_bounds)];
          Variable& kv = model.variables[static_cast<std::size_t>(kidx_bounds)];
          // elim = coeff * keep + offset
          // lb_e <= coeff*keep + offset <= ub_e
          if (std::abs(coeff) > tol) {
            double lo1 = (ev.lower_bound - offset) / coeff;
            double hi1 = (ev.upper_bound - offset) / coeff;
            if (coeff < 0) std::swap(lo1, hi1);
            tighten_bound(kv, lo1, hi1, tol, &out.stats.tightened_bounds);
            if (kv.lower_bound > kv.upper_bound + tol) {
              out.infeasible = true;
              out.message = "Substitution bounds infeasible for " + keep;
              return true;
            }
          }
        }

        apply_affine_substitution(model, elim, keep, coeff, offset,
                                  &out.objective_offset, model.sense);

        PresolveAction sub;
        sub.type = PresolveActionType::SubstituteVariable;
        sub.name = elim;
        sub.other = keep;
        sub.coeff = coeff;
        sub.value = offset;
        sub.note = "equality substitution";
        out.actions.push_back(sub);
        ++out.stats.substituted_variables;

        const int eidx = find_var(model, elim);
        if (eidx >= 0) remove_variable_at(model, eidx);

        PresolveAction drop;
        drop.type = PresolveActionType::DropConstraint;
        drop.name = c.name;
        drop.note = "substituted equality";
        out.actions.push_back(drop);
        model.constraints.erase(model.constraints.begin() +
                                static_cast<std::ptrdiff_t>(ci));
        ++out.stats.removed_constraints;
        progress = true;
        break;
      }
    }

    if (!progress) break;
    changed = true;
    if (out.infeasible) return true;
    // After bound changes, fix newly fixed vars
    fix_fixed_variables(model, out, tol);
    if (out.infeasible) return true;
  }

  return changed;
}

bool bound_tighten_from_rows(OptimizationModel& model, PresolveResult& out, double tol) {
  bool changed = false;
  const double INF_BOUND = 1e29;

  std::unordered_map<std::string, int> idx;
  idx.reserve(model.variables.size() * 2);
  for (std::size_t i = 0; i < model.variables.size(); ++i) {
    idx[model.variables[i].name] = static_cast<int>(i);
  }

  auto finite_lb = [INF_BOUND](const Variable& v) {
    return v.lower_bound > -INF_BOUND;
  };
  auto finite_ub = [INF_BOUND](const Variable& v) {
    return v.upper_bound < INF_BOUND;
  };

  for (const auto& c : model.constraints) {
    // Pre-resolve coefficients to variable indices once per row.
    std::vector<std::pair<int, double>> terms;
    terms.reserve(c.linear.size());
    for (const auto& kv : c.linear) {
      if (std::abs(kv.second) <= tol) continue;
      auto it = idx.find(kv.first);
      if (it == idx.end()) continue;
      terms.push_back({it->second, kv.second});
    }
    if (terms.size() < 2) continue;

    for (std::size_t t = 0; t < terms.size(); ++t) {
      const int vi = terms[t].first;
      const double a = terms[t].second;
      Variable& v = model.variables[static_cast<std::size_t>(vi)];

      double rest_min = 0.0;
      double rest_max = 0.0;
      bool rest_min_ok = true;
      bool rest_max_ok = true;
      for (std::size_t u = 0; u < terms.size(); ++u) {
        if (u == t) continue;
        const Variable& ov = model.variables[static_cast<std::size_t>(terms[u].first)];
        const double b = terms[u].second;
        if (b > 0) {
          if (!finite_lb(ov)) rest_min_ok = false;
          else rest_min += b * ov.lower_bound;
          if (!finite_ub(ov)) rest_max_ok = false;
          else rest_max += b * ov.upper_bound;
        } else {
          if (!finite_ub(ov)) rest_min_ok = false;
          else rest_min += b * ov.upper_bound;
          if (!finite_lb(ov)) rest_max_ok = false;
          else rest_max += b * ov.lower_bound;
        }
      }

      if (c.sense == ConstraintSense::Le || c.sense == ConstraintSense::Eq) {
        // a x <= rhs - rest_min  (needs finite rest_min)
        if (rest_min_ok) {
          if (a > 0) {
            changed |= tighten_bound(v, v.lower_bound, (c.rhs - rest_min) / a, tol,
                                     &out.stats.tightened_bounds);
          } else {
            changed |= tighten_bound(v, (c.rhs - rest_min) / a, v.upper_bound, tol,
                                     &out.stats.tightened_bounds);
          }
        }
      }
      if (c.sense == ConstraintSense::Ge || c.sense == ConstraintSense::Eq) {
        // a x >= rhs - rest_max  (needs finite rest_max)
        if (rest_max_ok) {
          if (a > 0) {
            changed |= tighten_bound(v, (c.rhs - rest_max) / a, v.upper_bound, tol,
                                     &out.stats.tightened_bounds);
          } else {
            changed |= tighten_bound(v, v.lower_bound, (c.rhs - rest_max) / a, tol,
                                     &out.stats.tightened_bounds);
          }
        }
      }

      if (v.lower_bound > v.upper_bound + tol) {
        out.infeasible = true;
        out.message = "Bound tightening proved infeasible on " + v.name;
        return true;
      }
    }
  }
  return changed;
}

void cleanup_zero_coeffs(OptimizationModel& model, double tol) {
  for (auto& c : model.constraints) {
    for (auto it = c.linear.begin(); it != c.linear.end();) {
      if (std::abs(it->second) <= tol) it = c.linear.erase(it);
      else ++it;
    }
  }
  for (auto it = model.objective.linear.begin(); it != model.objective.linear.end();) {
    if (std::abs(it->second) <= tol) it = model.objective.linear.erase(it);
    else ++it;
  }
}

bool dual_fix_unconstrained(OptimizationModel& model, PresolveResult& out, double tol) {
  // If a continuous variable does not appear in any constraint:
  // minimize: if c>0 fix at lb; if c<0 unbounded (unless ub finite then fix ub);
  // maximize: opposite.
  bool changed = false;
  std::unordered_set<std::string> appears;
  for (const auto& c : model.constraints) {
    for (const auto& kv : c.linear) {
      if (std::abs(kv.second) > tol) appears.insert(kv.first);
    }
  }

  for (std::size_t i = 0; i < model.variables.size();) {
    Variable& v = model.variables[i];
    if (appears.count(v.name)) {
      ++i;
      continue;
    }
    const double c = obj_coef(model, v.name);
    const bool maximize = (model.sense == Sense::Maximize);
    double fix = v.lower_bound;
    if (maximize) {
      if (c > tol) {
        if (!(v.upper_bound < 1e29)) {
          out.unbounded = true;
          out.message = "Presolve: unbounded variable " + v.name;
          return true;
        }
        fix = v.upper_bound;
      } else if (c < -tol) {
        fix = v.lower_bound;
      } else {
        fix = v.lower_bound;  // zero cost
      }
    } else {
      if (c > tol) {
        fix = v.lower_bound;
      } else if (c < -tol) {
        if (!(v.upper_bound < 1e29)) {
          out.unbounded = true;
          out.message = "Presolve: unbounded variable " + v.name;
          return true;
        }
        fix = v.upper_bound;
      } else {
        fix = v.lower_bound;
      }
    }
    v.lower_bound = fix;
    v.upper_bound = fix;
    changed = true;
    fix_fixed_variables(model, out, tol);
    if (out.infeasible || out.unbounded) return true;
    i = 0;  // restart; indices changed
    appears.clear();
    for (const auto& c : model.constraints) {
      for (const auto& kv : c.linear) {
        if (std::abs(kv.second) > tol) appears.insert(kv.first);
      }
    }
  }
  return changed;
}

}  // namespace

Presolver::Presolver(PresolveOptions options) : options_(std::move(options)) {}

PresolveResult Presolver::run(const OptimizationModel& model) const {
  PresolveResult out;
  out.reduced = model;
  const double tol = options_.tolerance;

  // Phase 2: full LP presolve only. QP/MILP keep the model intact for later phases.
  if (model.problem_type != ProblemType::LP) {
    out.message = "Presolve skipped for non-LP problem type.";
    return out;
  }

  cleanup_zero_coeffs(out.reduced, tol);

  for (int pass = 0; pass < options_.max_passes; ++pass) {
    bool changed = false;
    ++out.stats.passes;

    for (auto& v : out.reduced.variables) {
      if (v.lower_bound > v.upper_bound + tol) {
        out.infeasible = true;
        out.message = "Inconsistent bounds on " + v.name;
        return out;
      }
    }

    changed |= fix_fixed_variables(out.reduced, out, tol);
    if (out.infeasible) return out;

    if (options_.enable_singleton) {
      changed |= singleton_rows(out.reduced, out, tol, options_.enable_substitution);
      if (out.infeasible) return out;
    }

    if (options_.enable_bound_tightening) {
      changed |= bound_tighten_from_rows(out.reduced, out, tol);
      if (out.infeasible) return out;
      changed |= fix_fixed_variables(out.reduced, out, tol);
      if (out.infeasible) return out;
    }

    if (options_.enable_redundant) {
      int rem = 0;
      changed |= remove_empty_and_redundant(out.reduced, out, tol, &rem);
      out.stats.removed_constraints += rem;
      if (out.infeasible) return out;
    }

    changed |= dual_fix_unconstrained(out.reduced, out, tol);
    if (out.infeasible || out.unbounded) return out;

    cleanup_zero_coeffs(out.reduced, tol);

    if (!changed) break;
  }

  // Ensure at least one variable remains for solver schema; if all fixed, keep empty
  // and let engine/solver handle empty variable case via recovery-only path.
  if (out.reduced.variables.empty() && !out.infeasible && !out.unbounded) {
    out.message = "Presolve fixed all variables.";
  } else if (out.message.empty()) {
    std::ostringstream oss;
    oss << "Presolve: fixed=" << out.stats.fixed_variables
        << " subst=" << out.stats.substituted_variables
        << " dropped_cons=" << out.stats.removed_constraints
        << " bound_tights=" << out.stats.tightened_bounds
        << " passes=" << out.stats.passes;
    out.message = oss.str();
  }

  return out;
}

SolverResult Presolver::recover(const SolverResult& reduced_result,
                                const PresolveResult& prep,
                                Sense original_sense) const {
  SolverResult out = reduced_result;
  if (reduced_result.status != SolverStatus::Optimal &&
      reduced_result.status != SolverStatus::Feasible) {
    return out;
  }

  std::unordered_map<std::string, double> x = reduced_result.primal;

  // Actions were recorded in forward order; apply reverse for recovery.
  for (auto it = prep.actions.rbegin(); it != prep.actions.rend(); ++it) {
    const PresolveAction& a = *it;
    if (a.type == PresolveActionType::FixVariable) {
      x[a.name] = a.value;
    } else if (a.type == PresolveActionType::SubstituteVariable) {
      const double other = x.count(a.other) ? x[a.other] : 0.0;
      x[a.name] = a.coeff * other + a.value;
    }
  }

  out.primal = x;

  // Recompute objective in original model sense using recovered primal and offset.
  // Prefer recomputing from reduced objective + offset when possible.
  if (reduced_result.has_objective_value) {
    out.has_objective_value = true;
    out.objective_value = reduced_result.objective_value + prep.objective_offset;
    (void)original_sense;
  }

  if (!prep.message.empty()) {
    out.warnings.push_back(prep.message);
  }
  return out;
}

}  // namespace sovereign
