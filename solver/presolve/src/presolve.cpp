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

constexpr double kInfBound = 1e29;

// Implied bounds are recomputed pass after pass, so rows with large
// coefficients or bounds accumulate rounding error far above `tol`.
// Infeasibility is only claimed once a violation clears this margin.
double infeasibility_margin(double scale) {
  return 1e-6 * (1.0 + std::abs(scale));
}

// Bounds crossing by less than the margin are rounding noise: pin the variable
// at the midpoint instead of declaring the model infeasible.
bool bounds_conflict(Variable& v) {
  if (v.lower_bound <= v.upper_bound) return false;
  const double scale = std::max(std::abs(v.lower_bound), std::abs(v.upper_bound));
  if (v.lower_bound - v.upper_bound > infeasibility_margin(scale)) return true;
  const double mid = 0.5 * (v.lower_bound + v.upper_bound);
  v.lower_bound = mid;
  v.upper_bound = mid;
  return false;
}

std::unordered_map<std::string, int> variable_index(const OptimizationModel& model) {
  std::unordered_map<std::string, int> index;
  index.reserve(model.variables.size() * 2);
  for (std::size_t i = 0; i < model.variables.size(); ++i) {
    index.emplace(model.variables[i].name, static_cast<int>(i));
  }
  return index;
}

// Drops the constraints marked in `drop`, keeping the order of the rest.
void compact_constraints(OptimizationModel& model, const std::vector<char>& drop) {
  std::size_t out = 0;
  for (std::size_t i = 0; i < model.constraints.size(); ++i) {
    if (drop[i]) continue;
    if (out != i) model.constraints[out] = std::move(model.constraints[i]);
    ++out;
  }
  model.constraints.resize(out);
}

// Removes the named variables from the variable list, the objective and every
// constraint in one pass each.
void remove_variables(OptimizationModel& model, const std::unordered_set<std::string>& names) {
  if (names.empty()) return;
  std::size_t out = 0;
  for (std::size_t i = 0; i < model.variables.size(); ++i) {
    if (names.count(model.variables[i].name)) continue;
    if (out != i) model.variables[out] = std::move(model.variables[i]);
    ++out;
  }
  model.variables.resize(out);
  for (const auto& name : names) model.objective.linear.erase(name);
  for (auto it = model.objective.quadratic.begin(); it != model.objective.quadratic.end();) {
    if (names.count(it->first)) {
      it = model.objective.quadratic.erase(it);
      continue;
    }
    for (const auto& name : names) it->second.erase(name);
    if (it->second.empty()) it = model.objective.quadratic.erase(it);
    else ++it;
  }
  for (auto& c : model.constraints) {
    for (auto it = c.linear.begin(); it != c.linear.end();) {
      if (names.count(it->first)) it = c.linear.erase(it);
      else ++it;
    }
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

// Substitute name = coeff * other + offset into the objective and into the
// constraints listed for `name` in `rows_of` (a column index of the
// constraints, kept up to date as `other` enters new rows).
void apply_affine_substitution(OptimizationModel& model, const std::string& name,
                               const std::string& other, double coeff, double offset,
                               double* objective_offset,
                               std::unordered_map<std::string, std::vector<int>>& rows_of) {
  // Objective: c_name * (coeff*other + offset) + ...
  const double cn = obj_coef(model, name);
  if (cn != 0.0) {
    model.objective.linear[other] += cn * coeff;
    // Constant term: for reporting in original sense, accumulate cn*offset
    if (objective_offset) {
      *objective_offset += cn * offset;
    }
    model.objective.linear.erase(name);
  }

  auto rows = rows_of.find(name);
  if (rows == rows_of.end()) return;
  // References survive the rehash that rows_of[other] may trigger; iterators do not.
  const std::vector<int>& name_rows = rows->second;
  std::vector<int>& other_rows = rows_of[other];
  for (int r : name_rows) {
    Constraint& c = model.constraints[static_cast<std::size_t>(r)];
    auto it = c.linear.find(name);
    if (it == c.linear.end()) continue;
    const double a = it->second;
    c.linear.erase(it);
    if (std::abs(a * coeff) > 0.0) {
      auto ins = c.linear.emplace(other, 0.0);
      if (ins.second) other_rows.push_back(r);
      ins.first->second += a * coeff;
    }
    c.rhs -= a * offset;
  }
  rows_of.erase(name);
}

bool remove_empty_and_redundant(OptimizationModel& model, PresolveResult& out,
                                double tol, int* removed) {
  bool changed = false;
  std::vector<Constraint> kept;
  kept.reserve(model.constraints.size());
  const std::unordered_map<std::string, int> var_index = variable_index(model);

  for (auto& c : model.constraints) {
    double min_act = 0.0;
    double max_act = 0.0;
    double magnitude = std::abs(c.rhs);
    bool has_terms = false;
    for (const auto& kv : c.linear) {
      if (std::abs(kv.second) <= tol) continue;
      has_terms = true;
      const auto found = var_index.find(kv.first);
      if (found == var_index.end()) continue;
      const Variable& v = model.variables[static_cast<std::size_t>(found->second)];
      const double a = kv.second;
      if (a > 0) {
        min_act += a * v.lower_bound;
        max_act += a * v.upper_bound;
      } else {
        min_act += a * v.upper_bound;
        max_act += a * v.lower_bound;
      }
      if (std::abs(v.lower_bound) < kInfBound) magnitude = std::max(magnitude, std::abs(a * v.lower_bound));
      if (std::abs(v.upper_bound) < kInfBound) magnitude = std::max(magnitude, std::abs(a * v.upper_bound));
    }
    const double margin = infeasibility_margin(magnitude);

    if (!has_terms) {
      // 0 ? rhs
      bool ok = true;
      if (c.sense == ConstraintSense::Le) ok = (0.0 <= c.rhs + margin);
      else if (c.sense == ConstraintSense::Ge) ok = (0.0 >= c.rhs - margin);
      else ok = nearly_equal(0.0, c.rhs, margin);
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
      if (min_act > c.rhs + margin) {
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
      if (max_act < c.rhs - margin) {
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
      if (min_act > c.rhs + margin || max_act < c.rhs - margin) {
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

    kept.push_back(std::move(c));
  }

  model.constraints.swap(kept);
  return changed;
}

bool fix_fixed_variables(OptimizationModel& model, PresolveResult& out, double tol) {
  // Removing a variable never fixes another, so every currently fixed variable
  // is folded out in one sweep over the constraints.
  std::unordered_map<std::string, double> fixed;
  std::vector<char> drop(model.variables.size(), 0);
  for (std::size_t i = 0; i < model.variables.size(); ++i) {
    const Variable& v = model.variables[i];
    if (!is_fixed(v, tol) || fixed.count(v.name)) continue;
    if (v.lower_bound > v.upper_bound + tol) {
      out.infeasible = true;
      out.message = "Presolve detected inconsistent bounds on " + v.name;
      return true;
    }
    const double val = 0.5 * (v.lower_bound + v.upper_bound);
    out.objective_offset += obj_coef(model, v.name) * val;

    PresolveAction act;
    act.type = PresolveActionType::FixVariable;
    act.name = v.name;
    act.value = val;
    act.note = "fixed bounds";
    out.actions.push_back(act);
    ++out.stats.fixed_variables;

    fixed.emplace(v.name, val);
    drop[i] = 1;
  }
  if (fixed.empty()) return false;

  for (auto& c : model.constraints) {
    for (auto it = c.linear.begin(); it != c.linear.end();) {
      const auto f = fixed.find(it->first);
      if (f == fixed.end()) {
        ++it;
        continue;
      }
      c.rhs -= it->second * f->second;
      it = c.linear.erase(it);
    }
  }
  for (const auto& kv : fixed) model.objective.linear.erase(kv.first);
  for (auto it = model.objective.quadratic.begin(); it != model.objective.quadratic.end();) {
    if (fixed.count(it->first)) {
      it = model.objective.quadratic.erase(it);
      continue;
    }
    for (const auto& kv : fixed) it->second.erase(kv.first);
    if (it->second.empty()) it = model.objective.quadratic.erase(it);
    else ++it;
  }
  // Self-move-assignment empties a std::string under libstdc++, which erased
  // the names of every kept variable before the first dropped one.
  std::size_t out_i = 0;
  for (std::size_t i = 0; i < model.variables.size(); ++i) {
    if (drop[i]) continue;
    if (out_i != i) model.variables[out_i] = std::move(model.variables[i]);
    ++out_i;
  }
  model.variables.resize(out_i);
  return true;
}

bool singleton_rows(OptimizationModel& model, PresolveResult& out, double tol,
                    bool enable_substitution) {
  bool changed = false;

  // Every pass handles all singleton rows in one sweep (then, if there were
  // none, all two-variable equalities), so the work per pass is linear in the
  // model instead of one full rescan per reduction.
  for (;;) {
    bool progress = false;
    std::unordered_map<std::string, int> index = variable_index(model);
    std::vector<char> drop_row(model.constraints.size(), 0);
    std::vector<std::size_t> pairs;  // two-term equalities, for substitution
    for (std::size_t ci = 0; ci < model.constraints.size(); ++ci) {
      Constraint& c = model.constraints[ci];
      const std::pair<const std::string, double>* term = nullptr;
      int count = 0;
      for (const auto& kv : c.linear) {
        if (std::abs(kv.second) <= tol) continue;
        if (++count > 2) break;
        if (!term) term = &kv;
      }
      if (count == 2 && c.sense == ConstraintSense::Eq) pairs.push_back(ci);
      if (count != 1) continue;

      const std::string name = term->first;
      const double a = term->second;
      const auto found = index.find(name);
      if (found == index.end()) continue;
      Variable& v = model.variables[static_cast<std::size_t>(found->second)];

      if (c.sense == ConstraintSense::Eq) {
        if (std::abs(a) <= tol) continue;
        double val = c.rhs / a;
        const double margin = infeasibility_margin(val);
        if (val < v.lower_bound - margin || val > v.upper_bound + margin) {
          out.infeasible = true;
          out.message = "Singleton equality conflicts bounds on " + name;
          return true;
        }
        val = std::min(std::max(val, v.lower_bound), v.upper_bound);
        v.lower_bound = val;
        v.upper_bound = val;
        // Drop constraint; fixed-var pass will remove variable.
        PresolveAction drop;
        drop.type = PresolveActionType::DropConstraint;
        drop.name = c.name;
        drop.note = "singleton equality";
        out.actions.push_back(drop);
        drop_row[ci] = 1;
        ++out.stats.removed_constraints;
        progress = true;
        continue;
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

      if (bounds_conflict(v)) {
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
      drop_row[ci] = 1;
      ++out.stats.removed_constraints;
      progress = true;
    }
    if (progress) compact_constraints(model, drop_row);

    // Two-variable equality substitution: a x + b y = rhs => x = (-b/a) y + rhs/a
    if (enable_substitution && !progress && !pairs.empty()) {
      // Column index restricted to the variables of two-term equalities: only
      // those are substituted or receive the substituted terms.
      std::unordered_map<std::string, std::vector<int>> rows_of;
      for (std::size_t ci : pairs) {
        for (const auto& kv : model.constraints[ci].linear) rows_of.emplace(kv.first, std::vector<int>());
      }
      for (std::size_t ci = 0; ci < model.constraints.size(); ++ci) {
        for (const auto& kv : model.constraints[ci].linear) {
          const auto found = rows_of.find(kv.first);
          if (found != rows_of.end()) found->second.push_back(static_cast<int>(ci));
        }
      }
      std::unordered_set<std::string> eliminated;
      for (std::size_t ci : pairs) {
        Constraint& c = model.constraints[ci];
        std::vector<std::pair<std::string, double>> terms;
        for (const auto& kv : c.linear) {
          if (std::abs(kv.second) > tol) terms.push_back(kv);
        }
        // An earlier substitution in this sweep may have changed the row.
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

        // The equality row is dropped, so elim's bounds survive only as bounds
        // on keep: lb_e <= coeff*keep + offset <= ub_e. They are intersected
        // exactly; tighten_bound's refusal to create finite upper bounds would
        // silently drop them and let the recovered elim leave its range.
        const auto eidx_bounds = index.find(elim);
        const auto kidx_bounds = index.find(keep);
        if (eidx_bounds == index.end() || kidx_bounds == index.end() || std::abs(coeff) <= tol) continue;
        {
          const Variable& ev = model.variables[static_cast<std::size_t>(eidx_bounds->second)];
          Variable& kv = model.variables[static_cast<std::size_t>(kidx_bounds->second)];
          double lo1 = ev.lower_bound > -kInfBound ? (ev.lower_bound - offset) / coeff : -1e30;
          double hi1 = ev.upper_bound < kInfBound ? (ev.upper_bound - offset) / coeff : 1e30;
          if (coeff < 0) {
            lo1 = ev.upper_bound < kInfBound ? (ev.upper_bound - offset) / coeff : -1e30;
            hi1 = ev.lower_bound > -kInfBound ? (ev.lower_bound - offset) / coeff : 1e30;
          }
          if (lo1 > kv.lower_bound) {
            kv.lower_bound = lo1;
            ++out.stats.tightened_bounds;
          }
          if (hi1 < kv.upper_bound) {
            kv.upper_bound = hi1;
            ++out.stats.tightened_bounds;
          }
          if (bounds_conflict(kv)) {
            out.infeasible = true;
            out.message = "Substitution bounds infeasible for " + keep;
            return true;
          }
        }

        apply_affine_substitution(model, elim, keep, coeff, offset,
                                  &out.objective_offset, rows_of);

        PresolveAction sub;
        sub.type = PresolveActionType::SubstituteVariable;
        sub.name = elim;
        sub.other = keep;
        sub.coeff = coeff;
        sub.value = offset;
        sub.note = "equality substitution";
        out.actions.push_back(sub);
        ++out.stats.substituted_variables;

        if (index.count(elim)) eliminated.insert(elim);

        PresolveAction drop;
        drop.type = PresolveActionType::DropConstraint;
        drop.name = c.name;
        drop.note = "substituted equality";
        out.actions.push_back(drop);
        drop_row[ci] = 1;
        ++out.stats.removed_constraints;
        progress = true;
      }
      if (progress) {
        compact_constraints(model, drop_row);
        remove_variables(model, eliminated);
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
    double max_coef = 0.0;
    for (const auto& term : terms) max_coef = std::max(max_coef, std::abs(term.second));

    // Row activity bounds: the finite part of each sum and how many terms are
    // unbounded. The rest of the row for term t is the total minus t's own
    // contribution, so a row costs O(length) instead of O(length^2); totals are
    // updated whenever a bound in the row is tightened.
    struct Contribution {
      double min = 0.0, max = 0.0;
      int min_inf = 0, max_inf = 0;
    };
    auto contribution = [&](const Variable& ov, double b) {
      Contribution k;
      const bool lb_ok = finite_lb(ov), ub_ok = finite_ub(ov);
      if (b > 0) {
        if (lb_ok) k.min = b * ov.lower_bound; else k.min_inf = 1;
        if (ub_ok) k.max = b * ov.upper_bound; else k.max_inf = 1;
      } else {
        if (ub_ok) k.min = b * ov.upper_bound; else k.min_inf = 1;
        if (lb_ok) k.max = b * ov.lower_bound; else k.max_inf = 1;
      }
      return k;
    };
    Contribution total;
    double magnitude = 0.0;
    for (const auto& term : terms) {
      const Contribution k = contribution(model.variables[static_cast<std::size_t>(term.first)], term.second);
      total.min += k.min;
      total.max += k.max;
      total.min_inf += k.min_inf;
      total.max_inf += k.max_inf;
      magnitude += std::abs(k.min) + std::abs(k.max);
    }
    // total - own cancels badly when some finite bound is huge, so such rows
    // sum the rest term by term (and very long ones are left alone).
    const bool exact = magnitude > 1e5;
    if (exact && terms.size() > 2000) continue;

    for (std::size_t t = 0; t < terms.size(); ++t) {
      const int vi = terms[t].first;
      const double a = terms[t].second;
      Variable& v = model.variables[static_cast<std::size_t>(vi)];

      const Contribution own = contribution(v, a);
      Contribution rest{total.min - own.min, total.max - own.max, total.min_inf - own.min_inf,
                        total.max_inf - own.max_inf};
      if (exact) {
        rest = Contribution();
        for (std::size_t u = 0; u < terms.size(); ++u) {
          if (u == t) continue;
          const Contribution k = contribution(model.variables[static_cast<std::size_t>(terms[u].first)], terms[u].second);
          rest.min += k.min;
          rest.max += k.max;
          rest.min_inf += k.min_inf;
          rest.max_inf += k.max_inf;
        }
      }
      const double rest_min = rest.min;
      const double rest_max = rest.max;
      const bool rest_min_ok = rest.min_inf == 0;
      const bool rest_max_ok = rest.max_inf == 0;
      // Dividing by a coefficient far smaller than the row's largest one
      // magnifies the rounding error in rest_min/rest_max into the bound.
      const bool stable = std::abs(a) >= 1e-3 * max_coef;
      // Only improvements that clear rounding noise are applied; tiny ones
      // just feed error into the next pass.
      auto apply = [&](double lb, double ub, double candidate) {
        if (std::abs(candidate) > 1e9) return false;
        return tighten_bound(v, lb, ub, infeasibility_margin(candidate), &out.stats.tightened_bounds);
      };

      if (stable && (c.sense == ConstraintSense::Le || c.sense == ConstraintSense::Eq)) {
        // a x <= rhs - rest_min  (needs finite rest_min)
        if (rest_min_ok) {
          const double cand = (c.rhs - rest_min) / a;
          changed |= a > 0 ? apply(v.lower_bound, cand, cand) : apply(cand, v.upper_bound, cand);
        }
      }
      if (stable && (c.sense == ConstraintSense::Ge || c.sense == ConstraintSense::Eq)) {
        // a x >= rhs - rest_max  (needs finite rest_max)
        if (rest_max_ok) {
          const double cand = (c.rhs - rest_max) / a;
          changed |= a > 0 ? apply(cand, v.upper_bound, cand) : apply(v.lower_bound, cand, cand);
        }
      }

      if (bounds_conflict(v)) {
        out.infeasible = true;
        out.message = "Bound tightening proved infeasible on " + v.name;
        return true;
      }
      const Contribution now = contribution(v, a);
      total.min += now.min - own.min;
      total.max += now.max - own.max;
      total.min_inf += now.min_inf - own.min_inf;
      total.max_inf += now.max_inf - own.max_inf;
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
  const std::unordered_map<std::string, int> index = variable_index(model);
  std::vector<char> appears(model.variables.size(), 0);
  for (const auto& c : model.constraints) {
    for (const auto& kv : c.linear) {
      if (std::abs(kv.second) <= tol) continue;
      const auto found = index.find(kv.first);
      if (found != index.end()) appears[static_cast<std::size_t>(found->second)] = 1;
    }
  }

  // Fixing a variable that is in no constraint cannot change where any other
  // variable appears, so all of them are fixed in one sweep and folded out once.
  for (std::size_t i = 0; i < model.variables.size(); ++i) {
    Variable& v = model.variables[i];
    if (appears[i] || is_fixed(v, tol)) continue;
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
  }
  if (changed) {
    fix_fixed_variables(model, out, tol);
    if (out.infeasible || out.unbounded) return true;
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
      if (bounds_conflict(v)) {
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
