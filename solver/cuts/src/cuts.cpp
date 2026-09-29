#include "sovereign/cuts.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace sovereign {
namespace {

bool is_bin(VariableType t) { return t == VariableType::Binary; }
bool is_int(VariableType t) {
  return t == VariableType::Integer || t == VariableType::Binary;
}

const Variable* find_var(const OptimizationModel& m, const std::string& name) {
  for (const auto& v : m.variables) {
    if (v.name == name) return &v;
  }
  return nullptr;
}

}  // namespace

std::string check_cut_validity(
    const OptimizationModel& milp, const Constraint& cut,
    const std::vector<std::unordered_map<std::string, double>>& reference_points, double tol) {
  // 1. The cut must not remove a point we already know is feasible.
  //    This is the strongest available evidence of validity, and it is cheap.
  for (const auto& p : reference_points) {
    double lhs = 0.0;
    bool complete = true;
    for (const auto& kv : cut.linear) {
      auto it = p.find(kv.first);
      if (it == p.end()) {
        complete = false;
        break;
      }
      lhs += kv.second * it->second;
    }
    if (!complete) continue;
    double viol = 0.0;
    if (cut.sense == ConstraintSense::Le) {
      viol = lhs - cut.rhs;
    } else if (cut.sense == ConstraintSense::Ge) {
      viol = cut.rhs - lhs;
    } else {
      viol = std::abs(lhs - cut.rhs);
    }
    if (viol > tol) {
      std::ostringstream oss;
      oss << "cut '" << cut.name << "' removes a known integer-feasible point by "
          << viol << " (lhs " << lhs << " vs rhs " << cut.rhs << ")";
      return oss.str();
    }
  }

  // 2. Every variable must exist, and the cut must be satisfiable somewhere in
  //    the variables' bound box. A cut that no point of the box satisfies is
  //    malformed whatever the generator intended. (Coefficient signs and sizes
  //    say nothing on their own: valid MIR cuts put coefficients above 1, or
  //    negative ones after complementing, on binaries.)
  double box_min = 0.0;
  double box_max = 0.0;
  for (const auto& kv : cut.linear) {
    const Variable* v = find_var(milp, kv.first);
    if (v == nullptr) {
      return "cut '" + cut.name + "' references unknown variable " + kv.first;
    }
    const double lo = kv.second * (kv.second > 0.0 ? v->lower_bound : v->upper_bound);
    const double hi = kv.second * (kv.second > 0.0 ? v->upper_bound : v->lower_bound);
    box_min += std::max(lo, -1e30);
    box_max += std::min(hi, 1e30);
  }
  const bool unsatisfiable =
      (cut.sense != ConstraintSense::Ge && box_min > cut.rhs + tol) ||
      (cut.sense != ConstraintSense::Le && box_max < cut.rhs - tol);
  if (unsatisfiable) {
    std::ostringstream oss;
    oss << "cut '" << cut.name << "' is infeasible: no point within the variable bounds "
        << "satisfies it (activity range [" << box_min << ", " << box_max << "], rhs "
        << cut.rhs << ")";
    return oss.str();
  }

  return std::string();
}

std::vector<Cut> generate_cover_cuts(const OptimizationModel& milp,
                                     const std::unordered_map<std::string, double>& x,
                                     double int_tol, int max_cuts) {
  std::vector<Cut> cuts;
  int cid = 0;
  for (const auto& row : milp.constraints) {
    if (row.sense != ConstraintSense::Le) continue;
    // Collect binary terms with positive coeffs
    struct Term {
      std::string name;
      double a;
      double xv;
    };
    std::vector<Term> terms;
    // The knapsack is  sum_{binary, a>0} a_j x_j <= rhs - min(activity of every
    // other term). Using the raw rhs is only valid when the other terms can
    // never go negative; a row like x1 + x2 - 2y <= 0 would otherwise yield the
    // invalid cover x1 + x2 <= 1.
    double rhs = row.rhs;
    bool bounded = true;
    for (const auto& kv : row.linear) {
      const Variable* v = find_var(milp, kv.first);
      if (!v) {
        bounded = false;
        break;
      }
      if (is_bin(v->type) && kv.second > 1e-12) {
        auto it = x.find(kv.first);
        const double xv = it == x.end() ? 0.0 : it->second;
        terms.push_back({kv.first, kv.second, xv});
        continue;
      }
      const double at = kv.second > 0.0 ? v->lower_bound : v->upper_bound;
      if (std::abs(at) >= 1e20) {
        bounded = false;
        break;
      }
      rhs -= kv.second * at;
    }
    if (!bounded || terms.size() < 2) continue;

    // Minimal cover heuristic: take vars in decreasing a until sum > rhs
    std::sort(terms.begin(), terms.end(),
              [](const Term& p, const Term& q) { return p.a > q.a; });
    std::vector<Term> cover;
    double sum = 0.0;
    for (const auto& t : terms) {
      cover.push_back(t);
      sum += t.a;
      if (sum > rhs + 1e-9) break;
    }
    if (sum <= rhs + 1e-9 || cover.size() < 2) continue;

    // Check violation: sum x_i > |C|-1
    double lhs = 0.0;
    for (const auto& t : cover) lhs += t.xv;
    const double cut_rhs = static_cast<double>(cover.size()) - 1.0;
    if (lhs <= cut_rhs + int_tol) continue;

    Cut cut;
    std::ostringstream name;
    name << "cover_" << cid++;
    cut.constraint.name = name.str();
    cut.constraint.sense = ConstraintSense::Le;
    cut.constraint.rhs = cut_rhs;
    for (const auto& t : cover) cut.constraint.linear[t.name] = 1.0;
    cut.source = "cover";
    cuts.push_back(cut);
    if (static_cast<int>(cuts.size()) >= max_cuts) break;
  }
  return cuts;
}

std::vector<Cut> generate_mir_cuts(const OptimizationModel& milp,
                                   const std::unordered_map<std::string, double>& x,
                                   double int_tol, int max_cuts) {
  // Chvatal-Gomory rounding of a single <= row.
  //
  // For a row  sum_j a_j x_j <= b  we want to replace coefficients by their
  // floors. The step floor(a_j) x_j <= a_j x_j is only valid for x_j >= 0, so
  // the rounded inequality
  //
  //     sum_{j in S} floor(a_j) x_j  <=  floor(b)
  //
  // is valid ONLY IF every column j NOT in S satisfies floor(a_j) x_j <= a_j x_j
  // across its whole feasible range. Two things can break that:
  //
  //   (a) x_j is allowed to be negative (lower bound < 0) while a_j is
  //       fractional, because then floor(a_j) > a_j and the inequality flips;
  //   (b) x_j can go negative and a_j is negative.
  //
  // The previous version `continue`d past such columns, i.e. it silently
  // dropped them from the cut and then used the cut anyway. That emits an
  // INVALID cut which can remove integer-feasible points, and because cuts are
  // inherited by every descendant node it poisons the whole subtree.
  //
  // Correct behaviour: if any column cannot be safely rounded, abandon the cut
  // for that row. Cheap, and it makes the generator sound.
  std::vector<Cut> cuts;
  int cid = 0;
  for (const auto& row : milp.constraints) {
    if (row.sense != ConstraintSense::Le) continue;

    bool row_is_routable = true;
    bool any_frac = false;
    for (const auto& kv : row.linear) {
      const Variable* v = find_var(milp, kv.first);
      const double a = kv.second;
      const double fa = std::floor(a + 1e-12);
      const bool fractional = std::abs(a - fa) > 1e-9;

      // A negative lower bound is only safe when the coefficient is already
      // integral (then floor(a) == a and the step is an equality for every x).
      const bool can_be_negative = v != nullptr && v->lower_bound < -1e-9;
      if (can_be_negative && (fractional || a < 0.0)) {
        row_is_routable = false;
        break;
      }
      // Coefficients are never rounded unless the variable is integer, so a
      // continuous column with a negative lower bound also has to disqualify
      // the row: we would be dropping a term from the left-hand side.
      if (v != nullptr && !is_int(v->type) && can_be_negative) {
        row_is_routable = false;
        break;
      }
      // Dropping a continuous term is only a relaxation when that term is never
      // negative: from x - y <= 0.5 (y >= 0 continuous) the dropped -y would
      // turn into the invalid cut x <= 0.
      if (v == nullptr || (!is_int(v->type) && a < 0.0)) {
        row_is_routable = false;
        break;
      }
      if (fractional) any_frac = true;
    }
    if (!row_is_routable) continue;

    Constraint c;
    c.sense = ConstraintSense::Le;
    c.rhs = std::floor(row.rhs + 1e-12);
    double viol = -c.rhs;
    for (const auto& kv : row.linear) {
      const Variable* v = find_var(milp, kv.first);
      if (!v || !is_int(v->type)) continue;
      const double fa = std::floor(kv.second + 1e-12);
      if (std::abs(fa) > 1e-12) c.linear[kv.first] = fa;
      auto it = x.find(kv.first);
      if (it != x.end()) viol += fa * it->second;
    }
    if (!any_frac || c.linear.empty()) continue;
    if (viol <= int_tol) continue;

    std::ostringstream name;
    name << "cg_" << cid++;
    c.name = name.str();
    cuts.push_back(Cut{c, "gomory"});
    if (static_cast<int>(cuts.size()) >= max_cuts) break;
  }
  return cuts;
}

std::vector<Cut> generate_cmir_cuts(const OptimizationModel& milp,
                                    const std::unordered_map<std::string, double>& x,
                                    double int_tol, int max_cuts, std::size_t row_count) {
  // Complemented mixed-integer rounding (Marchand and Wolsey) on single rows.
  //
  // A <= row is rewritten over nonnegative variables: each integer x_j becomes
  // x_j - l_j or u_j - x_j (whichever bound is nearer the LP point), each
  // continuous one likewise. For  sum a_j xb_j + sum c_k yb_k <= b  with xb
  // integer >= 0 and yb >= 0, dividing by delta > 0 and applying MIR gives the
  // valid inequality
  //
  //   sum (floor(a_j/d) + max(0, f_j - f0)/(1 - f0)) xb_j
  //     + sum_{c_k < 0} c_k / (d (1 - f0)) yb_k  <=  floor(b/d),
  //
  // with f0 = frac(b/d) and f_j = frac(a_j/d). Continuous terms with c_k > 0
  // are dropped, which only relaxes the row. Upper bounds of xb are not needed
  // for validity. The cut is then mapped back to the original variables.
  constexpr double kInf = 1e20;
  std::unordered_map<std::string, const Variable*> vars;
  vars.reserve(milp.variables.size());
  for (const auto& v : milp.variables) vars[v.name] = &v;
  auto value = [&](const std::string& name) {
    auto it = x.find(name);
    return it == x.end() ? 0.0 : it->second;
  };

  struct Term {
    const std::string* name;
    bool integer;
    bool upper;     // substituted as u - x (else x - l)
    double bound;   // the l or u used
    double coef;    // coefficient on the nonnegative substitute
    double point;   // LP value of the nonnegative substitute
  };
  struct Candidate {
    Constraint cut;
    double efficacy;
  };
  std::vector<Candidate> found;

  auto separate = [&](const Constraint& row, double sign) {
    std::vector<Term> terms;
    terms.reserve(row.linear.size());
    double b = sign * row.rhs;
    bool fractional_integer = false;
    for (const auto& kv : row.linear) {
      const double a = sign * kv.second;
      if (a == 0.0) continue;
      auto it = vars.find(kv.first);
      if (it == vars.end()) return;
      const Variable& v = *it->second;
      const bool integer = is_int(v.type);
      double lo = v.lower_bound;
      double hi = v.upper_bound;
      if (integer) {
        lo = lo > -kInf ? std::ceil(lo - 1e-9) : -kInf;
        hi = hi < kInf ? std::floor(hi + 1e-9) : kInf;
      }
      const double xv = value(kv.first);
      if (integer && std::abs(xv - std::round(xv)) > int_tol) fractional_integer = true;
      const bool has_lo = lo > -kInf;
      const bool has_hi = hi < kInf;
      if (!has_lo && !has_hi) return;
      const bool upper = !has_lo || (has_hi && hi - xv < xv - lo);
      Term t{&kv.first, integer, upper, upper ? hi : lo, upper ? -a : a,
             upper ? hi - xv : xv - lo};
      b -= a * t.bound;
      terms.push_back(t);
    }
    if (!fractional_integer) return;

    // Candidate divisors: coefficients of integer terms strictly inside their
    // range at the LP point, plus 1.
    std::vector<double> deltas{1.0};
    for (const auto& t : terms) {
      if (t.integer && t.point > int_tol && std::abs(t.coef) > 1e-6) deltas.push_back(std::abs(t.coef));
    }
    std::sort(deltas.begin(), deltas.end());
    deltas.erase(std::unique(deltas.begin(), deltas.end(),
                             [](double p, double q) { return std::abs(p - q) <= 1e-9 * std::max(1.0, q); }),
                 deltas.end());
    if (deltas.size() > 8) deltas.resize(8);

    struct Built {
      std::vector<double> coef;
      double rhs = 0.0;
      double efficacy = -1.0;
    };
    auto build = [&](double delta) {
      Built out;
      const double beta = b / delta;
      const double f0 = beta - std::floor(beta);
      if (f0 < 0.01 || f0 > 0.99 || std::abs(beta) > 1e9) return out;
      out.coef.resize(terms.size());
      out.rhs = std::floor(beta);
      double activity = 0.0;
      double norm = 0.0;
      for (std::size_t i = 0; i < terms.size(); ++i) {
        const Term& t = terms[i];
        double g = 0.0;
        if (t.integer) {
          const double q = t.coef / delta;
          const double fj = q - std::floor(q);
          g = std::floor(q) + std::max(0.0, fj - f0) / (1.0 - f0);
        } else if (t.coef < 0.0) {
          g = t.coef / (delta * (1.0 - f0));
        }
        out.coef[i] = g;
        activity += g * t.point;
        norm += g * g;
      }
      if (norm <= 1e-18) return out;
      out.efficacy = (activity - out.rhs) / std::sqrt(norm);
      return out;
    };

    Built best;
    double best_delta = 0.0;
    for (double d : deltas) {
      Built c = build(d);
      if (c.efficacy > best.efficacy) {
        best = std::move(c);
        best_delta = d;
      }
    }
    if (best_delta > 0.0) {
      for (double scale : {0.5, 0.25, 0.125}) {
        Built c = build(best_delta * scale);
        if (c.efficacy > best.efficacy) best = std::move(c);
      }
    }
    if (best.efficacy < 1e-4) return;

    // Back to the original variables.
    Constraint cut;
    cut.sense = ConstraintSense::Le;
    double rhs = best.rhs;
    double largest = 0.0;
    double smallest = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < terms.size(); ++i) {
      const double g = best.coef[i];
      if (g == 0.0) continue;
      const Term& t = terms[i];
      const double c = t.upper ? -g : g;
      rhs += t.upper ? -g * t.bound : g * t.bound;
      cut.linear[*t.name] = c;
      largest = std::max(largest, std::abs(c));
      smallest = std::min(smallest, std::abs(c));
    }
    if (cut.linear.empty() || largest > 1e6 * smallest || std::abs(rhs) > 1e12) return;
    // Absorb rounding error in favour of validity.
    cut.rhs = rhs + 1e-9 * std::max(1.0, std::abs(rhs));
    found.push_back({std::move(cut), best.efficacy});
  };

  const std::size_t rows = std::min(row_count, milp.constraints.size());
  for (std::size_t r = 0; r < rows; ++r) {
    const Constraint& row = milp.constraints[r];
    if (row.sense != ConstraintSense::Ge) separate(row, 1.0);
    if (row.sense != ConstraintSense::Le) separate(row, -1.0);
  }

  std::sort(found.begin(), found.end(),
            [](const Candidate& p, const Candidate& q) { return p.efficacy > q.efficacy; });
  std::vector<Cut> cuts;
  for (auto& c : found) {
    if (static_cast<int>(cuts.size()) >= max_cuts) break;
    c.cut.name = "cmir_" + std::to_string(cuts.size());
    cuts.push_back(Cut{std::move(c.cut), "cmir"});
  }
  return cuts;
}

}  // namespace sovereign
