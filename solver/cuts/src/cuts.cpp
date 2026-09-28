#include "sovereign/cuts.hpp"

#include <algorithm>
#include <cmath>
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

  // 2. The cut must be satisfiable at all by an integer-feasible point. If the
  //    cut's own coefficients imply a variable outside its declared bounds, it
  //    is malformed regardless of the generators' intent.
  for (const auto& kv : cut.linear) {
    const Variable* v = find_var(milp, kv.first);
    if (v == nullptr) {
      return "cut '" + cut.name + "' references unknown variable " + kv.first;
    }
    if (v->type == VariableType::Binary &&
        (kv.second < -1e-12 || kv.second > 1.0 + 1e-12)) {
      std::ostringstream oss;
      oss << "cut '" << cut.name << "' puts coefficient " << kv.second
          << " on binary variable " << kv.first;
      return oss.str();
    }
  }

  // 3. A <= cut with a non-positive right-hand side and at least one strictly
  //    positive coefficient forces every such variable to 0. If one of them is
  //    fixed at its lower bound above 0, the cut is infeasible by construction.
  if (cut.sense == ConstraintSense::Le) {
    double forced_lb = 0.0;
    for (const auto& kv : cut.linear) {
      const Variable* v = find_var(milp, kv.first);
      if (v != nullptr && kv.second > 0.0) {
        forced_lb += kv.second * v->lower_bound;
      }
    }
    if (forced_lb > cut.rhs + tol) {
      std::ostringstream oss;
      oss << "cut '" << cut.name << "' is infeasible: its own coefficients force a lower "
          << "bound of " << forced_lb << " but the rhs is " << cut.rhs;
      return oss.str();
    }
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

}  // namespace sovereign
