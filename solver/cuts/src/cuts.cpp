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
    double rhs = row.rhs;
    for (const auto& kv : row.linear) {
      const Variable* v = find_var(milp, kv.first);
      if (!v || !is_bin(v->type) || kv.second <= 1e-12) continue;
      auto it = x.find(kv.first);
      const double xv = it == x.end() ? 0.0 : it->second;
      terms.push_back({kv.first, kv.second, xv});
    }
    if (terms.size() < 2) continue;

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
  // Simple Chvatal-Gomory on a single <= row with nonneg integer vars:
  // floor(a)'x <= floor(b) if a,x >= 0.
  std::vector<Cut> cuts;
  int cid = 0;
  for (const auto& row : milp.constraints) {
    if (row.sense != ConstraintSense::Le) continue;
    Constraint c;
    c.sense = ConstraintSense::Le;
    c.rhs = std::floor(row.rhs + 1e-12);
    bool any_frac = false;
    double viol = -c.rhs;
    for (const auto& kv : row.linear) {
      const Variable* v = find_var(milp, kv.first);
      if (!v || !is_int(v->type)) continue;
      if (v->lower_bound < -1e-9) continue;  // need nonnegative
      const double fa = std::floor(kv.second + 1e-12);
      if (std::abs(kv.second - fa) > 1e-9) any_frac = true;
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
