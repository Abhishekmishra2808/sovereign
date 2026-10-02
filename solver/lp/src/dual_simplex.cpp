#include "sovereign/dual_simplex.hpp"

#include "sovereign/sparse_lu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>

namespace sovereign {
namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();
constexpr double kInfBound = 1e20;
// Temporary box used only to make a cold start dual feasible. A variable still
// sitting on one of these at the end has no certified answer, so the solve
// reports failure instead of trusting it.
constexpr double kArtificialBound = 1e7;
constexpr double kMaxArtificialBound = 1e13;
constexpr int kMaxRechecks = 25;
constexpr double kMinWeight = 1e-4;

double as_bound(double v) {
  if (v >= kInfBound) return kInf;
  if (v <= -kInfBound) return -kInf;
  return v;
}

// Scale factors are rounded to powers of two so scaling introduces no rounding.
double pow2(double s) {
  if (!(s > 0.0) || !std::isfinite(s)) return 1.0;
  int e = 0;
  std::frexp(s, &e);
  return std::ldexp(1.0, e - 1);
}

// Product-form update E^{-1}: column `pivot` of E is the entering column in
// the old basis; only its off-pivot nonzeros are stored.
struct Eta {
  int pivot = 0;
  double pivot_value = 1.0;
  std::vector<int> idx;
  std::vector<double> val;
};

class DualSimplex {
 public:
  DualSimplex(const OptimizationModel& model, const DualSimplexOptions& opt)
      : model_(model), opt_(opt) {}

  SolverResult run(const LpBasis* warm, LpBasis* basis_out) {
    SolverResult result;
    if (!build(result)) return result;
    if (!initial_basis(warm)) {
      result.status = SolverStatus::NumericalError;
      result.message = "Dual simplex: initial basis is singular.";
      return result;
    }
    const int max_iter = opt_.max_iterations > 0 ? opt_.max_iterations
                                                 : 1000 + 20 * (n_ + m_);
    int rechecks = 0;
    int infeasible_retries = 0;
    std::vector<double> rho(static_cast<std::size_t>(m_));
    std::vector<double> alpha_row(static_cast<std::size_t>(n_ + m_));
    std::vector<double> alpha_col(static_cast<std::size_t>(m_));

    while (true) {
      if (iterations_ >= max_iter) {
        result.status = SolverStatus::IterationLimit;
        result.message = "Dual simplex hit its iteration limit.";
        result.iterations = iterations_;
        return result;
      }

      const int r = choose_leaving_row();
      if (r < 0) {
        // Candidate optimum. Re-derive everything from a fresh factor so the
        // verdict is not an artefact of eta-chain drift.
        if (!refresh()) return numerical(result, "refactorization failed");
        if (fix_dual_infeasibilities() > 0) compute_primal();
        if (choose_leaving_row() >= 0 || max_dual_infeasibility() > opt_.dual_tol) {
          if (++rechecks > kMaxRechecks) {
            return numerical(result, "could not reach a basis that is both primal and dual feasible");
          }
          continue;
        }
        // Moving a nonbasic onto a wider temporary bound keeps every reduced
        // cost, so dual feasibility survives and the dual simplex just resumes.
        if (opt_.widen_bounds && widen_active_artificial_bounds()) {
          compute_primal();
          continue;
        }
        return finish(result, basis_out);
      }

      const int p = head_[static_cast<std::size_t>(r)];
      const double xp = x_[static_cast<std::size_t>(p)];
      const bool below = xp < lower_[static_cast<std::size_t>(p)];
      const double target = below ? lower_[static_cast<std::size_t>(p)] : upper_[static_cast<std::size_t>(p)];
      const double delta = xp - target;

      std::fill(rho.begin(), rho.end(), 0.0);
      rho[static_cast<std::size_t>(r)] = 1.0;
      if (!btran(rho)) return numerical(result, "btran failed");
      row_of_nonbasics(rho, alpha_row);

      const int q = dual_ratio_test(alpha_row, below);
      if (q < 0) {
        if (certify_infeasible(rho)) {
          result.status = SolverStatus::Infeasible;
          result.iterations = iterations_;
          result.message = "LP infeasible (dual simplex: certified by an implied row with no feasible activity).";
          return result;
        }
        if (++infeasible_retries > 2 || !refresh()) {
          return numerical(result, "dual ray found but infeasibility could not be certified");
        }
        continue;
      }

      column(q, alpha_col);
      if (!ftran(alpha_col)) return numerical(result, "ftran failed");
      const double pivot = alpha_col[static_cast<std::size_t>(r)];
      const double row_pivot = alpha_row[static_cast<std::size_t>(q)];
      if (std::abs(pivot - row_pivot) > 1e-7 * (1.0 + std::abs(pivot)) || std::abs(pivot) < opt_.pivot_tol) {
        if (since_refactor_ > 0) {
          if (!refresh()) return numerical(result, "refactorization failed");
          continue;
        }
        if (std::abs(pivot) < opt_.pivot_tol) return numerical(result, "pivot element vanished");
      }

      if (opt_.steepest_edge) {
        // Row i of the new B^{-1} is rho_i - (alpha_i/alpha_r) rho_r, and
        // rho_i . rho_r = (B^{-1} rho_r)_i, so one extra ftran keeps every
        // weight ||rho_i||^2 exact up to rounding.
        double w_r = 0.0;
        for (const double v : rho) w_r += v * v;
        tau_ = rho;
        if (!ftran(tau_)) return numerical(result, "ftran failed");
        for (int i = 0; i < m_; ++i) {
          if (i == r) continue;
          const double ratio = alpha_col[static_cast<std::size_t>(i)] / pivot;
          if (ratio == 0.0) continue;
          double& w = weight_[static_cast<std::size_t>(i)];
          w = std::max(w + ratio * (ratio * w_r - 2.0 * tau_[static_cast<std::size_t>(i)]), kMinWeight);
        }
        weight_[static_cast<std::size_t>(r)] = std::max(w_r / (pivot * pivot), kMinWeight);
      }

      const double theta_d = d_[static_cast<std::size_t>(q)] / row_pivot;
      for (int k = 0; k < n_ + m_; ++k) {
        if (status_[static_cast<std::size_t>(k)] == BasisStatus::Basic) continue;
        d_[static_cast<std::size_t>(k)] -= theta_d * alpha_row[static_cast<std::size_t>(k)];
      }
      d_[static_cast<std::size_t>(q)] = 0.0;
      d_[static_cast<std::size_t>(p)] = -theta_d;

      const double theta_p = delta / pivot;
      for (int i = 0; i < m_; ++i) {
        x_[static_cast<std::size_t>(head_[static_cast<std::size_t>(i)])] -=
            theta_p * alpha_col[static_cast<std::size_t>(i)];
      }
      x_[static_cast<std::size_t>(q)] += theta_p;
      x_[static_cast<std::size_t>(p)] = target;

      status_[static_cast<std::size_t>(p)] = below ? BasisStatus::AtLower : BasisStatus::AtUpper;
      status_[static_cast<std::size_t>(q)] = BasisStatus::Basic;
      head_[static_cast<std::size_t>(r)] = q;
      Eta eta;
      eta.pivot = r;
      eta.pivot_value = pivot;
      for (int i = 0; i < m_; ++i) {
        const double a = alpha_col[static_cast<std::size_t>(i)];
        if (i != r && a != 0.0) {
          eta.idx.push_back(i);
          eta.val.push_back(a);
        }
      }
      etas_.push_back(std::move(eta));
      ++since_refactor_;
      ++iterations_;

      if (since_refactor_ >= opt_.refactor_every) {
        if (!refresh()) return numerical(result, "refactorization failed");
        if (fix_dual_infeasibilities() > 0) compute_primal();
      }
    }
  }

 private:
  // ---- problem construction --------------------------------------------
  bool build(SolverResult& result) {
    n_ = static_cast<int>(model_.variables.size());
    m_ = static_cast<int>(model_.constraints.size());
    sign_ = model_.sense == Sense::Maximize ? -1.0 : 1.0;

    std::unordered_map<std::string, int> index;
    index.reserve(static_cast<std::size_t>(n_) * 2);
    for (int j = 0; j < n_; ++j) index.emplace(model_.variables[static_cast<std::size_t>(j)].name, j);

    std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n_));
    for (int i = 0; i < m_; ++i) {
      for (const auto& kv : model_.constraints[static_cast<std::size_t>(i)].linear) {
        if (kv.second == 0.0) continue;
        auto it = index.find(kv.first);
        if (it == index.end()) {
          result.status = SolverStatus::Error;
          result.message = "Dual simplex: constraint references unknown variable " + kv.first;
          return false;
        }
        cols[static_cast<std::size_t>(it->second)].emplace_back(i, kv.second);
      }
    }

    row_scale_.assign(static_cast<std::size_t>(m_), 1.0);
    col_scale_.assign(static_cast<std::size_t>(n_), 1.0);
    if (opt_.scaling) {
      std::vector<double> rmax(static_cast<std::size_t>(m_), 0.0), rmin(static_cast<std::size_t>(m_), kInf);
      for (const auto& col : cols) {
        for (const auto& e : col) {
          const double v = std::abs(e.second);
          rmax[static_cast<std::size_t>(e.first)] = std::max(rmax[static_cast<std::size_t>(e.first)], v);
          rmin[static_cast<std::size_t>(e.first)] = std::min(rmin[static_cast<std::size_t>(e.first)], v);
        }
      }
      for (int i = 0; i < m_; ++i) {
        if (rmax[static_cast<std::size_t>(i)] > 0.0) {
          row_scale_[static_cast<std::size_t>(i)] =
              pow2(1.0 / std::sqrt(rmax[static_cast<std::size_t>(i)] * rmin[static_cast<std::size_t>(i)]));
        }
      }
      for (int j = 0; j < n_; ++j) {
        double cmax = 0.0, cmin = kInf;
        for (const auto& e : cols[static_cast<std::size_t>(j)]) {
          const double v = std::abs(e.second) * row_scale_[static_cast<std::size_t>(e.first)];
          cmax = std::max(cmax, v);
          cmin = std::min(cmin, v);
        }
        if (cmax > 0.0) col_scale_[static_cast<std::size_t>(j)] = pow2(1.0 / std::sqrt(cmax * cmin));
      }
    }

    col_ptr_.assign(static_cast<std::size_t>(n_) + 1, 0);
    row_idx_.clear();
    val_.clear();
    for (int j = 0; j < n_; ++j) {
      for (const auto& e : cols[static_cast<std::size_t>(j)]) {
        row_idx_.push_back(e.first);
        val_.push_back(e.second * row_scale_[static_cast<std::size_t>(e.first)] *
                       col_scale_[static_cast<std::size_t>(j)]);
      }
      col_ptr_[static_cast<std::size_t>(j) + 1] = static_cast<int>(row_idx_.size());
    }

    const int total = n_ + m_;
    lower_.assign(static_cast<std::size_t>(total), 0.0);
    upper_.assign(static_cast<std::size_t>(total), 0.0);
    cost_.assign(static_cast<std::size_t>(total), 0.0);
    for (int j = 0; j < n_; ++j) {
      const Variable& v = model_.variables[static_cast<std::size_t>(j)];
      const double s = col_scale_[static_cast<std::size_t>(j)];
      const double lo = as_bound(v.lower_bound), hi = as_bound(v.upper_bound);
      lower_[static_cast<std::size_t>(j)] = std::isfinite(lo) ? lo / s : lo;
      upper_[static_cast<std::size_t>(j)] = std::isfinite(hi) ? hi / s : hi;
      if (lower_[static_cast<std::size_t>(j)] > upper_[static_cast<std::size_t>(j)] + opt_.primal_tol) {
        result.status = SolverStatus::Infeasible;
        result.message = "LP infeasible: variable " + v.name + " has lower bound above upper bound.";
        return false;
      }
    }
    for (const auto& kv : model_.objective.linear) {
      auto it = index.find(kv.first);
      if (it == index.end()) continue;
      cost_[static_cast<std::size_t>(it->second)] =
          sign_ * kv.second * col_scale_[static_cast<std::size_t>(it->second)];
    }
    for (int i = 0; i < m_; ++i) {
      const Constraint& c = model_.constraints[static_cast<std::size_t>(i)];
      const double rs = row_scale_[static_cast<std::size_t>(i)];
      const double rhs = c.rhs * rs;
      const std::size_t k = static_cast<std::size_t>(n_ + i);
      lower_[k] = c.sense == ConstraintSense::Le ? -kInf : rhs;
      upper_[k] = c.sense == ConstraintSense::Ge ? kInf : rhs;
    }
    orig_lower_ = lower_;
    orig_upper_ = upper_;
    artificial_.assign(static_cast<std::size_t>(total), 0);
    return true;
  }

  // ---- linear algebra ---------------------------------------------------
  void column(int k, std::vector<double>& out) const {
    std::fill(out.begin(), out.end(), 0.0);
    if (k < n_) {
      for (int p = col_ptr_[static_cast<std::size_t>(k)]; p < col_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
        out[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(p)])] = val_[static_cast<std::size_t>(p)];
      }
    } else {
      out[static_cast<std::size_t>(k - n_)] = -1.0;
    }
  }

  bool refactor() {
    basis_ptr_.assign(1, 0);
    basis_idx_.clear();
    basis_val_.clear();
    for (int i = 0; i < m_; ++i) {
      const int k = head_[static_cast<std::size_t>(i)];
      if (k < n_) {
        for (int p = col_ptr_[static_cast<std::size_t>(k)]; p < col_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
          basis_idx_.push_back(row_idx_[static_cast<std::size_t>(p)]);
          basis_val_.push_back(val_[static_cast<std::size_t>(p)]);
        }
      } else {
        basis_idx_.push_back(k - n_);
        basis_val_.push_back(-1.0);
      }
      basis_ptr_.push_back(static_cast<int>(basis_idx_.size()));
    }
    etas_.clear();
    since_refactor_ = 0;
    return lu_.factorize(static_cast<std::size_t>(m_), basis_ptr_, basis_idx_, basis_val_);
  }

  bool ftran(std::vector<double>& v) const {
    if (!lu_.solve(v)) return false;
    for (const Eta& e : etas_) {
      const double vp = v[static_cast<std::size_t>(e.pivot)] / e.pivot_value;
      if (vp != 0.0) {
        for (std::size_t k = 0; k < e.idx.size(); ++k) {
          v[static_cast<std::size_t>(e.idx[k])] -= e.val[k] * vp;
        }
      }
      v[static_cast<std::size_t>(e.pivot)] = vp;
    }
    return true;
  }

  bool btran(std::vector<double>& v) const {
    for (auto it = etas_.rbegin(); it != etas_.rend(); ++it) {
      double sum = v[static_cast<std::size_t>(it->pivot)];
      for (std::size_t k = 0; k < it->idx.size(); ++k) {
        sum -= it->val[k] * v[static_cast<std::size_t>(it->idx[k])];
      }
      v[static_cast<std::size_t>(it->pivot)] = sum / it->pivot_value;
    }
    return lu_.solve_transpose(v);
  }

  // ---- basis state ------------------------------------------------------
  double nonbasic_value(int k) const {
    switch (status_[static_cast<std::size_t>(k)]) {
      case BasisStatus::AtLower: return lower_[static_cast<std::size_t>(k)];
      case BasisStatus::AtUpper: return upper_[static_cast<std::size_t>(k)];
      default: return 0.0;
    }
  }

  // Make a nonbasic status consistent with the bounds it refers to.
  void normalize_status(int k) {
    BasisStatus& s = status_[static_cast<std::size_t>(k)];
    if (s == BasisStatus::Basic) return;
    const bool lo = std::isfinite(lower_[static_cast<std::size_t>(k)]);
    const bool hi = std::isfinite(upper_[static_cast<std::size_t>(k)]);
    if (s == BasisStatus::AtLower && !lo) s = hi ? BasisStatus::AtUpper : BasisStatus::Free;
    else if (s == BasisStatus::AtUpper && !hi) s = lo ? BasisStatus::AtLower : BasisStatus::Free;
    else if (s == BasisStatus::Free && (lo || hi)) s = lo ? BasisStatus::AtLower : BasisStatus::AtUpper;
  }

  bool initial_basis(const LpBasis* warm) {
    const int total = n_ + m_;
    status_.assign(static_cast<std::size_t>(total), BasisStatus::AtLower);
    head_.clear();
    bool use_warm = warm != nullptr && static_cast<int>(warm->cols.size()) == n_ &&
                    static_cast<int>(warm->rows.size()) <= m_;
    if (use_warm) {
      for (int j = 0; j < n_; ++j) status_[static_cast<std::size_t>(j)] = warm->cols[static_cast<std::size_t>(j)];
      for (int i = 0; i < m_; ++i) {
        status_[static_cast<std::size_t>(n_ + i)] =
            i < static_cast<int>(warm->rows.size()) ? warm->rows[static_cast<std::size_t>(i)] : BasisStatus::Basic;
      }
      for (int k = 0; k < total; ++k) {
        if (status_[static_cast<std::size_t>(k)] == BasisStatus::Basic) head_.push_back(k);
      }
      if (static_cast<int>(head_.size()) != m_) use_warm = false;
    }
    if (use_warm) {
      for (int k = 0; k < total; ++k) normalize_status(k);
      if (!refactor()) use_warm = false;
    }
    if (!use_warm) {
      head_.clear();
      for (int j = 0; j < n_; ++j) {
        status_[static_cast<std::size_t>(j)] =
            cost_[static_cast<std::size_t>(j)] >= 0.0 ? BasisStatus::AtLower : BasisStatus::AtUpper;
        normalize_status(j);
      }
      for (int i = 0; i < m_; ++i) {
        status_[static_cast<std::size_t>(n_ + i)] = BasisStatus::Basic;
        head_.push_back(n_ + i);
      }
      if (!refactor()) return false;
    }
    x_.assign(static_cast<std::size_t>(total), 0.0);
    d_.assign(static_cast<std::size_t>(total), 0.0);
    weight_.assign(static_cast<std::size_t>(m_), 1.0);
    compute_duals();
    fix_dual_infeasibilities();
    compute_primal();
    return true;
  }

  void compute_primal() {
    std::vector<double> rhs(static_cast<std::size_t>(m_), 0.0);
    for (int k = 0; k < n_ + m_; ++k) {
      if (status_[static_cast<std::size_t>(k)] == BasisStatus::Basic) continue;
      const double v = nonbasic_value(k);
      x_[static_cast<std::size_t>(k)] = v;
      if (v == 0.0) continue;
      if (k < n_) {
        for (int p = col_ptr_[static_cast<std::size_t>(k)]; p < col_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
          rhs[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(p)])] -= val_[static_cast<std::size_t>(p)] * v;
        }
      } else {
        rhs[static_cast<std::size_t>(k - n_)] += v;
      }
    }
    ftran(rhs);
    for (int i = 0; i < m_; ++i) {
      x_[static_cast<std::size_t>(head_[static_cast<std::size_t>(i)])] = rhs[static_cast<std::size_t>(i)];
    }
  }

  void compute_duals() {
    std::vector<double> y(static_cast<std::size_t>(m_));
    for (int i = 0; i < m_; ++i) {
      y[static_cast<std::size_t>(i)] = cost_[static_cast<std::size_t>(head_[static_cast<std::size_t>(i)])];
    }
    btran(y);
    for (int k = 0; k < n_ + m_; ++k) {
      if (status_[static_cast<std::size_t>(k)] == BasisStatus::Basic) {
        d_[static_cast<std::size_t>(k)] = 0.0;
      } else if (k < n_) {
        double dot = 0.0;
        for (int p = col_ptr_[static_cast<std::size_t>(k)]; p < col_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
          dot += val_[static_cast<std::size_t>(p)] * y[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(p)])];
        }
        d_[static_cast<std::size_t>(k)] = cost_[static_cast<std::size_t>(k)] - dot;
      } else {
        d_[static_cast<std::size_t>(k)] = y[static_cast<std::size_t>(k - n_)];
      }
    }
  }

  bool refresh() {
    if (!refactor()) return false;
    compute_primal();
    compute_duals();
    return true;
  }

  bool is_fixed(int k) const {
    return lower_[static_cast<std::size_t>(k)] == upper_[static_cast<std::size_t>(k)];
  }

  double dual_infeasibility(int k) const {
    if (status_[static_cast<std::size_t>(k)] == BasisStatus::Basic || is_fixed(k)) return 0.0;
    const double dk = d_[static_cast<std::size_t>(k)];
    switch (status_[static_cast<std::size_t>(k)]) {
      case BasisStatus::AtLower: return std::max(0.0, -dk);
      case BasisStatus::AtUpper: return std::max(0.0, dk);
      default: return std::abs(dk);
    }
  }

  double max_dual_infeasibility() const {
    double worst = 0.0;
    for (int k = 0; k < n_ + m_; ++k) worst = std::max(worst, dual_infeasibility(k));
    return worst;
  }

  // Restore dual feasibility by moving nonbasics to the bound their reduced
  // cost asks for; a missing bound is replaced by a temporary artificial one.
  int fix_dual_infeasibilities() {
    int changed = 0;
    for (int k = 0; k < n_ + m_; ++k) {
      if (dual_infeasibility(k) <= opt_.dual_tol) continue;
      const std::size_t kk = static_cast<std::size_t>(k);
      const bool want_upper = d_[kk] < 0.0;
      if (want_upper) {
        if (!std::isfinite(upper_[kk])) {
          upper_[kk] = (std::isfinite(lower_[kk]) ? lower_[kk] : 0.0) + kArtificialBound;
          artificial_[kk] = 1;
        }
        status_[kk] = BasisStatus::AtUpper;
      } else {
        if (!std::isfinite(lower_[kk])) {
          lower_[kk] = (std::isfinite(upper_[kk]) ? upper_[kk] : 0.0) - kArtificialBound;
          artificial_[kk] = 1;
        }
        status_[kk] = BasisStatus::AtLower;
      }
      ++changed;
    }
    return changed;
  }

  // Push every temporary bound that a nonbasic is sitting on 100x further out.
  // Returns false once nothing is active or the box has grown past the point
  // where an answer on it would mean the LP is unbounded in practice.
  bool widen_active_artificial_bounds() {
    bool widened = false;
    for (int k = 0; k < n_ + m_; ++k) {
      const std::size_t kk = static_cast<std::size_t>(k);
      if (!artificial_[kk]) continue;
      if (status_[kk] == BasisStatus::AtUpper && upper_[kk] != orig_upper_[kk]) {
        const double base = std::isfinite(lower_[kk]) && lower_[kk] == orig_lower_[kk] ? lower_[kk] : 0.0;
        const double width = std::max(upper_[kk] - base, kArtificialBound) * 100.0;
        if (width > kMaxArtificialBound) return false;
        upper_[kk] = base + width;
        widened = true;
      } else if (status_[kk] == BasisStatus::AtLower && lower_[kk] != orig_lower_[kk]) {
        const double base = std::isfinite(upper_[kk]) && upper_[kk] == orig_upper_[kk] ? upper_[kk] : 0.0;
        const double width = std::max(base - lower_[kk], kArtificialBound) * 100.0;
        if (width > kMaxArtificialBound) return false;
        lower_[kk] = base - width;
        widened = true;
      }
    }
    return widened;
  }

  // ---- pricing and ratio test ------------------------------------------
  int choose_leaving_row() const {
    int best = -1;
    double best_infeas = 0.0;
    for (int i = 0; i < m_; ++i) {
      const int k = head_[static_cast<std::size_t>(i)];
      const double xk = x_[static_cast<std::size_t>(k)];
      double infeas = 0.0;
      if (xk < lower_[static_cast<std::size_t>(k)] - opt_.primal_tol) infeas = lower_[static_cast<std::size_t>(k)] - xk;
      else if (xk > upper_[static_cast<std::size_t>(k)] + opt_.primal_tol) infeas = xk - upper_[static_cast<std::size_t>(k)];
      if (infeas > 0.0 && opt_.steepest_edge) infeas = infeas * infeas / weight_[static_cast<std::size_t>(i)];
      if (infeas > best_infeas) {
        best_infeas = infeas;
        best = i;
      }
    }
    return best;
  }

  void row_of_nonbasics(const std::vector<double>& rho, std::vector<double>& alpha_row) const {
    for (int k = 0; k < n_ + m_; ++k) {
      if (status_[static_cast<std::size_t>(k)] == BasisStatus::Basic) {
        alpha_row[static_cast<std::size_t>(k)] = 0.0;
      } else if (k < n_) {
        double dot = 0.0;
        for (int p = col_ptr_[static_cast<std::size_t>(k)]; p < col_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
          dot += val_[static_cast<std::size_t>(p)] * rho[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(p)])];
        }
        alpha_row[static_cast<std::size_t>(k)] = dot;
      } else {
        alpha_row[static_cast<std::size_t>(k)] = -rho[static_cast<std::size_t>(k - n_)];
      }
    }
  }

  // Two-pass Harris ratio test on the reduced costs. `below` says the leaving
  // variable is under its lower bound (so it must increase).
  int dual_ratio_test(const std::vector<double>& alpha_row, bool below) const {
    struct Cand { int k; double slack; double step; };
    std::vector<Cand> cands;
    double theta_max = kInf;
    for (int k = 0; k < n_ + m_; ++k) {
      const BasisStatus s = status_[static_cast<std::size_t>(k)];
      if (s == BasisStatus::Basic || is_fixed(k)) continue;
      const double a = below ? -alpha_row[static_cast<std::size_t>(k)] : alpha_row[static_cast<std::size_t>(k)];
      const double dk = d_[static_cast<std::size_t>(k)];
      double slack = 0.0, step = 0.0;
      if (s == BasisStatus::AtLower) {
        if (a <= opt_.pivot_tol) continue;
        slack = dk;
        step = a;
      } else if (s == BasisStatus::AtUpper) {
        if (a >= -opt_.pivot_tol) continue;
        slack = -dk;
        step = -a;
      } else {
        if (std::abs(a) <= opt_.pivot_tol) continue;
        slack = std::abs(dk);
        step = std::abs(a);
      }
      cands.push_back({k, slack, step});
      theta_max = std::min(theta_max, (std::max(slack, 0.0) + opt_.dual_tol) / step);
    }
    int best = -1;
    double best_step = 0.0;
    if (!opt_.harris) {
      double best_ratio = kInf;
      for (const Cand& c : cands) {
        const double ratio = std::max(c.slack, 0.0) / c.step;
        if (ratio < best_ratio) {
          best_ratio = ratio;
          best = c.k;
        }
      }
      return best;
    }
    for (const Cand& c : cands) {
      if (c.slack / c.step <= theta_max && c.step > best_step) {
        best_step = c.step;
        best = c.k;
      }
    }
    return best;
  }

  // rho' [A -I] z = 0 holds for EVERY rho, so if the implied row cannot reach
  // zero anywhere inside the original bounds (with a primal-tolerance margin on
  // every variable), the LP is infeasible regardless of factorization accuracy.
  bool certify_infeasible(const std::vector<double>& rho) const {
    const int total = n_ + m_;
    std::vector<double> g(static_cast<std::size_t>(total), 0.0);
    double gmax = 0.0;
    for (int k = 0; k < total; ++k) {
      double gk = 0.0;
      if (k < n_) {
        for (int p = col_ptr_[static_cast<std::size_t>(k)]; p < col_ptr_[static_cast<std::size_t>(k) + 1]; ++p) {
          gk += val_[static_cast<std::size_t>(p)] * rho[static_cast<std::size_t>(row_idx_[static_cast<std::size_t>(p)])];
        }
      } else {
        gk = -rho[static_cast<std::size_t>(k - n_)];
      }
      g[static_cast<std::size_t>(k)] = gk;
      gmax = std::max(gmax, std::abs(gk));
    }
    if (gmax == 0.0) return false;
    double lo = 0.0, hi = 0.0, margin = 0.0, magnitude = 0.0;
    bool lo_finite = true, hi_finite = true;
    for (int k = 0; k < total; ++k) {
      const double gk = g[static_cast<std::size_t>(k)];
      if (std::abs(gk) <= 1e-12 * gmax) continue;
      const double l = orig_lower_[static_cast<std::size_t>(k)], u = orig_upper_[static_cast<std::size_t>(k)];
      const double at_l = gk * l, at_u = gk * u;
      const double mn = gk > 0 ? at_l : at_u, mx = gk > 0 ? at_u : at_l;
      if (std::isfinite(mn)) lo += mn; else lo_finite = false;
      if (std::isfinite(mx)) hi += mx; else hi_finite = false;
      margin += std::abs(gk) * opt_.primal_tol;
      if (std::isfinite(l)) magnitude = std::max(magnitude, std::abs(at_l));
      if (std::isfinite(u)) magnitude = std::max(magnitude, std::abs(at_u));
    }
    margin += 1e-9 * (1.0 + magnitude);
    return (lo_finite && lo > margin) || (hi_finite && hi < -margin);
  }

  // ---- results ----------------------------------------------------------
  SolverResult& numerical(SolverResult& result, const std::string& why) {
    result.status = SolverStatus::NumericalError;
    result.iterations = iterations_;
    result.message = "Dual simplex could not certify a result: " + why + ".";
    return result;
  }

  SolverResult& finish(SolverResult& result, LpBasis* basis_out) {
    for (int k = 0; k < n_ + m_; ++k) {
      if (!artificial_[static_cast<std::size_t>(k)]) continue;
      const BasisStatus s = status_[static_cast<std::size_t>(k)];
      if ((s == BasisStatus::AtLower && lower_[static_cast<std::size_t>(k)] != orig_lower_[static_cast<std::size_t>(k)]) ||
          (s == BasisStatus::AtUpper && upper_[static_cast<std::size_t>(k)] != orig_upper_[static_cast<std::size_t>(k)])) {
        return numerical(result, "a temporary bound is active, so the LP may be unbounded");
      }
    }

    // Certify in the ORIGINAL units, independently of the scaled solve.
    std::vector<double> xs(static_cast<std::size_t>(n_));
    double bound_violation = 0.0;
    for (int j = 0; j < n_; ++j) {
      const Variable& v = model_.variables[static_cast<std::size_t>(j)];
      const double xj = x_[static_cast<std::size_t>(j)] * col_scale_[static_cast<std::size_t>(j)];
      xs[static_cast<std::size_t>(j)] = xj;
      const double lo = as_bound(v.lower_bound), hi = as_bound(v.upper_bound);
      if (std::isfinite(lo)) bound_violation = std::max(bound_violation, (lo - xj) / (1.0 + std::abs(lo)));
      if (std::isfinite(hi)) bound_violation = std::max(bound_violation, (xj - hi) / (1.0 + std::abs(hi)));
    }
    std::vector<double> activity(static_cast<std::size_t>(m_), 0.0);
    for (int j = 0; j < n_; ++j) {
      for (int p = col_ptr_[static_cast<std::size_t>(j)]; p < col_ptr_[static_cast<std::size_t>(j) + 1]; ++p) {
        const int i = row_idx_[static_cast<std::size_t>(p)];
        const double a = val_[static_cast<std::size_t>(p)] /
                         (row_scale_[static_cast<std::size_t>(i)] * col_scale_[static_cast<std::size_t>(j)]);
        activity[static_cast<std::size_t>(i)] += a * xs[static_cast<std::size_t>(j)];
      }
    }
    double row_violation = 0.0;
    for (int i = 0; i < m_; ++i) {
      const Constraint& c = model_.constraints[static_cast<std::size_t>(i)];
      const double ai = activity[static_cast<std::size_t>(i)];
      double viol = 0.0;
      if (c.sense != ConstraintSense::Ge) viol = std::max(viol, ai - c.rhs);
      if (c.sense != ConstraintSense::Le) viol = std::max(viol, c.rhs - ai);
      row_violation = std::max(row_violation, viol / (1.0 + std::abs(c.rhs)));
    }
    const double primal_violation = std::max(bound_violation, row_violation);
    constexpr double kCertifyTol = 1e-6;
    if (primal_violation > kCertifyTol) {
      std::ostringstream oss;
      oss << "final point violates the original constraints by " << primal_violation;
      return numerical(result, oss.str());
    }

    double obj = model_.objective.constant;
    for (int j = 0; j < n_; ++j) {
      auto it = model_.objective.linear.find(model_.variables[static_cast<std::size_t>(j)].name);
      if (it != model_.objective.linear.end()) obj += it->second * xs[static_cast<std::size_t>(j)];
    }
    for (int j = 0; j < n_; ++j) {
      result.primal[model_.variables[static_cast<std::size_t>(j)].name] = xs[static_cast<std::size_t>(j)];
    }
    result.status = SolverStatus::Optimal;
    result.has_objective_value = true;
    result.objective_value = obj;
    result.iterations = iterations_;
    result.primal_residual = primal_violation;
    result.dual_residual = max_dual_infeasibility();
    result.message = "Optimal solution found by bounded dual simplex.";

    if (basis_out != nullptr) {
      basis_out->cols.assign(status_.begin(), status_.begin() + n_);
      basis_out->rows.assign(status_.begin() + n_, status_.end());
    }
    return result;
  }

  const OptimizationModel& model_;
  DualSimplexOptions opt_;
  int n_ = 0;
  int m_ = 0;
  double sign_ = 1.0;

  std::vector<int> col_ptr_, row_idx_;
  std::vector<double> val_;
  std::vector<double> row_scale_, col_scale_;
  std::vector<double> lower_, upper_, cost_, orig_lower_, orig_upper_;
  std::vector<char> artificial_;

  std::vector<BasisStatus> status_;
  std::vector<int> head_;
  std::vector<double> x_, d_;
  std::vector<double> weight_;  // dual steepest-edge weights, per basis row
  std::vector<double> tau_;
  SparseLU lu_;
  std::vector<int> basis_ptr_, basis_idx_;
  std::vector<double> basis_val_;
  std::vector<Eta> etas_;
  int since_refactor_ = 0;
  int iterations_ = 0;
};

}  // namespace

SolverResult solve_lp_dual_simplex(const OptimizationModel& lp, const LpBasis* warm,
                                   LpBasis* basis_out, const DualSimplexOptions& options) {
  if (!lp.objective.quadratic.empty()) {
    SolverResult r;
    r.status = SolverStatus::Error;
    r.message = "Dual simplex expects a linear objective.";
    return r;
  }
  DualSimplexOptions opt = options;
  if (const char* pricing = std::getenv("SOVEREIGN_DUAL_PRICING")) {
    if (std::strcmp(pricing, "dantzig") == 0) opt.steepest_edge = false;
  }
  if (const char* off = std::getenv("SOVEREIGN_DUAL_DISABLE")) {
    const std::string list = std::string(",") + off + ",";
    if (list.find(",scaling,") != std::string::npos) opt.scaling = false;
    if (list.find(",harris,") != std::string::npos) opt.harris = false;
    if (list.find(",widening,") != std::string::npos) opt.widen_bounds = false;
  }
  return DualSimplex(lp, opt).run(warm, basis_out);
}

}  // namespace sovereign
