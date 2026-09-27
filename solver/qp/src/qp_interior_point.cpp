#include "sovereign/qp_interior_point.hpp"

#include "sovereign/dense_lu.hpp"
#include "sovereign/sparse_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sovereign {
namespace {

struct IpmQp {
  SparseMatrixCSC A;             // m x n equalities (with slacks)
  std::vector<double> b;         // m
  std::vector<double> c;         // n (minimization linear term after shift)
  std::vector<double> Q;         // n x n dense column-major (slack block = 0)
  std::vector<std::string> names;
  std::vector<double> shift;
  std::vector<double> col_scale;
  int n_structural = 0;
  int m = 0;
  int n = 0;
  Sense original_sense = Sense::Minimize;
};

double max_abs(const std::vector<double>& v) {
  double m = 0.0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

std::vector<double> matvec_At(const SparseMatrixCSC& A, const std::vector<double>& y) {
  std::vector<double> out;
  A.multiply_transpose(y, out);
  return out;
}

void matvec_Q(const std::vector<double>& Q, int n, const std::vector<double>& x,
              std::vector<double>& out) {
  out.assign(static_cast<std::size_t>(n), 0.0);
  for (int j = 0; j < n; ++j) {
    const double xj = x[static_cast<std::size_t>(j)];
    if (xj == 0.0) continue;
    for (int i = 0; i < n; ++i) {
      out[static_cast<std::size_t>(i)] +=
          Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
            static_cast<std::size_t>(i)] *
          xj;
    }
  }
}

int var_index(const OptimizationModel& m, const std::string& name) {
  for (std::size_t i = 0; i < m.variables.size(); ++i) {
    if (m.variables[i].name == name) return static_cast<int>(i);
  }
  return -1;
}

void build_Q_structural(const OptimizationModel& model, std::vector<double>& Qs,
                        int n_struct, double sense_sign) {
  Qs.assign(static_cast<std::size_t>(n_struct) * static_cast<std::size_t>(n_struct), 0.0);
  for (const auto& row : model.objective.quadratic) {
    const int i = var_index(model, row.first);
    if (i < 0) continue;
    for (const auto& col : row.second) {
      const int j = var_index(model, col.first);
      if (j < 0) continue;
      // Model stores terms for 1/2 x'Qx with Q_ij contributions as given.
      Qs[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_struct) +
         static_cast<std::size_t>(i)] += sense_sign * col.second;
      if (i != j) {
        Qs[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_struct) +
           static_cast<std::size_t>(j)] += sense_sign * col.second;
      }
    }
  }
  // Symmetrize
  for (int i = 0; i < n_struct; ++i) {
    for (int j = i + 1; j < n_struct; ++j) {
      const double mid =
          0.5 *
          (Qs[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_struct) +
              static_cast<std::size_t>(i)] +
           Qs[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_struct) +
              static_cast<std::size_t>(j)]);
      Qs[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_struct) +
         static_cast<std::size_t>(i)] = mid;
      Qs[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_struct) +
         static_cast<std::size_t>(j)] = mid;
    }
    Qs[static_cast<std::size_t>(i) * static_cast<std::size_t>(n_struct) +
       static_cast<std::size_t>(i)] += 1e-14;
  }
}

double step_to_bound(const std::vector<double>& x, const std::vector<double>& dx) {
  double alpha = 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (dx[i] < 0.0) alpha = std::min(alpha, -x[i] / dx[i]);
  }
  return alpha;
}

IpmQp build_qp_form(const OptimizationModel& model) {
  IpmQp lp;
  lp.original_sense = model.sense;
  lp.n_structural = static_cast<int>(model.variables.size());
  const int m0 = static_cast<int>(model.constraints.size());
  const double sense_sign = (model.sense == Sense::Maximize) ? -1.0 : 1.0;

  struct BoundConstraint {
    int var = 0;
    double ub = 0.0;
  };
  std::vector<BoundConstraint> ub_cons;
  lp.shift.assign(static_cast<std::size_t>(lp.n_structural), 0.0);
  for (int i = 0; i < lp.n_structural; ++i) {
    const Variable& v = model.variables[static_cast<std::size_t>(i)];
    lp.shift[static_cast<std::size_t>(i)] = v.lower_bound;
    if (std::isfinite(v.upper_bound) && v.upper_bound < 1e29) {
      BoundConstraint bc;
      bc.var = i;
      bc.ub = v.upper_bound - v.lower_bound;
      if (bc.ub < -1e-12) {
        throw std::runtime_error("Inconsistent bounds for variable " + v.name);
      }
      ub_cons.push_back(bc);
    }
  }

  std::vector<double> Qs;
  build_Q_structural(model, Qs, lp.n_structural, sense_sign);

  // Linear term after y = x - lb: c_eff = sense*c + Q*lb
  std::vector<double> c_eff(static_cast<std::size_t>(lp.n_structural), 0.0);
  for (int i = 0; i < lp.n_structural; ++i) {
    auto it = model.objective.linear.find(model.variables[static_cast<std::size_t>(i)].name);
    if (it != model.objective.linear.end()) c_eff[static_cast<std::size_t>(i)] = sense_sign * it->second;
  }
  std::vector<double> Qlb;
  matvec_Q(Qs, lp.n_structural, lp.shift, Qlb);
  for (int i = 0; i < lp.n_structural; ++i) {
    c_eff[static_cast<std::size_t>(i)] += Qlb[static_cast<std::size_t>(i)];
  }

  const int m_total = m0 + static_cast<int>(ub_cons.size());
  std::vector<double> rhs(static_cast<std::size_t>(m_total), 0.0);
  std::vector<std::vector<std::pair<int, double>>> rows(static_cast<std::size_t>(m_total));
  auto add_coeff = [&](int row, int col, double val) {
    if (val == 0.0) return;
    rows[static_cast<std::size_t>(row)].push_back({col, val});
  };
  std::unordered_map<std::string, int> var_index;
  for (int i = 0; i < lp.n_structural; ++i) {
    var_index[model.variables[static_cast<std::size_t>(i)].name] = i;
  }
  for (int r = 0; r < m0; ++r) {
    const Constraint& cons = model.constraints[static_cast<std::size_t>(r)];
    double b = cons.rhs;
    for (const auto& kv : cons.linear) {
      auto it = var_index.find(kv.first);
      if (it == var_index.end()) throw std::runtime_error("Unknown variable: " + kv.first);
      add_coeff(r, it->second, kv.second);
      b -= kv.second * lp.shift[static_cast<std::size_t>(it->second)];
    }
    rhs[static_cast<std::size_t>(r)] = b;
  }
  for (std::size_t k = 0; k < ub_cons.size(); ++k) {
    const int row = m0 + static_cast<int>(k);
    add_coeff(row, ub_cons[k].var, 1.0);
    rhs[static_cast<std::size_t>(row)] = ub_cons[k].ub;
  }

  enum class RowKind { Le, Ge, Eq };
  std::vector<RowKind> kinds(static_cast<std::size_t>(m_total), RowKind::Le);
  for (int r = 0; r < m0; ++r) {
    const auto s = model.constraints[static_cast<std::size_t>(r)].sense;
    if (s == ConstraintSense::Le) kinds[static_cast<std::size_t>(r)] = RowKind::Le;
    else if (s == ConstraintSense::Ge) kinds[static_cast<std::size_t>(r)] = RowKind::Ge;
    else kinds[static_cast<std::size_t>(r)] = RowKind::Eq;
  }
  for (std::size_t k = 0; k < ub_cons.size(); ++k) {
    kinds[static_cast<std::size_t>(m0 + static_cast<int>(k))] = RowKind::Le;
  }

  int n_slack = 0;
  for (int r = 0; r < m_total; ++r) {
    if (kinds[static_cast<std::size_t>(r)] != RowKind::Eq) ++n_slack;
  }
  const int n_total = lp.n_structural + n_slack;
  lp.n = n_total;
  lp.m = m_total;
  lp.A.resize(static_cast<std::size_t>(m_total), static_cast<std::size_t>(n_total));
  lp.b = rhs;
  lp.c.assign(static_cast<std::size_t>(n_total), 0.0);
  lp.names.assign(static_cast<std::size_t>(n_total), std::string());
  lp.col_scale.assign(static_cast<std::size_t>(n_total), 1.0);
  lp.Q.assign(static_cast<std::size_t>(n_total) * static_cast<std::size_t>(n_total), 0.0);

  for (int i = 0; i < lp.n_structural; ++i) {
    lp.names[static_cast<std::size_t>(i)] = model.variables[static_cast<std::size_t>(i)].name;
    lp.c[static_cast<std::size_t>(i)] = c_eff[static_cast<std::size_t>(i)];
    for (int j = 0; j < lp.n_structural; ++j) {
      lp.Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(n_total) +
           static_cast<std::size_t>(i)] =
          Qs[static_cast<std::size_t>(j) * static_cast<std::size_t>(lp.n_structural) +
             static_cast<std::size_t>(i)];
    }
  }

  std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n_total));
  for (int r = 0; r < m_total; ++r) {
    for (const auto& e : rows[static_cast<std::size_t>(r)]) {
      cols[static_cast<std::size_t>(e.first)].push_back({r, e.second});
    }
  }
  int slack_col = lp.n_structural;
  for (int r = 0; r < m_total; ++r) {
    if (kinds[static_cast<std::size_t>(r)] == RowKind::Le) {
      cols[static_cast<std::size_t>(slack_col)].push_back({r, 1.0});
      lp.names[static_cast<std::size_t>(slack_col)] = "__slack_" + std::to_string(r);
      ++slack_col;
    } else if (kinds[static_cast<std::size_t>(r)] == RowKind::Ge) {
      cols[static_cast<std::size_t>(slack_col)].push_back({r, -1.0});
      lp.names[static_cast<std::size_t>(slack_col)] = "__surplus_" + std::to_string(r);
      ++slack_col;
    }
  }
  for (int j = 0; j < n_total; ++j) {
    auto& entries = cols[static_cast<std::size_t>(j)];
    std::sort(entries.begin(), entries.end());
    std::vector<std::pair<int, double>> merged;
    for (const auto& e : entries) {
      if (!merged.empty() && merged.back().first == e.first) merged.back().second += e.second;
      else merged.push_back(e);
    }
    for (const auto& e : merged) {
      if (std::abs(e.second) > 0.0) lp.A.push_back(e.first, e.second);
    }
    lp.A.finish_column(static_cast<std::size_t>(j));
  }
  return lp;
}

bool solve_newton_qp(const IpmQp& lp, const std::vector<double>& x,
                     const std::vector<double>& s, const std::vector<double>& rp,
                     const std::vector<double>& rd, const std::vector<double>& rxs,
                     std::vector<double>& dx, std::vector<double>& dy,
                     std::vector<double>& ds) {
  const int m = lp.m;
  const int n = lp.n;
  const int N = n + m;
  // Augmented KKT (column-major):
  // [ Q + X^{-1}S   -A^T ] [dx] = [ -rd + X^{-1} rxs ]
  // [ A               0  ] [dy]   [  rp ]
  // with rp = b - Ax (same convention as LP-IPM: A dx = rp).
  std::vector<double> K(static_cast<std::size_t>(N) * static_cast<std::size_t>(N), 0.0);
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      K[static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
        static_cast<std::size_t>(i)] =
          lp.Q[static_cast<std::size_t>(j) * static_cast<std::size_t>(n) +
               static_cast<std::size_t>(i)];
    }
    const double xj = std::max(x[static_cast<std::size_t>(j)], 1e-16);
    const double sj = std::max(s[static_cast<std::size_t>(j)], 1e-16);
    K[static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
      static_cast<std::size_t>(j)] += sj / xj;
  }
  // A and A^T blocks
  for (int j = 0; j < n; ++j) {
    for (int p = lp.A.col_ptr[static_cast<std::size_t>(j)];
         p < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const int r = lp.A.row_idx[static_cast<std::size_t>(p)];
      const double a = lp.A.values[static_cast<std::size_t>(p)];
      // A in bottom-left: row n+r, col j
      K[static_cast<std::size_t>(j) * static_cast<std::size_t>(N) +
        static_cast<std::size_t>(n + r)] += a;
      // A^T appears as -A^T in the Newton system
      //   (Q+X^{-1}S) dx - A^T dy = -rd + X^{-1} rxs
      K[static_cast<std::size_t>(n + r) * static_cast<std::size_t>(N) +
        static_cast<std::size_t>(j)] -= a;
    }
  }
  // Stabilize zero dual block slightly
  for (int i = 0; i < m; ++i) {
    K[static_cast<std::size_t>(n + i) * static_cast<std::size_t>(N) +
      static_cast<std::size_t>(n + i)] -= 1e-12;
  }

  std::vector<double> rhs(static_cast<std::size_t>(N), 0.0);
  for (int j = 0; j < n; ++j) {
    const double xj = std::max(x[static_cast<std::size_t>(j)], 1e-16);
    // (Q+X^{-1}S) dx - A^T dy = -rd + X^{-1} rxs
    // with rxs defined as in LP-IPM (affine: rxs = -XSe).
    rhs[static_cast<std::size_t>(j)] =
        -rd[static_cast<std::size_t>(j)] + rxs[static_cast<std::size_t>(j)] / xj;
  }
  for (int i = 0; i < m; ++i) {
    rhs[static_cast<std::size_t>(n + i)] = rp[static_cast<std::size_t>(i)];
  }

  DenseLU lu;
  if (!lu.factorize(std::move(K), static_cast<std::size_t>(N))) return false;
  if (!lu.solve(rhs)) return false;

  dx.assign(static_cast<std::size_t>(n), 0.0);
  dy.assign(static_cast<std::size_t>(m), 0.0);
  ds.assign(static_cast<std::size_t>(n), 0.0);
  for (int j = 0; j < n; ++j) dx[static_cast<std::size_t>(j)] = rhs[static_cast<std::size_t>(j)];
  for (int i = 0; i < m; ++i) dy[static_cast<std::size_t>(i)] = rhs[static_cast<std::size_t>(n + i)];

  // S dx + X ds = rxs  =>  ds = (rxs - S dx) / X
  for (int j = 0; j < n; ++j) {
    const double xj = std::max(x[static_cast<std::size_t>(j)], 1e-16);
    ds[static_cast<std::size_t>(j)] =
        (rxs[static_cast<std::size_t>(j)] -
         s[static_cast<std::size_t>(j)] * dx[static_cast<std::size_t>(j)]) /
        xj;
  }
  return true;
}

SolverResult solve_qp_ipm(const IpmQp& lp, const QpInteriorPointOptions& opt,
                          const OptimizationModel& original) {
  SolverResult result;
  const int m = lp.m;
  const int n = lp.n;
  if (n <= 0) {
    result.status = SolverStatus::Error;
    result.message = "QP-IPM: empty model.";
    return result;
  }

  std::vector<double> x(static_cast<std::size_t>(n), 1.0);
  std::vector<double> s(static_cast<std::size_t>(n), 1.0);
  std::vector<double> y(static_cast<std::size_t>(m), 0.0);

  // Mehrotra-like starting point (same spirit as LP-IPM).
  if (m > 0) {
    std::vector<double> ones(static_cast<std::size_t>(n), 1.0);
    // Solve (A A^T) dy = b - A*ones, then push x,s positive.
    std::vector<double> M(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
    for (int j = 0; j < n; ++j) {
      for (int p = lp.A.col_ptr[static_cast<std::size_t>(j)];
           p < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
        const int r = lp.A.row_idx[static_cast<std::size_t>(p)];
        const double a = lp.A.values[static_cast<std::size_t>(p)];
        for (int q = lp.A.col_ptr[static_cast<std::size_t>(j)];
             q < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++q) {
          const int r2 = lp.A.row_idx[static_cast<std::size_t>(q)];
          const double a2 = lp.A.values[static_cast<std::size_t>(q)];
          M[static_cast<std::size_t>(r2) * static_cast<std::size_t>(m) +
            static_cast<std::size_t>(r)] += a * a2;
        }
      }
    }
    DenseLU lu;
    if (lu.factorize(std::move(M), static_cast<std::size_t>(m))) {
      std::vector<double> Ax;
      lp.A.multiply(ones, Ax);
      std::vector<double> dy = lp.b;
      for (int i = 0; i < m; ++i) dy[static_cast<std::size_t>(i)] -= Ax[static_cast<std::size_t>(i)];
      if (lu.solve(dy)) {
        std::vector<double> Atdy = matvec_At(lp.A, dy);
        std::vector<double> Qones;
        matvec_Q(lp.Q, n, ones, Qones);
        for (int j = 0; j < n; ++j) {
          x[static_cast<std::size_t>(j)] =
              std::max(1.0, std::abs(Atdy[static_cast<std::size_t>(j)]));
          s[static_cast<std::size_t>(j)] = std::max(
              1.0, std::abs(Qones[static_cast<std::size_t>(j)] +
                            lp.c[static_cast<std::size_t>(j)] -
                            Atdy[static_cast<std::size_t>(j)]));
        }
        y = dy;
      }
    }
  }

  const double bnorm = std::max(1.0, max_abs(lp.b));
  const double cnorm = std::max(1.0, max_abs(lp.c));
  const double tau = opt.fraction_to_boundary;

  for (int it = 0; it < opt.max_iterations; ++it) {
    std::vector<double> Ax;
    lp.A.multiply(x, Ax);
    std::vector<double> rp = lp.b;
    for (int i = 0; i < m; ++i) rp[static_cast<std::size_t>(i)] -= Ax[static_cast<std::size_t>(i)];

    std::vector<double> Qx;
    matvec_Q(lp.Q, n, x, Qx);
    std::vector<double> Aty = matvec_At(lp.A, y);
    std::vector<double> rd(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
      rd[static_cast<std::size_t>(j)] = Qx[static_cast<std::size_t>(j)] +
                                        lp.c[static_cast<std::size_t>(j)] -
                                        Aty[static_cast<std::size_t>(j)] -
                                        s[static_cast<std::size_t>(j)];
    }

    const double mu = dot(x, s) / static_cast<double>(n);
    const double p_res = max_abs(rp) / bnorm;
    const double d_res = max_abs(rd) / std::max(1.0, cnorm + max_abs(Qx));
    const double gap = mu / (1.0 + std::abs(dot(lp.c, x) + 0.5 * dot(x, Qx)));

    if (p_res < opt.feasibility_tol && d_res < opt.feasibility_tol &&
        gap < opt.optimality_tol) {
      result.status = SolverStatus::Optimal;
      result.iterations = it + 1;
      result.has_objective_value = true;
      result.message =
          "Optimal convex QP found by Mehrotra predictor-corrector IPM "
          "(Hessian in KKT (1,1) block).";

      double obj = original.objective.constant;
      for (int j = 0; j < lp.n_structural; ++j) {
        const double yj = x[static_cast<std::size_t>(j)];
        const double xv = lp.shift[static_cast<std::size_t>(j)] + yj;
        result.primal[lp.names[static_cast<std::size_t>(j)]] = xv;
      }
      for (const auto& kv : original.objective.linear) {
        auto it = result.primal.find(kv.first);
        if (it != result.primal.end()) obj += kv.second * it->second;
      }
      for (const auto& row : original.objective.quadratic) {
        auto iti = result.primal.find(row.first);
        if (iti == result.primal.end()) continue;
        for (const auto& col : row.second) {
          auto itj = result.primal.find(col.first);
          if (itj == result.primal.end()) continue;
          obj += 0.5 * col.second * iti->second * itj->second;
        }
      }
      result.objective_value = obj;
      return result;
    }

    // Affine predictor
    std::vector<double> rxs(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
      rxs[static_cast<std::size_t>(j)] =
          -x[static_cast<std::size_t>(j)] * s[static_cast<std::size_t>(j)];
    }
    std::vector<double> dx_aff, dy_aff, ds_aff;
    if (!solve_newton_qp(lp, x, s, rp, rd, rxs, dx_aff, dy_aff, ds_aff)) {
      result.status = SolverStatus::Error;
      result.message = "QP-IPM Newton factorization failed (affine).";
      result.iterations = it;
      return result;
    }
    const double alpha_p_aff = step_to_bound(x, dx_aff);
    const double alpha_d_aff = step_to_bound(s, ds_aff);
    double mu_aff = 0.0;
    for (int j = 0; j < n; ++j) {
      mu_aff += (x[static_cast<std::size_t>(j)] + alpha_p_aff * dx_aff[static_cast<std::size_t>(j)]) *
                (s[static_cast<std::size_t>(j)] + alpha_d_aff * ds_aff[static_cast<std::size_t>(j)]);
    }
    mu_aff /= static_cast<double>(n);
    const double sigma = (mu > 0.0) ? std::min(1.0, std::pow(mu_aff / mu, 3.0)) : 0.0;

    for (int j = 0; j < n; ++j) {
      rxs[static_cast<std::size_t>(j)] =
          -x[static_cast<std::size_t>(j)] * s[static_cast<std::size_t>(j)] -
          dx_aff[static_cast<std::size_t>(j)] * ds_aff[static_cast<std::size_t>(j)] +
          sigma * mu;
    }
    std::vector<double> dx, dy, ds;
    if (!solve_newton_qp(lp, x, s, rp, rd, rxs, dx, dy, ds)) {
      result.status = SolverStatus::Error;
      result.message = "QP-IPM Newton factorization failed (corrector).";
      result.iterations = it;
      return result;
    }

    double alpha_p = std::min(1.0, tau * step_to_bound(x, dx));
    double alpha_d = std::min(1.0, tau * step_to_bound(s, ds));
    for (int j = 0; j < n; ++j) {
      x[static_cast<std::size_t>(j)] =
          std::max(1e-14, x[static_cast<std::size_t>(j)] + alpha_p * dx[static_cast<std::size_t>(j)]);
      s[static_cast<std::size_t>(j)] =
          std::max(1e-14, s[static_cast<std::size_t>(j)] + alpha_d * ds[static_cast<std::size_t>(j)]);
    }
    for (int i = 0; i < m; ++i) {
      y[static_cast<std::size_t>(i)] += alpha_d * dy[static_cast<std::size_t>(i)];
    }
    result.iterations = it + 1;
  }

  result.status = SolverStatus::Error;
  result.message = "QP-IPM iteration limit reached without convergence.";
  return result;
}

}  // namespace

QpInteriorPointSolver::QpInteriorPointSolver(QpInteriorPointOptions options)
    : options_(std::move(options)) {}

SolverResult QpInteriorPointSolver::solve(const OptimizationModel& model) const {
  try {
    if (model.problem_type != ProblemType::QP && model.problem_type != ProblemType::LP) {
      SolverResult r;
      r.status = SolverStatus::Error;
      r.message = "QpInteriorPointSolver expects QP.";
      return r;
    }
    for (const auto& v : model.variables) {
      if (v.type != VariableType::Continuous) {
        SolverResult r;
        r.status = SolverStatus::Error;
        r.message = "QP-IPM variables must be continuous.";
        return r;
      }
    }
    IpmQp lp = build_qp_form(model);
    return solve_qp_ipm(lp, options_, model);
  } catch (const std::exception& ex) {
    SolverResult r;
    r.status = SolverStatus::Error;
    r.message = std::string("QP-IPM error: ") + ex.what();
    return r;
  }
}

}  // namespace sovereign
