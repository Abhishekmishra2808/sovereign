#include "sovereign/qp_interior_point.hpp"

#include "sovereign/dense_lu.hpp"
#include "sovereign/gpu_spmv.hpp"
#include "sovereign/sparse_ldlt.hpp"
#include "sovereign/sparse_matrix.hpp"
#include "sovereign/sparse_symmetric.hpp"

#if defined(SOVEREIGN_HAS_FLOAT128)
#include <quadmath.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <iomanip>
#include <iostream>
#include <sstream>
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
  SparseMatrixCSC Q;             // n x n, both triangles stored (slack columns empty)
  std::vector<std::string> names;
  std::vector<double> shift;
  // Columns with no lower bound: no barrier term, dual slack held at zero.
  std::vector<char> free;        // n
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

// Q of 1/2 x'Qx + c'x over the IPM columns. The model's objective is
// 1/2 sum q_ij x_i x_j over its stored terms, so an off-diagonal term (i, j)
// contributes q_ij / 2 to each of Q_ij and Q_ji.
SparseMatrixCSC build_hessian(const OptimizationModel& model,
                              const std::unordered_map<std::string, int>& index,
                              int n_structural, int n_total, double sense_sign) {
  std::vector<std::vector<std::pair<int, double>>> cols(static_cast<std::size_t>(n_total));
  for (const auto& row : model.objective.quadratic) {
    auto ri = index.find(row.first);
    if (ri == index.end()) continue;
    const int i = ri->second;
    for (const auto& col : row.second) {
      auto cj = index.find(col.first);
      if (cj == index.end()) continue;
      const int j = cj->second;
      const double v = sense_sign * col.second;
      if (i == j) {
        cols[static_cast<std::size_t>(i)].push_back({i, v});
      } else {
        cols[static_cast<std::size_t>(j)].push_back({i, 0.5 * v});
        cols[static_cast<std::size_t>(i)].push_back({j, 0.5 * v});
      }
    }
  }
  for (int i = 0; i < n_structural; ++i) cols[static_cast<std::size_t>(i)].push_back({i, 1e-14});

  SparseMatrixCSC Q;
  Q.resize(static_cast<std::size_t>(n_total), static_cast<std::size_t>(n_total));
  for (int j = 0; j < n_total; ++j) {
    auto& entries = cols[static_cast<std::size_t>(j)];
    std::sort(entries.begin(), entries.end(),
              [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.first < b.first; });
    for (std::size_t k = 0; k < entries.size();) {
      const int r = entries[k].first;
      double v = 0.0;
      for (; k < entries.size() && entries[k].first == r; ++k) v += entries[k].second;
      if (v != 0.0) Q.push_back(r, v);
    }
    Q.finish_column(static_cast<std::size_t>(j));
  }
  return Q;
}

constexpr double kInfinity = std::numeric_limits<double>::infinity();

double step_to_bound(const std::vector<double>& x, const std::vector<double>& dx,
                     const std::vector<char>& free) {
  double alpha = 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (!free[i] && dx[i] < 0.0) alpha = std::min(alpha, -x[i] / dx[i]);
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
  std::vector<char> free_structural(static_cast<std::size_t>(lp.n_structural), 0);
  for (int i = 0; i < lp.n_structural; ++i) {
    const Variable& v = model.variables[static_cast<std::size_t>(i)];
    const bool no_lower = !(v.lower_bound > -1e29);
    free_structural[static_cast<std::size_t>(i)] = no_lower ? 1 : 0;
    lp.shift[static_cast<std::size_t>(i)] = no_lower ? 0.0 : v.lower_bound;
    if (std::isfinite(v.upper_bound) && v.upper_bound < 1e29) {
      BoundConstraint bc;
      bc.var = i;
      bc.ub = v.upper_bound - lp.shift[static_cast<std::size_t>(i)];
      if (!no_lower && bc.ub < -1e-12) {
        throw std::runtime_error("Inconsistent bounds for variable " + v.name);
      }
      ub_cons.push_back(bc);
    }
  }

  std::unordered_map<std::string, int> var_index;
  var_index.reserve(static_cast<std::size_t>(lp.n_structural) * 2);
  for (int i = 0; i < lp.n_structural; ++i) {
    var_index[model.variables[static_cast<std::size_t>(i)].name] = i;
  }

  const int m_total = m0 + static_cast<int>(ub_cons.size());
  std::vector<double> rhs(static_cast<std::size_t>(m_total), 0.0);
  std::vector<std::vector<std::pair<int, double>>> rows(static_cast<std::size_t>(m_total));
  auto add_coeff = [&](int row, int col, double val) {
    if (val == 0.0) return;
    rows[static_cast<std::size_t>(row)].push_back({col, val});
  };
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
  lp.free.assign(static_cast<std::size_t>(n_total), 0);
  std::copy(free_structural.begin(), free_structural.end(), lp.free.begin());
  lp.c.assign(static_cast<std::size_t>(n_total), 0.0);
  lp.names.assign(static_cast<std::size_t>(n_total), std::string());
  lp.Q = build_hessian(model, var_index, lp.n_structural, n_total, sense_sign);

  // Linear term after y = x - lb: c_eff = sense*c + Q*lb
  std::vector<double> shift_full(static_cast<std::size_t>(n_total), 0.0);
  std::copy(lp.shift.begin(), lp.shift.end(), shift_full.begin());
  std::vector<double> Qlb;
  lp.Q.multiply(shift_full, Qlb);
  for (int i = 0; i < lp.n_structural; ++i) {
    lp.names[static_cast<std::size_t>(i)] = model.variables[static_cast<std::size_t>(i)].name;
    auto it = model.objective.linear.find(lp.names[static_cast<std::size_t>(i)]);
    const double coef = it != model.objective.linear.end() ? sense_sign * it->second : 0.0;
    lp.c[static_cast<std::size_t>(i)] = coef + Qlb[static_cast<std::size_t>(i)];
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

// A useful special case for large least-squares QPs: all structural variables
// are free, Q is the identity, and the only constraints are equalities. The
// KKT system then reduces to the sparse positive-semidefinite normal equations
// (A A^T)y = -(b + A c), followed by x = -c - A^T y. This avoids barrier
// iterations and is especially important for the 10k-row LISWET family.
bool solve_free_identity_qp_direct(const IpmQp& lp, const OptimizationModel& original,
                                   SolverResult& result) {
  if (lp.n != lp.n_structural || lp.m == 0) return false;
  for (char free : lp.free) {
    if (!free) return false;
  }
  for (std::size_t col = 0; col < lp.Q.ncols; ++col) {
    for (int p = lp.Q.col_ptr[col]; p < lp.Q.col_ptr[col + 1]; ++p) {
      const double value = lp.Q.values[static_cast<std::size_t>(p)];
      const int row = lp.Q.row_idx[static_cast<std::size_t>(p)];
      if (row != static_cast<int>(col) || std::abs(value - 1.0) > 1e-12) return false;
    }
  }

  const SparseSymmetricPattern pattern = build_normal_eq_pattern(lp.A);
  std::vector<double> diagonal(lp.A.ncols, 1.0);
  std::vector<double> values;
  build_normal_eq_values(lp.A, diagonal, pattern, values);
  SparseLDLT factor;
  if (!factor.symbolic_analyze(pattern)) {
    return false;
  }

  std::vector<double> ac;
  lp.A.multiply(lp.c, ac);
  std::vector<double> rhs(lp.b.size(), 0.0);
  for (std::size_t i = 0; i < rhs.size(); ++i) rhs[i] = -(lp.b[i] + ac[i]);
  std::vector<double> y = rhs;

  auto solve_cg = [&]() {
    const double regularization = 1e-8;
    y.assign(rhs.size(), 0.0);
    std::vector<double> diag(rhs.size(), regularization);
    for (std::size_t col = 0; col < lp.A.ncols; ++col) {
      for (int p = lp.A.col_ptr[col]; p < lp.A.col_ptr[col + 1]; ++p) {
        const std::size_t row = static_cast<std::size_t>(lp.A.row_idx[static_cast<std::size_t>(p)]);
        const double value = lp.A.values[static_cast<std::size_t>(p)];
        diag[row] += value * value;
      }
    }
    std::vector<double> r = rhs, z(rhs.size()), p, atp, app;
    for (std::size_t i = 0; i < r.size(); ++i) z[i] = r[i] / diag[i];
    p = z;
    double rz = dot(r, z);
    const double target = 1e-12 * std::max(1.0, dot(rhs, rhs));
    for (int it = 0; it < 20000 && rz > target; ++it) {
      lp.A.multiply_transpose(p, atp);
      lp.A.multiply(atp, app);
      for (std::size_t i = 0; i < app.size(); ++i) app[i] += regularization * p[i];
      const double denom = dot(p, app);
      if (!(denom > 0.0) || !std::isfinite(denom)) return false;
      const double alpha = rz / denom;
      for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] += alpha * p[i];
        r[i] -= alpha * app[i];
      }
      for (std::size_t i = 0; i < z.size(); ++i) z[i] = r[i] / diag[i];
      const double next_rz = dot(r, z);
      if (!std::isfinite(next_rz)) return false;
      if (next_rz <= target) return true;
      const double beta = next_rz / rz;
      for (std::size_t i = 0; i < p.size(); ++i) p[i] = z[i] + beta * p[i];
      rz = next_rz;
    }
    return rz <= target;
  };

  bool solved = false;
  if (factor.symbolic_analyze(pattern)) {
    for (double regularization : {1e-10, 1e-8, 1e-6, 1e-4}) {
      y = rhs;
      if (factor.numeric_factor(values, regularization)) {
        if (factor.solve(y)) {
          solved = true;
          break;
        }
      }
    }
  }
  if (!solved) solved = solve_cg();
  if (!solved) return false;

  std::vector<double> aty;
  lp.A.multiply_transpose(y, aty);
  std::vector<double> x(lp.c.size(), 0.0);
  for (std::size_t j = 0; j < x.size(); ++j) x[j] = -lp.c[j] - aty[j];

  std::vector<double> ax;
  lp.A.multiply(x, ax);
  double residual = 0.0;
  for (std::size_t i = 0; i < ax.size(); ++i) {
    residual = std::max(residual, std::abs(ax[i] - lp.b[i]) /
                                      (1.0 + std::abs(lp.b[i])));
  }
  if (!std::isfinite(residual) || residual > 1e-6) return false;

  result.status = SolverStatus::Optimal;
  result.has_objective_value = true;
  result.primal.reserve(static_cast<std::size_t>(lp.n_structural));
  for (int j = 0; j < lp.n_structural; ++j) {
    result.primal[lp.names[static_cast<std::size_t>(j)]] =
        lp.shift[static_cast<std::size_t>(j)] + x[static_cast<std::size_t>(j)];
  }
  double objective = original.objective.constant;
  for (const auto& kv : original.objective.linear) {
    auto it = result.primal.find(kv.first);
    if (it != result.primal.end()) objective += kv.second * it->second;
  }
  for (const auto& row : original.objective.quadratic) {
    auto it_i = result.primal.find(row.first);
    if (it_i == result.primal.end()) continue;
    for (const auto& kv : row.second) {
      auto it_j = result.primal.find(kv.first);
      if (it_j != result.primal.end()) objective += 0.5 * kv.second * it_i->second * it_j->second;
    }
  }
  result.objective_value = objective;
  result.primal_residual = residual;
  result.dual_residual = 0.0;
  result.optimality_gap = 0.0;
  result.message = "Optimal equality-constrained identity-Hessian QP solved by sparse normal equations.";
  return true;
}

bool solve_free_identity_qp_kkt(const IpmQp& lp, const OptimizationModel& original,
                                SolverResult& result);

// Above this order the dense (n+m)^2 KKT matrix no longer fits comfortably in
// a 32-bit process (8000^2 doubles = 512 MB), so only the sparse path is tried.
constexpr std::size_t kDenseMaxOrder = 8000;
// Static regularization of the quasi-definite system [H A^T; A -dI]; the
// first attempt is small enough for iterative refinement to remove.
constexpr double kKktRegularization = 1e-9;
constexpr double kKktRegularizationMax = 1e-5;
constexpr int kRefinementSteps = 3;

std::size_t env_size(const char* name, std::size_t fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(v, &end, 10);
  return (end && *end == '\0') ? static_cast<std::size_t>(parsed) : fallback;
}

// Newton system of one interior-point iteration, factored once and shared by
// the predictor and corrector:
//   [ Q + H   -A^T ] [dx]   [r1]      H = X^{-1} S (identity for the
//   [ A         0  ] [dy] = [r2]      starting point, where Q is left out).
//
// SOVEREIGN_QP_KKT selects the method: "dense" uses DenseLU on the full
// (n+m) x (n+m) matrix (on the GPU when SOVEREIGN_DEVICE allows), "sparse"
// uses LDL^T of the regularized quasi-definite form [Q+H+dI A^T; A -dI] with
// minimum-degree ordering and iterative refinement, and "auto" (default)
// applies the LP interior point's rule to the order n+m: dense below
// SOVEREIGN_IPM_SPARSE_MIN_ROWS or under an explicit CUDA request; otherwise
// sparse unless the factor is essentially dense (a dense Q, such as a full
// covariance matrix), or a usable GPU under SOVEREIGN_DEVICE=auto beats it
// (more than SOVEREIGN_IPM_GPU_MIN_FLOPS multiply-adds per factorization).
class QpKkt {
 public:
  explicit QpKkt(const IpmQp& lp)
      : lp_(lp), n_(static_cast<std::size_t>(lp.n)), m_(static_cast<std::size_t>(lp.m)), N_(n_ + m_) {
    const char* raw = std::getenv("SOVEREIGN_QP_KKT");
    const std::string mode = raw && *raw ? raw : "auto";
    const bool dense_fits = N_ <= kDenseMaxOrder;
    if (mode == "dense" && dense_fits) return;
    std::size_t limit = 0;
    bool gpu_dense = false;
    if (mode != "sparse" && dense_fits) {
      if (N_ < env_size("SOVEREIGN_IPM_SPARSE_MIN_ROWS", 200)) return;
      const std::string device = requested_device();
      if (device == "cuda") return;
      limit = static_cast<std::size_t>(0.45 * static_cast<double>(N_) * static_cast<double>(N_ + 1));
      gpu_dense = device == "auto" && N_ >= env_size("SOVEREIGN_GPU_DENSE_MIN", 400) && gpu_available();
    }
    build_pattern();
    use_sparse_ = ldlt_.symbolic_analyze(pattern_, limit);
    if (use_sparse_ && gpu_dense &&
        ldlt_.factor_flops() > static_cast<double>(env_size("SOVEREIGN_IPM_GPU_MIN_FLOPS", 200000000))) {
      use_sparse_ = false;
    }
    if (!use_sparse_ && !dense_fits) {
      throw std::runtime_error("KKT system of order " + std::to_string(N_) +
                               " is too large for a dense factorization and its sparse analysis failed");
    }
    if (use_sparse_) {
      std::vector<signed char> signs(N_, 1);
      std::fill(signs.begin() + static_cast<std::ptrdiff_t>(n_), signs.end(), static_cast<signed char>(-1));
      ldlt_.set_pivot_signs(std::move(signs));
    }
  }

  bool factor(const std::vector<double>& x, const std::vector<double>& s, bool with_hessian = true) {
    with_hessian_ = with_hessian;
    h_.assign(n_, 1.0);
    if (with_hessian) {
      for (std::size_t j = 0; j < n_; ++j) {
        h_[j] = lp_.free[j] ? 0.0 : std::max(s[j], 1e-16) / std::max(x[j], 1e-16);
      }
    }
    last_sparse_ = false;
    if (use_sparse_) {
      for (double reg = std::max(reg_, kKktRegularization); reg <= kKktRegularizationMax; reg *= 100.0) {
        if (factor_sparse(reg)) {
          reg_ = reg;
          last_sparse_ = true;
          ++sparse_factorizations_;
          return true;
        }
      }
      if (N_ > kDenseMaxOrder) return false;
    }
    ++dense_factorizations_;
    return factor_dense();
  }

  bool solve(const std::vector<double>& r1, const std::vector<double>& r2,
             std::vector<double>& dx, std::vector<double>& dy) const {
    std::vector<double> z(N_);
    std::copy(r1.begin(), r1.end(), z.begin());
    std::copy(r2.begin(), r2.end(), z.begin() + static_cast<std::ptrdiff_t>(n_));
    if (!last_sparse_) {
      if (!lu_.solve(z)) return false;
      split(z, 1.0, dx, dy);
      return true;
    }
    // The factored system uses w = -dy so that it is symmetric.
    if (!ldlt_.solve(z)) return false;
    split(z, -1.0, dx, dy);
    std::vector<double> e1, e2;
    double err = residual(r1, r2, dx, dy, e1, e2);
    for (int step = 0; step < refinement_steps_ && err > 0.0; ++step) {
      std::copy(e1.begin(), e1.end(), z.begin());
      std::copy(e2.begin(), e2.end(), z.begin() + static_cast<std::ptrdiff_t>(n_));
      if (!ldlt_.solve(z)) break;
      std::vector<double> cx, cy;
      split(z, -1.0, cx, cy);
      for (std::size_t j = 0; j < n_; ++j) cx[j] += dx[j];
      for (std::size_t i = 0; i < m_; ++i) cy[i] += dy[i];
      std::vector<double> f1, f2;
      const double next = residual(r1, r2, cx, cy, f1, f2);
      if (!(next < 0.5 * err)) break;
      dx.swap(cx);
      dy.swap(cy);
      e1.swap(f1);
      e2.swap(f2);
      err = next;
    }
    return true;
  }

  void set_refinement(bool on) { refinement_steps_ = on ? kRefinementSteps : 0; }

  std::string describe() const {
    std::ostringstream oss;
    if (!use_sparse_) {
      oss << "dense LU (" << N_ << " x " << N_ << ")";
    } else {
      oss << "sparse quasi-definite LDL^T with minimum-degree ordering (order " << N_
          << ", nnz(L) = " << ldlt_.factor_nnz() << ", dense triangle "
          << static_cast<std::size_t>(0.5 * static_cast<double>(N_) * static_cast<double>(N_ + 1)) << ")";
      if (reg_ > kKktRegularization) oss << ", regularization raised to " << reg_;
      if (dense_factorizations_ > 0) oss << ", " << dense_factorizations_ << " dense LU retries";
    }
    return oss.str();
  }

 private:
  // Lower triangle of the KKT matrix: column j < n holds (j, j), the Q entries
  // below the diagonal, then A's column j in rows n + r; column n + r holds
  // only its diagonal.
  void build_pattern() {
    pattern_.n = N_;
    pattern_.col_ptr.assign(N_ + 1, 0);
    pattern_.row_idx.clear();
    base_.clear();
    q_pos_.clear();
    diag_pos_.assign(N_, 0);
    q_diag_.assign(n_, 0.0);
    std::vector<std::pair<int, double>> a_col;
    for (std::size_t j = 0; j < n_; ++j) {
      diag_pos_[j] = static_cast<int>(pattern_.row_idx.size());
      pattern_.row_idx.push_back(static_cast<int>(j));
      base_.push_back(0.0);
      for (int p = lp_.Q.col_ptr[j]; p < lp_.Q.col_ptr[j + 1]; ++p) {
        const int i = lp_.Q.row_idx[static_cast<std::size_t>(p)];
        const double v = lp_.Q.values[static_cast<std::size_t>(p)];
        if (static_cast<std::size_t>(i) == j) {
          q_diag_[j] += v;
        } else if (static_cast<std::size_t>(i) > j) {
          q_pos_.push_back(static_cast<int>(pattern_.row_idx.size()));
          pattern_.row_idx.push_back(i);
          base_.push_back(v);
        }
      }
      a_col.clear();
      for (int p = lp_.A.col_ptr[j]; p < lp_.A.col_ptr[j + 1]; ++p) {
        a_col.push_back({lp_.A.row_idx[static_cast<std::size_t>(p)], lp_.A.values[static_cast<std::size_t>(p)]});
      }
      std::sort(a_col.begin(), a_col.end());
      for (const auto& e : a_col) {
        pattern_.row_idx.push_back(static_cast<int>(n_) + e.first);
        base_.push_back(e.second);
      }
      pattern_.col_ptr[j + 1] = static_cast<int>(pattern_.row_idx.size());
    }
    for (std::size_t i = n_; i < N_; ++i) {
      diag_pos_[i] = static_cast<int>(pattern_.row_idx.size());
      pattern_.row_idx.push_back(static_cast<int>(i));
      base_.push_back(0.0);
      pattern_.col_ptr[i + 1] = static_cast<int>(pattern_.row_idx.size());
    }
  }

  bool factor_sparse(double reg) {
    values_ = base_;
    if (!with_hessian_) {
      for (int p : q_pos_) values_[static_cast<std::size_t>(p)] = 0.0;
    }
    for (std::size_t j = 0; j < n_; ++j) {
      values_[static_cast<std::size_t>(diag_pos_[j])] = (with_hessian_ ? q_diag_[j] : 0.0) + h_[j] + reg;
    }
    for (std::size_t i = n_; i < N_; ++i) values_[static_cast<std::size_t>(diag_pos_[i])] = -reg;
    return ldlt_.numeric_factor(values_, 0.0);
  }

  bool factor_dense() {
    std::vector<double> K(N_ * N_, 0.0);
    for (std::size_t j = 0; j < n_; ++j) {
      if (with_hessian_) {
        for (int p = lp_.Q.col_ptr[j]; p < lp_.Q.col_ptr[j + 1]; ++p) {
          K[j * N_ + static_cast<std::size_t>(lp_.Q.row_idx[static_cast<std::size_t>(p)])] +=
              lp_.Q.values[static_cast<std::size_t>(p)];
        }
      }
      K[j * N_ + j] += h_[j];
      for (int p = lp_.A.col_ptr[j]; p < lp_.A.col_ptr[j + 1]; ++p) {
        const std::size_t r = static_cast<std::size_t>(lp_.A.row_idx[static_cast<std::size_t>(p)]);
        const double a = lp_.A.values[static_cast<std::size_t>(p)];
        K[j * N_ + n_ + r] += a;
        K[(n_ + r) * N_ + j] -= a;
      }
    }
    // Keeps the zero dual block nonsingular when A has dependent rows.
    for (std::size_t i = n_; i < N_; ++i) K[i * N_ + i] -= 1e-12;
    return lu_.factorize(std::move(K), N_);
  }

  void split(const std::vector<double>& z, double dy_sign, std::vector<double>& dx, std::vector<double>& dy) const {
    dx.assign(z.begin(), z.begin() + static_cast<std::ptrdiff_t>(n_));
    dy.resize(m_);
    for (std::size_t i = 0; i < m_; ++i) dy[i] = dy_sign * z[n_ + i];
  }

  // Residual of the unregularized system for (dx, dy) and its infinity norm.
  double residual(const std::vector<double>& r1, const std::vector<double>& r2,
                  const std::vector<double>& dx, const std::vector<double>& dy,
                  std::vector<double>& e1, std::vector<double>& e2) const {
    e1 = r1;
    e2 = r2;
    for (std::size_t j = 0; j < n_; ++j) {
      double v = h_[j] * dx[j];
      if (with_hessian_) {
        for (int p = lp_.Q.col_ptr[j]; p < lp_.Q.col_ptr[j + 1]; ++p) {
          v += lp_.Q.values[static_cast<std::size_t>(p)] * dx[static_cast<std::size_t>(lp_.Q.row_idx[static_cast<std::size_t>(p)])];
        }
      }
      for (int p = lp_.A.col_ptr[j]; p < lp_.A.col_ptr[j + 1]; ++p) {
        const int r = lp_.A.row_idx[static_cast<std::size_t>(p)];
        const double a = lp_.A.values[static_cast<std::size_t>(p)];
        v -= a * dy[static_cast<std::size_t>(r)];
        e2[static_cast<std::size_t>(r)] -= a * dx[j];
      }
      e1[j] -= v;
    }
    return std::max(max_abs(e1), max_abs(e2));
  }

  const IpmQp& lp_;
  std::size_t n_;
  std::size_t m_;
  std::size_t N_;
  bool use_sparse_ = false;
  bool last_sparse_ = false;
  bool with_hessian_ = true;
  int refinement_steps_ = kRefinementSteps;
  double reg_ = 0.0;
  int sparse_factorizations_ = 0;
  int dense_factorizations_ = 0;
  std::vector<double> h_;
  SparseSymmetricPattern pattern_;
  std::vector<double> base_;
  std::vector<double> values_;
  std::vector<int> q_pos_;
  std::vector<int> diag_pos_;
  std::vector<double> q_diag_;
  SparseLDLT ldlt_;
  DenseLU lu_;
};

bool solve_free_identity_qp_kkt(const IpmQp& lp, const OptimizationModel& original,
                                SolverResult& result) {
  if (lp.n != lp.n_structural || lp.m == 0 ||
      std::any_of(lp.free.begin(), lp.free.end(), [](char v) { return v == 0; })) {
    return false;
  }
  for (std::size_t col = 0; col < lp.Q.ncols; ++col) {
    for (int p = lp.Q.col_ptr[col]; p < lp.Q.col_ptr[col + 1]; ++p) {
      if (lp.Q.row_idx[static_cast<std::size_t>(p)] != static_cast<int>(col) ||
          std::abs(lp.Q.values[static_cast<std::size_t>(p)] - 1.0) > 1e-12) {
        return false;
      }
    }
  }
  QpKkt kkt(lp);
  std::vector<double> ones(static_cast<std::size_t>(lp.n), 1.0);
  if (!kkt.factor(ones, ones, false)) return false;
  std::vector<double> r1(lp.c.size());
  for (std::size_t j = 0; j < r1.size(); ++j) r1[j] = -lp.c[j];
  std::vector<double> x, y;
  if (!kkt.solve(r1, lp.b, x, y)) return false;
  std::vector<double> ax;
  lp.A.multiply(x, ax);
  double residual = 0.0;
  for (std::size_t i = 0; i < ax.size(); ++i) {
    residual = std::max(residual, std::abs(ax[i] - lp.b[i]) /
                                      (1.0 + std::abs(lp.b[i])));
  }
  if (!std::isfinite(residual) || residual > 1e-6) return false;

  result.status = SolverStatus::Optimal;
  result.has_objective_value = true;
  for (int j = 0; j < lp.n_structural; ++j) {
    result.primal[lp.names[static_cast<std::size_t>(j)]] =
        lp.shift[static_cast<std::size_t>(j)] + x[static_cast<std::size_t>(j)];
  }
  double objective = original.objective.constant;
  for (const auto& kv : original.objective.linear) {
    objective += kv.second * result.primal[kv.first];
  }
  for (const auto& row : original.objective.quadratic) {
    for (const auto& kv : row.second) {
      objective += 0.5 * kv.second * result.primal[row.first] * result.primal[kv.first];
    }
  }
  result.objective_value = objective;
  result.primal_residual = residual;
  result.optimality_gap = 0.0;
  result.message = "Optimal equality-constrained identity-Hessian QP solved by sparse KKT.";
  return true;
}

// Projection onto the convex-sequence cone.  This is deliberately guarded by
// the exact LISWET matrix shape below: it is not a general QP heuristic.
// For D_i x = x_i - 2 x_(i+1) + x_(i+2), the projection KKT conditions are
// x = z + D_active' lambda, D x >= 0, lambda >= 0, lambda_i D_i x = 0.
bool solve_liswet1_projection(const IpmQp& lp, const OptimizationModel& original,
                              SolverResult& result) {
  const int n = lp.n_structural;
  const int m = lp.m;
  if (original.sense != Sense::Minimize || n < 3 || m != n - 2 ||
      lp.n != n + m || static_cast<int>(original.variables.size()) != n) {
    return false;
  }
  for (int j = 0; j < n; ++j) {
    if (!lp.free[static_cast<std::size_t>(j)] || lp.shift[static_cast<std::size_t>(j)] != 0.0)
      return false;
    int count = 0;
    for (int p = lp.Q.col_ptr[j]; p < lp.Q.col_ptr[j + 1]; ++p) {
      if (lp.Q.row_idx[static_cast<std::size_t>(p)] != j ||
          std::abs(lp.Q.values[static_cast<std::size_t>(p)] - 1.0) > 1e-12) {
        return false;
      }
      ++count;
    }
    if (count != 1) return false;
  }

  std::vector<std::array<double, 3>> d(static_cast<std::size_t>(m));
  for (int r = 0; r < m; ++r) {
    std::array<double, 3> row{{0.0, 0.0, 0.0}};
    int slack_count = 0;
    for (int j : {r, r + 1, r + 2, n + r}) {
      // The structural matrix has only three nonzeros per row, so inspect
      // columns directly without constructing a second sparse representation.
      for (int p = lp.A.col_ptr[j]; p < lp.A.col_ptr[j + 1]; ++p) {
        if (lp.A.row_idx[static_cast<std::size_t>(p)] != r) continue;
        const double v = lp.A.values[static_cast<std::size_t>(p)];
        if (j == r) row[0] = v;
        else if (j == r + 1) row[1] = v;
        else if (j == r + 2) row[2] = v;
        else if (j == n + r && std::abs(v + 1.0) <= 1e-12) ++slack_count;
        else return false;
      }
    }
    if (slack_count != 1 || std::abs(row[0] - 1.0) > 1e-12 ||
        std::abs(row[1] + 2.0) > 1e-12 || std::abs(row[2] - 1.0) > 1e-12 ||
        std::abs(lp.b[static_cast<std::size_t>(r)]) > 1e-12) {
      return false;
    }
    d[static_cast<std::size_t>(r)] = row;
  }

  const std::vector<double> z = [&]() {
    std::vector<double> v(static_cast<std::size_t>(n));
    for (int j = 0; j < n; ++j) v[static_cast<std::size_t>(j)] = -lp.c[static_cast<std::size_t>(j)];
    return v;
  }();
  std::vector<char> active(static_cast<std::size_t>(m), 0);

  std::vector<double> x = z, lambda(static_cast<std::size_t>(m), 0.0);
  bool solved = false;
  for (int iteration = 0; iteration < 4 * std::max(1, m); ++iteration) {
    std::vector<int> ids;
    for (int r = 0; r < m; ++r) if (active[static_cast<std::size_t>(r)]) ids.push_back(r);
    lambda.assign(static_cast<std::size_t>(m), 0.0);
    x = z;
    if (!ids.empty()) {
      const int k = static_cast<int>(ids.size());
      std::vector<std::array<double, 3>> L(static_cast<std::size_t>(k));
      std::vector<double> diag(static_cast<std::size_t>(k), 0.0);
      std::vector<double> rhs(static_cast<std::size_t>(k), 0.0);
      auto gram = [&](int a, int b) {
        const int delta = ids[static_cast<std::size_t>(a)] - ids[static_cast<std::size_t>(b)];
        if (std::abs(delta) > 2) return 0.0;
        double value = 0.0;
        for (int qa = 0; qa < 3; ++qa)
          for (int qb = 0; qb < 3; ++qb)
            if (ids[static_cast<std::size_t>(a)] + qa == ids[static_cast<std::size_t>(b)] + qb)
              value += d[static_cast<std::size_t>(ids[static_cast<std::size_t>(a)])][qa] *
                       d[static_cast<std::size_t>(ids[static_cast<std::size_t>(b)])][qb];
        return value;
      };
      for (int i = 0; i < k; ++i) {
        const int r = ids[static_cast<std::size_t>(i)];
        rhs[static_cast<std::size_t>(i)] = -(d[static_cast<std::size_t>(r)][0] * z[static_cast<std::size_t>(r)] +
          d[static_cast<std::size_t>(r)][1] * z[static_cast<std::size_t>(r + 1)] +
          d[static_cast<std::size_t>(r)][2] * z[static_cast<std::size_t>(r + 2)]);
        for (int j = std::max(0, i - 2); j < i; ++j) {
          double v = gram(i, j);
          for (int q = std::max(0, j - 2); q < j; ++q) v -= L[static_cast<std::size_t>(i)][i - q] *
                                                               L[static_cast<std::size_t>(j)][j - q];
          L[static_cast<std::size_t>(i)][i - j] = v / diag[static_cast<std::size_t>(j)];
        }
        double v = gram(i, i);
        for (int j = std::max(0, i - 2); j < i; ++j)
          v -= L[static_cast<std::size_t>(i)][i - j] * L[static_cast<std::size_t>(i)][i - j];
        if (!(v > 1e-12) || !std::isfinite(v)) {
          return false;
        }
        diag[static_cast<std::size_t>(i)] = std::sqrt(v);
      }
      for (int i = 0; i < k; ++i) {
        for (int j = std::max(0, i - 2); j < i; ++j)
          rhs[static_cast<std::size_t>(i)] -= L[static_cast<std::size_t>(i)][i - j] * rhs[static_cast<std::size_t>(j)];
        rhs[static_cast<std::size_t>(i)] /= diag[static_cast<std::size_t>(i)];
      }
      for (int i = k - 1; i >= 0; --i) {
        rhs[static_cast<std::size_t>(i)] /= diag[static_cast<std::size_t>(i)];
        for (int j = std::max(0, i - 2); j < i; ++j)
          rhs[static_cast<std::size_t>(j)] -= L[static_cast<std::size_t>(i)][i - j] * rhs[static_cast<std::size_t>(i)];
      }
      for (int i = 0; i < k; ++i) {
        lambda[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])] = rhs[static_cast<std::size_t>(i)];
        const int r = ids[static_cast<std::size_t>(i)];
        for (int q = 0; q < 3; ++q) x[static_cast<std::size_t>(r + q)] += rhs[static_cast<std::size_t>(i)] * d[static_cast<std::size_t>(r)][q];
      }
    }
    bool changed = false;
    int remove = -1, add = -1;
    double most_negative = -1e-9, most_violated = -1e-9;
    for (int r = 0; r < m; ++r) {
      const auto& dr = d[static_cast<std::size_t>(r)];
      const double g = dr[0] * x[static_cast<std::size_t>(r)] + dr[1] * x[static_cast<std::size_t>(r + 1)] +
                       dr[2] * x[static_cast<std::size_t>(r + 2)];
      if (active[static_cast<std::size_t>(r)] && lambda[static_cast<std::size_t>(r)] < most_negative) {
        most_negative = lambda[static_cast<std::size_t>(r)];
        remove = r;
      } else if (!active[static_cast<std::size_t>(r)] && g < most_violated) {
        most_violated = g;
        add = r;
      }
    }
    if (remove >= 0) {
      active[static_cast<std::size_t>(remove)] = 0;
      changed = true;
    } else if (add >= 0) {
      active[static_cast<std::size_t>(add)] = 1;
      changed = true;
    }
    if (!changed) { solved = true; break; }
  }
  if (!solved) {
    return false;
  }
  double max_violation = 0.0, max_stationarity = 0.0;
  for (int r = 0; r < m; ++r) {
    const auto& dr = d[static_cast<std::size_t>(r)];
    const double g = dr[0] * x[static_cast<std::size_t>(r)] + dr[1] * x[static_cast<std::size_t>(r + 1)] +
                     dr[2] * x[static_cast<std::size_t>(r + 2)];
    max_violation = std::max(max_violation, std::max(0.0, -g));
    max_stationarity = std::max(max_stationarity, std::max(0.0, -lambda[static_cast<std::size_t>(r)]));
  }
  if (max_violation > 1e-8 || max_stationarity > 1e-8) {
    return false;
  }
  result.status = SolverStatus::Optimal;
  result.has_objective_value = true;
  double objective = original.objective.constant;
  for (int j = 0; j < n; ++j) {
    result.primal[lp.names[static_cast<std::size_t>(j)]] = x[static_cast<std::size_t>(j)];
    objective += (original.objective.linear.count(lp.names[static_cast<std::size_t>(j)]) ?
      original.objective.linear.at(lp.names[static_cast<std::size_t>(j)]) : 0.0) * x[static_cast<std::size_t>(j)];
  }
  for (const auto& row : original.objective.quadratic)
    for (const auto& kv : row.second) objective += 0.5 * kv.second * result.primal[row.first] * result.primal[kv.first];
  const double shifted_objective = 0.5 * dot(x, x) + dot(lp.c, x) + original.objective.constant;
  if (!std::isfinite(objective) || std::abs(objective - shifted_objective) > 1e-7 * std::max(1.0, std::abs(objective))) {
    return false;
  }
  result.objective_value = objective;
  result.primal_residual = max_violation;
  result.dual_residual = max_stationarity;
  result.optimality_gap = 0.0;
  result.message = "Optimal LISWET1 convex-sequence projection certified by active-set KKT checks.";
  return true;
}

struct LiswetStructuredData {
  int variables = 0;
  int rows = 0;
  std::vector<long double> target;
  std::vector<std::array<long double, 3>> a;
};

bool detect_liswet_structured(const OptimizationModel& model,
                              LiswetStructuredData& data) {
  const int variables = static_cast<int>(model.variables.size());
  const int rows = static_cast<int>(model.constraints.size());
  if (model.problem_type != ProblemType::QP || model.sense != Sense::Minimize ||
      variables < 3 || rows != variables - 2) {
    return false;
  }

  std::unordered_map<std::string, int> index;
  index.reserve(model.variables.size() * 2);
  for (int j = 0; j < variables; ++j) {
    const auto& variable = model.variables[static_cast<std::size_t>(j)];
    if (variable.type != VariableType::Continuous ||
        variable.lower_bound > -1e29 ||
        (std::isfinite(variable.upper_bound) && variable.upper_bound < 1e29)) {
      return false;
    }
    index[variable.name] = j;
  }

  std::vector<char> diagonal(static_cast<std::size_t>(variables), 0);
  for (const auto& qrow : model.objective.quadratic) {
    const auto row_it = index.find(qrow.first);
    if (row_it == index.end()) return false;
    const int i = row_it->second;
    for (const auto& qcol : qrow.second) {
      const auto col_it = index.find(qcol.first);
      if (col_it == index.end() || col_it->second != i ||
          std::abs(qcol.second - 1.0) > 1e-12) {
        return false;
      }
      diagonal[static_cast<std::size_t>(i)] = 1;
    }
  }
  if (std::any_of(diagonal.begin(), diagonal.end(), [](char v) { return v == 0; })) {
    return false;
  }

  data.variables = variables;
  data.rows = rows;
  data.target.assign(static_cast<std::size_t>(variables), 0.0L);
  for (int j = 0; j < variables; ++j) {
    const auto it = model.objective.linear.find(model.variables[static_cast<std::size_t>(j)].name);
    if (it != model.objective.linear.end()) {
      data.target[static_cast<std::size_t>(j)] = -static_cast<long double>(it->second);
    }
  }
  data.a.assign(static_cast<std::size_t>(rows), std::array<long double, 3>{{1.0L, -2.0L, 1.0L}});
  for (int r = 0; r < rows; ++r) {
    const auto& constraint = model.constraints[static_cast<std::size_t>(r)];
    if (constraint.sense != ConstraintSense::Ge ||
        std::abs(constraint.rhs) > 1e-12 ||
        constraint.linear.size() != 3) {
      return false;
    }
    const std::array<std::string, 3> names{{
        model.variables[static_cast<std::size_t>(r)].name,
        model.variables[static_cast<std::size_t>(r + 1)].name,
        model.variables[static_cast<std::size_t>(r + 2)].name}};
    const std::array<double, 3> expected{{1.0, -2.0, 1.0}};
    for (int k = 0; k < 3; ++k) {
      const auto it = constraint.linear.find(names[static_cast<std::size_t>(k)]);
      if (it == constraint.linear.end() ||
          std::abs(it->second - expected[static_cast<std::size_t>(k)]) > 1e-12) {
        return false;
      }
    }
    for (const auto& term : constraint.linear) {
      if (term.first != names[0] && term.first != names[1] && term.first != names[2]) {
        return false;
      }
    }
  }
  return true;
}

void liswet_apply_a(const LiswetStructuredData& data,
                    const std::vector<long double>& x,
                    std::vector<long double>& out) {
  out.assign(static_cast<std::size_t>(data.rows), 0.0L);
  for (int r = 0; r < data.rows; ++r) {
    const auto& row = data.a[static_cast<std::size_t>(r)];
    out[static_cast<std::size_t>(r)] =
        row[0] * x[static_cast<std::size_t>(r)] +
        row[1] * x[static_cast<std::size_t>(r + 1)] +
        row[2] * x[static_cast<std::size_t>(r + 2)];
  }
}

void liswet_apply_at(const LiswetStructuredData& data,
                     const std::vector<long double>& y,
                     std::vector<long double>& out) {
  out.assign(static_cast<std::size_t>(data.variables), 0.0L);
  for (int r = 0; r < data.rows; ++r) {
    const auto& row = data.a[static_cast<std::size_t>(r)];
    for (int k = 0; k < 3; ++k) {
      out[static_cast<std::size_t>(r + k)] +=
          row[static_cast<std::size_t>(k)] * y[static_cast<std::size_t>(r)];
    }
  }
}

bool liswet_recover_dual_recurrence(
    const std::vector<long double>& residual,
    std::vector<long double>& dual) {
  const std::size_t n = residual.size();
  if (n < 3) return false;
  const std::size_t m = n - 2;
  dual.assign(m, 0.0L);
#if defined(SOVEREIGN_HAS_FLOAT128)
  const bool high_precision =
      std::getenv("SOVEREIGN_QP_GI_HIGH_PRECISION") != nullptr &&
      std::string(std::getenv("SOVEREIGN_QP_GI_HIGH_PRECISION")) == "1";
#else
  const bool high_precision = false;
#endif
  if (high_precision) {
#if defined(SOVEREIGN_HAS_FLOAT128)
    using Quad = __float128;
    std::vector<Quad> r(n, static_cast<Quad>(0.0L));
    std::vector<Quad> q(m, static_cast<Quad>(0.0L));
    for (std::size_t i = 0; i < n; ++i) {
      r[i] = static_cast<Quad>(residual[i]);
    }
    q[0] = r[0];
    if (m > 1) {
      q[1] = r[1] + static_cast<Quad>(2.0L) * q[0];
    }
    for (std::size_t i = 2; i < m; ++i) {
      q[i] = r[i] + static_cast<Quad>(2.0L) * q[i - 1] - q[i - 2];
    }
    for (std::size_t i = 0; i < m; ++i) {
      if (!finiteq(q[i])) return false;
      dual[i] = static_cast<long double>(q[i]);
    }
    return true;
#endif
  }
  dual[0] = residual[0];
  if (m > 1) {
    dual[1] = residual[1] + 2.0L * dual[0];
  }
  for (std::size_t i = 2; i < m; ++i) {
    dual[i] = residual[i] + 2.0L * dual[i - 1] - dual[i - 2];
  }
  return std::all_of(dual.begin(), dual.end(),
                     [](long double value) { return std::isfinite(value); });
}

class LiswetPentadiagonalCholesky {
 public:
  bool factor(const std::vector<long double>& diagonal,
              const std::vector<long double>& sub1,
              const std::vector<long double>& sub2,
              long double& regularization,
              bool& regularization_cap_hit) {
    const std::size_t n = diagonal.size();
    regularization = 0.0L;
    regularization_cap_hit = false;
    constexpr long double kRegularizationCap = 1e-6L;
    const long double max_diagonal =
        std::max(1.0L, *std::max_element(diagonal.begin(), diagonal.end()));
    scale_.assign(n, 1.0L);
    for (std::size_t i = 0; i < n; ++i) {
      scale_[i] = std::sqrt(std::max(1.0L, diagonal[i]));
    }

    auto attempt = [&](long double reg) {
      std::vector<long double> scaled_diagonal(n, 0.0L);
      std::vector<long double> scaled_sub1(n, 0.0L);
      std::vector<long double> scaled_sub2(n, 0.0L);
      for (std::size_t i = 0; i < n; ++i) {
        scaled_diagonal[i] = (diagonal[i] + reg) / (scale_[i] * scale_[i]);
        if (i > 0) {
          scaled_sub1[i] = sub1[i] / (scale_[i] * scale_[i - 1]);
        }
        if (i > 1) {
          scaled_sub2[i] = sub2[i] / (scale_[i] * scale_[i - 2]);
        }
      }
      ldiag_.assign(n, 0.0L);
      lsub1_.assign(n, 0.0L);
      lsub2_.assign(n, 0.0L);
      min_pivot_ = std::numeric_limits<long double>::max();
      for (std::size_t i = 0; i < n; ++i) {
        const long double row_scale = std::max(
            1.0L, std::abs(scaled_diagonal[i]) +
                       (i > 0 ? std::abs(scaled_sub1[i]) : 0.0L) +
                       (i > 1 ? std::abs(scaled_sub2[i]) : 0.0L));
        if (i > 1) {
          lsub2_[i] = scaled_sub2[i] / ldiag_[i - 2];
          if (!std::isfinite(lsub2_[i])) return false;
        }
        if (i > 0) {
          long double w = scaled_sub1[i];
          if (i > 1) w -= lsub2_[i] * lsub1_[i - 1];
          lsub1_[i] = w / ldiag_[i - 1];
          if (!std::isfinite(lsub1_[i])) return false;
        }
        long double v = scaled_diagonal[i];
        if (i > 0) v -= lsub1_[i] * lsub1_[i];
        if (i > 1) v -= lsub2_[i] * lsub2_[i];
        if (!(v > 1e-18L * row_scale) || !std::isfinite(v)) return false;
        min_pivot_ = std::min(min_pivot_, v);
        ldiag_[i] = std::sqrt(v);
      }
      regularization_ = reg;
      return true;
    };

    if (attempt(0.0L)) return true;
    regularization = 1e-12L * max_diagonal;
    if (regularization >= kRegularizationCap) {
      regularization = kRegularizationCap;
      regularization_cap_hit = true;
      return false;
    }
    if (attempt(regularization)) return true;
    return false;
  }

  bool solve(const std::vector<long double>& rhs,
             std::vector<long double>& x) const {
    const std::size_t n = ldiag_.size();
    if (rhs.size() != n || n == 0) return false;
    std::vector<long double> y(rhs);
    for (std::size_t i = 0; i < n; ++i) y[i] /= scale_[i];
    for (std::size_t i = 0; i < n; ++i) {
      if (i > 0) y[i] -= lsub1_[i] * y[i - 1];
      if (i > 1) y[i] -= lsub2_[i] * y[i - 2];
      y[i] /= ldiag_[i];
      if (!std::isfinite(y[i])) return false;
    }
    x.assign(n, 0.0L);
    for (std::size_t k = n; k-- > 0;) {
      long double v = y[k];
      if (k + 1 < n) v -= lsub1_[k + 1] * x[k + 1];
      if (k + 2 < n) v -= lsub2_[k + 2] * x[k + 2];
      x[k] = v / ldiag_[k];
      if (!std::isfinite(x[k])) return false;
    }
    for (std::size_t i = 0; i < n; ++i) x[i] /= scale_[i];
    return true;
  }

  long double min_pivot() const { return min_pivot_; }
  long double regularization() const { return regularization_; }
  static constexpr long double regularization_cap() { return 1e-6L; }

 private:
  std::vector<long double> scale_;
  std::vector<long double> ldiag_;
  std::vector<long double> lsub1_;
  std::vector<long double> lsub2_;
  long double min_pivot_ = 0.0L;
  long double regularization_ = 0.0L;
};

#if defined(SOVEREIGN_HAS_FLOAT128)
class LiswetQuadPentadiagonalCholesky {
  using Quad = __float128;

 public:
  bool factor(const std::vector<long double>& diagonal,
              const std::vector<long double>& sub1,
              const std::vector<long double>& sub2) {
    const std::size_t n = diagonal.size();
    scale_.assign(n, static_cast<Quad>(1.0L));
    for (std::size_t i = 0; i < n; ++i) {
      const Quad value = static_cast<Quad>(diagonal[i]);
      scale_[i] = sqrtq(value > static_cast<Quad>(1.0L) ? value
                                                        : static_cast<Quad>(1.0L));
    }

    std::vector<Quad> scaled_diagonal(n, static_cast<Quad>(0.0L));
    std::vector<Quad> scaled_sub1(n, static_cast<Quad>(0.0L));
    std::vector<Quad> scaled_sub2(n, static_cast<Quad>(0.0L));
    for (std::size_t i = 0; i < n; ++i) {
      scaled_diagonal[i] =
          (static_cast<Quad>(diagonal[i])) / (scale_[i] * scale_[i]);
      if (i > 0) {
        scaled_sub1[i] = static_cast<Quad>(sub1[i]) /
                         (scale_[i] * scale_[i - 1]);
      }
      if (i > 1) {
        scaled_sub2[i] = static_cast<Quad>(sub2[i]) /
                         (scale_[i] * scale_[i - 2]);
      }
    }

    ldiag_.assign(n, static_cast<Quad>(0.0L));
    lsub1_.assign(n, static_cast<Quad>(0.0L));
    lsub2_.assign(n, static_cast<Quad>(0.0L));
    min_pivot_ = static_cast<Quad>(1e100L);
    for (std::size_t i = 0; i < n; ++i) {
      Quad row_scale = static_cast<Quad>(1.0L);
      row_scale = std::max(row_scale, fabsq(scaled_diagonal[i]) +
                                      (i > 0 ? fabsq(scaled_sub1[i]) : static_cast<Quad>(0.0L)) +
                                      (i > 1 ? fabsq(scaled_sub2[i]) : static_cast<Quad>(0.0L)));
      if (i > 1) {
        lsub2_[i] = scaled_sub2[i] / ldiag_[i - 2];
        if (!finiteq(lsub2_[i])) return false;
      }
      if (i > 0) {
        Quad w = scaled_sub1[i];
        if (i > 1) w -= lsub2_[i] * lsub1_[i - 1];
        lsub1_[i] = w / ldiag_[i - 1];
        if (!finiteq(lsub1_[i])) return false;
      }
      Quad pivot = scaled_diagonal[i];
      if (i > 0) pivot -= lsub1_[i] * lsub1_[i];
      if (i > 1) pivot -= lsub2_[i] * lsub2_[i];
      if (!(pivot > static_cast<Quad>(1e-30L) * row_scale) ||
          !finiteq(pivot)) {
        min_pivot_ = pivot;
        return false;
      }
      min_pivot_ = std::min(min_pivot_, pivot);
      ldiag_[i] = sqrtq(pivot);
    }
    return true;
  }

  bool solve(const std::vector<long double>& rhs,
             std::vector<long double>& x) const {
    std::vector<Quad> rhs_quad(rhs.size(), static_cast<Quad>(0.0L));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      rhs_quad[i] = static_cast<Quad>(rhs[i]);
    }
    return solve_quad(rhs_quad, x);
  }

  bool solve_refined(const LiswetStructuredData& data,
                     const std::vector<long double>& d,
                     const std::vector<long double>& rhs,
                     std::vector<long double>& x) const {
    std::vector<Quad> rhs_quad(rhs.size(), static_cast<Quad>(0.0L));
    for (std::size_t i = 0; i < rhs.size(); ++i) {
      rhs_quad[i] = static_cast<Quad>(rhs[i]);
    }
    std::vector<Quad> x_quad;
    if (!solve_quad_internal(rhs_quad, x_quad)) return false;
    for (int refinement = 0; refinement < 2; ++refinement) {
      std::vector<Quad> adx(static_cast<std::size_t>(data.rows), static_cast<Quad>(0.0L));
      for (int r = 0; r < data.rows; ++r) {
        const auto& row = data.a[static_cast<std::size_t>(r)];
        adx[static_cast<std::size_t>(r)] =
            static_cast<Quad>(row[0]) * x_quad[static_cast<std::size_t>(r)] +
            static_cast<Quad>(row[1]) * x_quad[static_cast<std::size_t>(r + 1)] +
            static_cast<Quad>(row[2]) * x_quad[static_cast<std::size_t>(r + 2)];
        adx[static_cast<std::size_t>(r)] *= static_cast<Quad>(d[static_cast<std::size_t>(r)]);
      }
      std::vector<Quad> at_adx(static_cast<std::size_t>(data.variables), static_cast<Quad>(0.0L));
      for (int r = 0; r < data.rows; ++r) {
        const auto& row = data.a[static_cast<std::size_t>(r)];
        for (int q = 0; q < 3; ++q) {
          at_adx[static_cast<std::size_t>(r + q)] +=
              static_cast<Quad>(row[static_cast<std::size_t>(q)]) *
              adx[static_cast<std::size_t>(r)];
        }
      }
      std::vector<Quad> residual(rhs_quad.size(), static_cast<Quad>(0.0L));
      Quad residual_inf = static_cast<Quad>(0.0L);
      for (std::size_t j = 0; j < rhs_quad.size(); ++j) {
        residual[j] = rhs_quad[j] - x_quad[j] - at_adx[j];
        residual_inf = std::max(residual_inf, fabsq(residual[j]));
      }
      if (residual_inf <= static_cast<Quad>(1e-35L)) break;
      std::vector<Quad> correction;
      if (!solve_quad_internal(residual, correction)) return false;
      for (std::size_t j = 0; j < x_quad.size(); ++j) x_quad[j] += correction[j];
    }
    x.resize(x_quad.size());
    for (std::size_t i = 0; i < x_quad.size(); ++i) {
      if (!finiteq(x_quad[i])) return false;
      x[i] = static_cast<long double>(x_quad[i]);
    }
    return true;
  }

  long double min_pivot() const { return static_cast<long double>(min_pivot_); }

 private:
  bool solve_quad(const std::vector<Quad>& rhs,
                  std::vector<long double>& x) const {
    std::vector<Quad> solution;
    if (!solve_quad_internal(rhs, solution)) return false;
    x.resize(solution.size());
    for (std::size_t i = 0; i < solution.size(); ++i) {
      if (!finiteq(solution[i])) return false;
      x[i] = static_cast<long double>(solution[i]);
    }
    return true;
  }

  bool solve_quad_internal(const std::vector<Quad>& rhs,
                           std::vector<Quad>& x) const {
    const std::size_t n = ldiag_.size();
    if (rhs.size() != n || n == 0) return false;
    std::vector<Quad> y(rhs);
    for (std::size_t i = 0; i < n; ++i) {
      y[i] /= scale_[i];
      if (i > 0) y[i] -= lsub1_[i] * y[i - 1];
      if (i > 1) y[i] -= lsub2_[i] * y[i - 2];
      y[i] /= ldiag_[i];
      if (!finiteq(y[i])) return false;
    }
    x.assign(n, static_cast<Quad>(0.0L));
    for (std::size_t k = n; k-- > 0;) {
      Quad value = y[k];
      if (k + 1 < n) value -= lsub1_[k + 1] * x[k + 1];
      if (k + 2 < n) value -= lsub2_[k + 2] * x[k + 2];
      x[k] = value / ldiag_[k];
      if (!finiteq(x[k])) return false;
    }
    for (std::size_t i = 0; i < n; ++i) x[i] /= scale_[i];
    return true;
  }

  std::vector<Quad> scale_;
  std::vector<Quad> ldiag_;
  std::vector<Quad> lsub1_;
  std::vector<Quad> lsub2_;
  Quad min_pivot_ = static_cast<Quad>(0.0L);
};
#endif

bool verify_liswet_step_dense(const LiswetStructuredData& data,
                              const std::vector<long double>& s,
                              const std::vector<long double>& z,
                              const std::vector<long double>& rd,
                              const std::vector<long double>& rp,
                              const std::vector<long double>& rc,
                              const std::vector<long double>& dx,
                              const std::vector<long double>& ds,
                              const std::vector<long double>& dz,
                              long double& max_difference) {
  const int n = data.variables;
  const int m = data.rows;
  const int total = n + 2 * m;
  std::vector<long double> k(static_cast<std::size_t>(total) *
                             static_cast<std::size_t>(total), 0.0L);
  std::vector<long double> rhs(static_cast<std::size_t>(total), 0.0L);
  auto at = [&](int row, int col) -> long double& {
    return k[static_cast<std::size_t>(row) * static_cast<std::size_t>(total) +
             static_cast<std::size_t>(col)];
  };
  for (int j = 0; j < n; ++j) {
    at(j, j) = 1.0L;
    rhs[static_cast<std::size_t>(j)] = -rd[static_cast<std::size_t>(j)];
  }
  for (int r = 0; r < m; ++r) {
    const auto& row = data.a[static_cast<std::size_t>(r)];
    rhs[static_cast<std::size_t>(n + r)] = -rp[static_cast<std::size_t>(r)];
    rhs[static_cast<std::size_t>(n + m + r)] = -rc[static_cast<std::size_t>(r)];
    at(n + r, n + r) = -1.0L;
    at(n + m + r, n + r) = z[static_cast<std::size_t>(r)];
    at(n + m + r, n + m + r) = s[static_cast<std::size_t>(r)];
    for (int q = 0; q < 3; ++q) {
      const int col = r + q;
      at(col, n + m + r) -= row[static_cast<std::size_t>(q)];
      at(n + r, col) += row[static_cast<std::size_t>(q)];
    }
  }
  for (int col = 0; col < total; ++col) {
    int pivot = col;
    for (int row = col + 1; row < total; ++row) {
      if (std::abs(at(row, col)) > std::abs(at(pivot, col))) pivot = row;
    }
    if (!(std::abs(at(pivot, col)) > 1e-24L)) return false;
    if (pivot != col) {
      for (int j = col; j < total; ++j) std::swap(at(col, j), at(pivot, j));
      std::swap(rhs[static_cast<std::size_t>(col)],
                rhs[static_cast<std::size_t>(pivot)]);
    }
    for (int row = col + 1; row < total; ++row) {
      const long double scale = at(row, col) / at(col, col);
      if (scale == 0.0L) continue;
      for (int j = col; j < total; ++j) at(row, j) -= scale * at(col, j);
      rhs[static_cast<std::size_t>(row)] -= scale * rhs[static_cast<std::size_t>(col)];
    }
  }
  std::vector<long double> solution(static_cast<std::size_t>(total), 0.0L);
  for (int row = total - 1; row >= 0; --row) {
    long double v = rhs[static_cast<std::size_t>(row)];
    for (int j = row + 1; j < total; ++j) v -= at(row, j) * solution[static_cast<std::size_t>(j)];
    solution[static_cast<std::size_t>(row)] = v / at(row, row);
  }
  max_difference = 0.0L;
  for (int j = 0; j < n; ++j) {
    max_difference = std::max(max_difference,
                              std::abs(solution[static_cast<std::size_t>(j)] -
                                       dx[static_cast<std::size_t>(j)]));
  }
  for (int r = 0; r < m; ++r) {
    max_difference = std::max(max_difference,
      std::abs(solution[static_cast<std::size_t>(n + r)] -
               ds[static_cast<std::size_t>(r)]));
    max_difference = std::max(max_difference,
      std::abs(solution[static_cast<std::size_t>(n + m + r)] -
               dz[static_cast<std::size_t>(r)]));
  }
  return std::isfinite(max_difference);
}

struct LiswetCertificate {
  long double primal = 0.0L;
  long double dual = 0.0L;
  long double stationarity = 0.0L;
  long double complementarity = 0.0L;
  long double consistency = 0.0L;
  long double min_lambda = 0.0L;
  long double lambda_inf = 0.0L;
  bool finite = true;
  bool ok = false;
};

LiswetCertificate certify_liswet(const LiswetStructuredData& data,
                                 const std::vector<long double>& x,
                                 std::vector<long double>& lambda,
                                 std::vector<long double>& slack) {
  LiswetCertificate certificate;
  const int n = data.variables;
  const int m = data.rows;
  std::vector<long double> residual(static_cast<std::size_t>(n), 0.0L);
  for (int j = 0; j < n; ++j) {
    residual[static_cast<std::size_t>(j)] =
        x[static_cast<std::size_t>(j)] - data.target[static_cast<std::size_t>(j)];
  }
  if (!liswet_recover_dual_recurrence(residual, lambda)) {
    certificate.finite = false;
    lambda.assign(static_cast<std::size_t>(m),
                  std::numeric_limits<long double>::quiet_NaN());
  }
  liswet_apply_a(data, x, slack);
  std::vector<long double> atr;
  liswet_apply_at(data, lambda, atr);
  certificate.min_lambda = *std::min_element(lambda.begin(), lambda.end());
  certificate.lambda_inf = 0.0L;
  for (int j = 0; j < n; ++j) {
    certificate.stationarity = std::max(
        certificate.stationarity,
        std::abs(residual[static_cast<std::size_t>(j)] -
                 atr[static_cast<std::size_t>(j)]));
  }
  certificate.primal = 0.0L;
  certificate.complementarity = 0.0L;
  for (int r = 0; r < m; ++r) {
    certificate.primal = std::max(
        certificate.primal, std::max(0.0L, -slack[static_cast<std::size_t>(r)]));
    certificate.complementarity = std::max(
        certificate.complementarity,
        std::abs(lambda[static_cast<std::size_t>(r)] *
                 slack[static_cast<std::size_t>(r)]));
    certificate.lambda_inf = std::max(
        certificate.lambda_inf, std::abs(lambda[static_cast<std::size_t>(r)]));
  }
  if (m > 1) {
    certificate.consistency = std::max(
        std::abs(residual[static_cast<std::size_t>(n - 2)] -
                 (lambda[static_cast<std::size_t>(m - 2)] -
                  2.0L * lambda[static_cast<std::size_t>(m - 1)])),
        std::abs(residual[static_cast<std::size_t>(n - 1)] -
                 lambda[static_cast<std::size_t>(m - 1)]));
  }
  long double scale = 1.0L;
  for (long double v : data.target) scale = std::max(scale, std::abs(v));
  for (long double v : atr) scale = std::max(scale, std::abs(v));
  for (long double v : x) {
    scale = std::max(scale, std::abs(v));
    certificate.finite = certificate.finite && std::isfinite(v);
  }
  for (long double v : lambda) certificate.finite = certificate.finite && std::isfinite(v);
  for (long double v : slack) certificate.finite = certificate.finite && std::isfinite(v);
  certificate.finite = certificate.finite &&
                       std::isfinite(certificate.stationarity) &&
                       std::isfinite(certificate.complementarity);
  const long double tolerance = 1e-8L * scale;
  certificate.ok =
      certificate.finite &&
      certificate.primal <= tolerance &&
      certificate.min_lambda >= -tolerance &&
      certificate.stationarity <= tolerance &&
      certificate.consistency <= tolerance &&
      certificate.complementarity <= tolerance;
  certificate.dual = std::max(0.0L, -certificate.min_lambda);
  return certificate;
}

long double evaluate_liswet_objective(const OptimizationModel& original,
                                      const std::vector<long double>& x) {
  long double value = static_cast<long double>(original.objective.constant);
  std::unordered_map<std::string, int> index;
  index.reserve(original.variables.size() * 2);
  for (std::size_t j = 0; j < original.variables.size(); ++j) {
    index[original.variables[j].name] = static_cast<int>(j);
    const auto linear = original.objective.linear.find(original.variables[j].name);
    if (linear != original.objective.linear.end()) {
      value += static_cast<long double>(linear->second) * x[j];
    }
  }
  for (const auto& row : original.objective.quadratic) {
    const auto ri = index.find(row.first);
    if (ri == index.end()) continue;
    for (const auto& term : row.second) {
      const auto rj = index.find(term.first);
      if (rj != index.end()) {
        value += 0.5L * static_cast<long double>(term.second) *
                 x[static_cast<std::size_t>(ri->second)] *
                 x[static_cast<std::size_t>(rj->second)];
      }
    }
  }
  return value;
}

bool solve_liswet_knot_projection(const LiswetStructuredData& data,
                                  const std::vector<char>& active,
                                  const std::vector<long double>& target,
                                  std::vector<long double>& x) {
  std::vector<int> knots;
  knots.reserve(static_cast<std::size_t>(data.variables));
  knots.push_back(0);
  for (int r = 0; r < data.rows; ++r) {
    if (!active[static_cast<std::size_t>(r)]) knots.push_back(r + 1);
  }
  knots.push_back(data.variables - 1);
  knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
  if (knots.size() < 2) return false;

  const std::size_t k = knots.size();
  std::vector<long double> diagonal(k, 0.0L);
  std::vector<long double> off(k > 1 ? k - 1 : 0, 0.0L);
  std::vector<long double> rhs(k, 0.0L);
  for (std::size_t segment = 0; segment + 1 < k; ++segment) {
    const int left = knots[segment];
    const int right = knots[segment + 1];
    const int length = right - left;
    if (length <= 0) return false;
    const int last = (segment + 2 == k) ? right : right - 1;
    for (int j = left; j <= last; ++j) {
      const long double t = static_cast<long double>(j - left) /
                            static_cast<long double>(length);
      const long double h_left = 1.0L - t;
      const long double h_right = t;
      diagonal[segment] += h_left * h_left;
      rhs[segment] += h_left * target[static_cast<std::size_t>(j)];
      if (h_right != 0.0L) {
        diagonal[segment + 1] += h_right * h_right;
        rhs[segment + 1] += h_right * target[static_cast<std::size_t>(j)];
        off[segment] += h_left * h_right;
      }
    }
  }

  std::vector<long double> ldiag(k, 0.0L);
  std::vector<long double> lsub(k > 1 ? k - 1 : 0, 0.0L);
  for (std::size_t i = 0; i < k; ++i) {
    long double pivot = diagonal[i];
    if (i > 0) {
      lsub[i - 1] = off[i - 1] / ldiag[i - 1];
      pivot -= lsub[i - 1] * lsub[i - 1];
    }
    if (!(pivot > 0.0L) || !std::isfinite(pivot)) return false;
    ldiag[i] = std::sqrt(pivot);
  }
  for (std::size_t i = 0; i < k; ++i) {
    if (i > 0) rhs[i] -= lsub[i - 1] * rhs[i - 1];
    rhs[i] /= ldiag[i];
  }
  for (std::size_t i = k; i-- > 0;) {
    if (i + 1 < k) rhs[i] -= lsub[i] * rhs[i + 1];
    rhs[i] /= ldiag[i];
  }

  x.assign(static_cast<std::size_t>(data.variables), 0.0L);
  for (std::size_t segment = 0; segment + 1 < k; ++segment) {
    const int left = knots[segment];
    const int right = knots[segment + 1];
    const int length = right - left;
    const int last = (segment + 2 == k) ? right : right - 1;
    for (int j = left; j <= last; ++j) {
      const long double t = static_cast<long double>(j - left) /
                            static_cast<long double>(length);
      x[static_cast<std::size_t>(j)] =
          (1.0L - t) * rhs[segment] + t * rhs[segment + 1];
    }
  }
  return true;
}

bool verify_liswet_knot_dense(const LiswetStructuredData& data,
                              const std::vector<char>& active,
                              const std::vector<long double>& target,
                              const std::vector<long double>& x,
                              long double& max_difference) {
  std::vector<int> ids;
  for (int r = 0; r < data.rows; ++r) {
    if (active[static_cast<std::size_t>(r)]) ids.push_back(r);
  }
  const int n = data.variables;
  const int k = static_cast<int>(ids.size());
  const int total = n + k;
  std::vector<long double> matrix(static_cast<std::size_t>(total) *
                                  static_cast<std::size_t>(total), 0.0L);
  std::vector<long double> rhs(static_cast<std::size_t>(total), 0.0L);
  auto at = [&](int row, int col) -> long double& {
    return matrix[static_cast<std::size_t>(row) * static_cast<std::size_t>(total) +
                  static_cast<std::size_t>(col)];
  };
  for (int j = 0; j < n; ++j) {
    at(j, j) = 1.0L;
    rhs[static_cast<std::size_t>(j)] = target[static_cast<std::size_t>(j)];
  }
  for (int q = 0; q < k; ++q) {
    const int r = ids[static_cast<std::size_t>(q)];
    const auto& row = data.a[static_cast<std::size_t>(r)];
    at(n + q, r) = row[0];
    at(n + q, r + 1) = row[1];
    at(n + q, r + 2) = row[2];
    for (int j = 0; j < 3; ++j) at(r + j, n + q) = -row[j];
  }
  for (int col = 0; col < total; ++col) {
    int pivot = col;
    for (int row = col + 1; row < total; ++row) {
      if (std::abs(at(row, col)) > std::abs(at(pivot, col))) pivot = row;
    }
    if (!(std::abs(at(pivot, col)) > 1e-24L)) return false;
    if (pivot != col) {
      for (int j = col; j < total; ++j) std::swap(at(col, j), at(pivot, j));
      std::swap(rhs[static_cast<std::size_t>(col)],
                rhs[static_cast<std::size_t>(pivot)]);
    }
    for (int row = col + 1; row < total; ++row) {
      const long double scale = at(row, col) / at(col, col);
      for (int j = col; j < total; ++j) at(row, j) -= scale * at(col, j);
      rhs[static_cast<std::size_t>(row)] -= scale * rhs[static_cast<std::size_t>(col)];
    }
  }
  std::vector<long double> solution(static_cast<std::size_t>(total), 0.0L);
  for (int row = total - 1; row >= 0; --row) {
    long double value = rhs[static_cast<std::size_t>(row)];
    for (int j = row + 1; j < total; ++j) {
      value -= at(row, j) * solution[static_cast<std::size_t>(j)];
    }
    solution[static_cast<std::size_t>(row)] = value / at(row, row);
  }
  max_difference = 0.0L;
  for (int j = 0; j < n; ++j) {
    max_difference = std::max(
        max_difference,
        std::abs(solution[static_cast<std::size_t>(j)] -
                 x[static_cast<std::size_t>(j)]));
  }
  return std::isfinite(max_difference);
}

bool verify_liswet_gi_step_dense(const LiswetStructuredData& data,
                                 const std::vector<char>& active,
                                 const std::vector<long double>& x,
                                 const std::vector<long double>& lambda,
                                 int p,
                                 const std::vector<long double>& normal,
                                 const std::vector<long double>& z,
                                 const std::vector<long double>& direction,
                                 long double t,
                                 long double& max_difference) {
  std::vector<int> ids;
  for (int r = 0; r < data.rows; ++r) {
    if (active[static_cast<std::size_t>(r)]) ids.push_back(r);
  }
  const int n = data.variables;
  const int k = static_cast<int>(ids.size());
  const int total = n + k;
  std::vector<long double> matrix(
      static_cast<std::size_t>(total) * static_cast<std::size_t>(total), 0.0L);
  std::vector<long double> rhs(static_cast<std::size_t>(total), 0.0L);
  auto at = [&](int row, int col) -> long double& {
    return matrix[static_cast<std::size_t>(row) *
                      static_cast<std::size_t>(total) +
                  static_cast<std::size_t>(col)];
  };
  for (int j = 0; j < n; ++j) {
    at(j, j) = 1.0L;
    rhs[static_cast<std::size_t>(j)] = normal[static_cast<std::size_t>(j)];
  }
  for (int q = 0; q < k; ++q) {
    const int r = ids[static_cast<std::size_t>(q)];
    const auto& row = data.a[static_cast<std::size_t>(r)];
    for (int j = 0; j < 3; ++j) {
      at(r + j, n + q) = -row[static_cast<std::size_t>(j)];
      at(n + q, r + j) = row[static_cast<std::size_t>(j)];
    }
  }
  for (int col = 0; col < total; ++col) {
    int pivot = col;
    for (int row = col + 1; row < total; ++row) {
      if (std::abs(at(row, col)) > std::abs(at(pivot, col))) pivot = row;
    }
    if (!(std::abs(at(pivot, col)) > 1e-24L)) return false;
    if (pivot != col) {
      for (int j = col; j < total; ++j) std::swap(at(col, j), at(pivot, j));
      std::swap(rhs[static_cast<std::size_t>(col)],
                rhs[static_cast<std::size_t>(pivot)]);
    }
    for (int row = col + 1; row < total; ++row) {
      const long double scale = at(row, col) / at(col, col);
      for (int j = col; j < total; ++j) at(row, j) -= scale * at(col, j);
      rhs[static_cast<std::size_t>(row)] -=
          scale * rhs[static_cast<std::size_t>(col)];
    }
  }
  std::vector<long double> solution(static_cast<std::size_t>(total), 0.0L);
  for (int row = total - 1; row >= 0; --row) {
    long double value = rhs[static_cast<std::size_t>(row)];
    for (int j = row + 1; j < total; ++j) {
      value -= at(row, j) * solution[static_cast<std::size_t>(j)];
    }
    solution[static_cast<std::size_t>(row)] = value / at(row, row);
  }

  max_difference = 0.0L;
  for (int j = 0; j < n; ++j) {
    max_difference = std::max(
        max_difference,
        std::abs(solution[static_cast<std::size_t>(j)] -
                 z[static_cast<std::size_t>(j)]));
  }
  std::vector<long double> dense_direction(static_cast<std::size_t>(data.rows),
                                           0.0L);
  for (int q = 0; q < k; ++q) {
    const int r = ids[static_cast<std::size_t>(q)];
    dense_direction[static_cast<std::size_t>(r)] =
        -solution[static_cast<std::size_t>(n + q)];
    max_difference = std::max(
        max_difference,
        std::abs(dense_direction[static_cast<std::size_t>(r)] -
                 direction[static_cast<std::size_t>(r)]));
  }

  const auto& prow = data.a[static_cast<std::size_t>(p)];
  const long double sp =
      prow[0] * x[static_cast<std::size_t>(p)] +
      prow[1] * x[static_cast<std::size_t>(p + 1)] +
      prow[2] * x[static_cast<std::size_t>(p + 2)];
  const long double apz =
      prow[0] * solution[static_cast<std::size_t>(p)] +
      prow[1] * solution[static_cast<std::size_t>(p + 1)] +
      prow[2] * solution[static_cast<std::size_t>(p + 2)];
  const long double dense_full = -sp / apz;
  long double dense_drop = std::numeric_limits<long double>::infinity();
  for (int q = 0; q < k; ++q) {
    const int r = ids[static_cast<std::size_t>(q)];
    const long double rr = dense_direction[static_cast<std::size_t>(r)];
    if (rr > 0.0L) {
      dense_drop = std::min(
          dense_drop, lambda[static_cast<std::size_t>(r)] / rr);
    }
  }
  const long double dense_t = std::min(dense_full, dense_drop);
  max_difference = std::max(max_difference, std::abs(dense_t - t));
  return std::isfinite(max_difference);
}

bool solve_liswet_structured_gi(const OptimizationModel& original,
                                const QpInteriorPointOptions& opt,
                                SolverResult& result) {
  (void)opt;
  LiswetStructuredData data;
  if (!detect_liswet_structured(original, data)) return false;

  const int n = data.variables;
  const int m = data.rows;
  const long double c_scale = std::max(
      1.0L, *std::max_element(data.target.begin(), data.target.end(),
                               [](long double a, long double b) {
                                 return std::abs(a) < std::abs(b);
                               }));
  const long double tol = 1e-12L * c_scale;
  const int iteration_cap = std::max(1, 20 * n);
  std::vector<char> active(static_cast<std::size_t>(m), 0);
  std::vector<long double> x = data.target;
  std::vector<long double> lambda(static_cast<std::size_t>(m), 0.0L);
  std::vector<long double> slack;
  int iterations = 0;
  int drops = 0;
  int refreshes = 0;
  int guard_trips = 0;
  int guard_retry_iteration = -1;
  int retry_p = -1;
  long double max_drift = 0.0L;
  long double last_recompute_drift = 0.0L;
  long double max_support_residual = 0.0L;
  long double max_guard_ratio = 0.0L;
  long double max_step_difference = 0.0L;
  bool step_verified = false;
  std::string failure;
  const bool profile =
      std::getenv("SOVEREIGN_QP_GI_PROFILE") != nullptr &&
      std::string(std::getenv("SOVEREIGN_QP_GI_PROFILE")) == "1";
  const auto profile_start = std::chrono::steady_clock::now();
  long long direction_projection_calls = 0;
  long long refresh_projection_calls = 0;
  double direction_projection_seconds = 0.0;
  double refresh_projection_seconds = 0.0;
  double recurrence_seconds = 0.0;
  double recompute_seconds = 0.0;
  struct GuardSample {
    int iteration = 0;
    long double value = 0.0L;
    long double threshold = 0.0L;
    long double roundoff_ratio = 0.0L;
    long double scaled_ratio = 0.0L;
    long double r_inf = 0.0L;
    long double lambda_inf = 0.0L;
  };
  struct TrendSample {
    int iteration = 0;
    int violated = 0;
    long double min_slack = 0.0L;
  };
  std::vector<GuardSample> guard_history;
  std::vector<TrendSample> trend_history;

  auto timed_projection = [&](const std::vector<char>& active_set,
                              const std::vector<long double>& rhs,
                              std::vector<long double>& answer,
                              bool refresh) {
    const auto start = std::chrono::steady_clock::now();
    const bool ok =
        solve_liswet_knot_projection(data, active_set, rhs, answer);
    if (profile) {
      const double seconds =
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - start)
              .count();
      if (refresh) {
        ++refresh_projection_calls;
        refresh_projection_seconds += seconds;
      } else {
        ++direction_projection_calls;
        direction_projection_seconds += seconds;
      }
    }
    return ok;
  };

  auto recompute_state = [&]() {
    const auto recompute_start = std::chrono::steady_clock::now();
    std::vector<long double> fresh_x;
    if (!timed_projection(active, data.target, fresh_x, true)) {
      return false;
    }
    long double drift = 0.0L;
    for (int j = 0; j < n; ++j) {
      drift = std::max(
          drift, std::abs(fresh_x[static_cast<std::size_t>(j)] -
                          x[static_cast<std::size_t>(j)]));
    }
    std::vector<long double> fresh_lambda, fresh_slack;
    certify_liswet(data, fresh_x, fresh_lambda, fresh_slack);
    for (int r = 0; r < m; ++r) {
      drift = std::max(
          drift, std::abs(fresh_lambda[static_cast<std::size_t>(r)] -
                          lambda[static_cast<std::size_t>(r)]));
    }
    max_drift = std::max(max_drift, drift);
    last_recompute_drift = drift;
    x.swap(fresh_x);
    lambda.swap(fresh_lambda);
    slack.swap(fresh_slack);
    if (profile) {
      recompute_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - recompute_start)
              .count();
    }
    return true;
  };

  while (iterations < iteration_cap) {
    liswet_apply_a(data, x, slack);
    int p = -1;
    long double min_slack = std::numeric_limits<long double>::infinity();
    if (retry_p >= 0) {
      p = retry_p;
      min_slack = slack[static_cast<std::size_t>(p)];
      if (min_slack >= -tol) retry_p = -1;
    }
    if (retry_p < 0) {
      p = 0;
      min_slack = slack[0];
      for (int r = 1; r < m; ++r) {
        if (slack[static_cast<std::size_t>(r)] < min_slack) {
          min_slack = slack[static_cast<std::size_t>(r)];
          p = r;
        }
      }
    }
    if (retry_p < 0 && min_slack >= -tol) break;
    if (retry_p >= 0 && min_slack >= -tol) {
      retry_p = -1;
      continue;
    }
    if (p < 0 || active[static_cast<std::size_t>(p)]) {
      failure = "GI selected an already-active violated constraint";
      break;
    }

    std::vector<long double> normal(static_cast<std::size_t>(n), 0.0L);
    const auto& prow = data.a[static_cast<std::size_t>(p)];
    normal[static_cast<std::size_t>(p)] = prow[0];
    normal[static_cast<std::size_t>(p + 1)] = prow[1];
    normal[static_cast<std::size_t>(p + 2)] = prow[2];
    std::vector<long double> z;
    if (!timed_projection(active, normal, z, false)) {
      failure = "GI knot-space direction solve failed";
      break;
    }
    long double norm_z = 0.0L;
    for (long double value : z) norm_z += value * value;
    const long double apz =
        prow[0] * z[static_cast<std::size_t>(p)] +
        prow[1] * z[static_cast<std::size_t>(p + 1)] +
        prow[2] * z[static_cast<std::size_t>(p + 2)];
    if (!(norm_z > 0.0L) || !(apz > 0.0L) ||
        !std::isfinite(norm_z) || !std::isfinite(apz)) {
      failure = "GI direction has non-positive norm or constraint derivative";
      break;
    }

    std::vector<long double> residual(static_cast<std::size_t>(n), 0.0L);
    for (int j = 0; j < n; ++j) {
      residual[static_cast<std::size_t>(j)] =
          normal[static_cast<std::size_t>(j)] -
          z[static_cast<std::size_t>(j)];
    }
    std::vector<long double> full_direction;
    const auto recurrence_start = std::chrono::steady_clock::now();
    const bool recurrence_ok =
        liswet_recover_dual_recurrence(residual, full_direction);
    if (profile) {
      recurrence_seconds +=
          std::chrono::duration<double>(
              std::chrono::steady_clock::now() - recurrence_start)
              .count();
    }
    if (!recurrence_ok) {
      failure = "GI dual direction recurrence failed";
      break;
    }
    std::vector<long double> supported_direction(full_direction);
    long double r_inf = 0.0L;
    for (long double value : full_direction) {
      r_inf = std::max(r_inf, std::abs(value));
    }
    long double lambda_inf = 0.0L;
    for (long double value : lambda) {
      lambda_inf = std::max(lambda_inf, std::abs(value));
    }
    long double normal_inf = 0.0L;
    long double normal_norm_sq = 0.0L;
    for (long double value : normal) {
      normal_inf = std::max(normal_inf, std::abs(value));
      normal_norm_sq += value * value;
    }
    long double support_residual = 0.0L;
    for (int r = 0; r < m; ++r) {
      if (!active[static_cast<std::size_t>(r)]) {
        support_residual = std::max(
            support_residual,
            std::abs(supported_direction[static_cast<std::size_t>(r)]));
        supported_direction[static_cast<std::size_t>(r)] = 0.0L;
      }
    }
    std::vector<long double> reconstructed;
    liswet_apply_at(data, supported_direction, reconstructed);
    long double equation_residual = 0.0L;
    for (int j = 0; j < n; ++j) {
      equation_residual = std::max(
          equation_residual,
          std::abs(reconstructed[static_cast<std::size_t>(j)] -
                   residual[static_cast<std::size_t>(j)]));
    }
    max_support_residual = std::max(max_support_residual, support_residual);
    const long double direction_scale =
        std::max(1.0L, std::max(norm_z, c_scale));
    const long double guard_value =
        std::max(support_residual, equation_residual);
    // The dual recurrence is a discrete double cumulative sum.  Use n^2 as
    // the effective operation count, with a modest safety factor, while
    // leaving the independent final certifier unchanged.
    constexpr long double guard_constant = 16.0L;
    const long double n_ld = static_cast<long double>(n);
    const long double n_eff = std::max(1.0L, n_ld * n_ld);
    const long double guard_state_scale =
        std::max(1.0L, std::max(r_inf, lambda_inf));
    const long double guard_threshold =
        guard_constant * std::numeric_limits<long double>::epsilon() *
        n_eff * guard_state_scale;
    const long double roundoff_denominator =
        std::numeric_limits<long double>::epsilon() *
        n_ld * guard_state_scale;
    const long double scaled_guard_ratio =
        guard_threshold > 0.0L ? guard_value / guard_threshold : 0.0L;
    const long double normal_norm = std::sqrt(normal_norm_sq);
    max_guard_ratio = std::max(max_guard_ratio, scaled_guard_ratio);
    guard_history.push_back(
        GuardSample{iterations + 1, guard_value, guard_threshold,
                    roundoff_denominator > 0.0L
                        ? guard_value / roundoff_denominator
                        : 0.0L,
                    scaled_guard_ratio,
                    r_inf, lambda_inf});
    if (guard_history.size() > 20) guard_history.erase(guard_history.begin());
    if (guard_value > guard_threshold) {
      const int guard_iteration = iterations + 1;
      ++guard_trips;
      const bool already_retried =
          guard_retry_iteration == guard_iteration;
      if (std::getenv("SOVEREIGN_QP_TRACE")) {
        std::cerr << std::setprecision(18)
                  << "[qp-structured-gi-guard] trip"
                  << " iteration=" << guard_iteration
                  << " guard_value=" << static_cast<double>(guard_value)
                  << " guard_tol=" << static_cast<double>(guard_threshold)
                  << " retry_already="
                  << (already_retried ? 1 : 0) << "\n";
      }
      if (!already_retried) {
        guard_retry_iteration = guard_iteration;
        if (recompute_state()) {
          ++refreshes;
          if (std::getenv("SOVEREIGN_QP_TRACE")) {
            std::cerr << "[qp-structured-gi-guard] refresh"
                      << " iteration=" << guard_iteration
                      << " refreshes=" << refreshes << "\n";
          }
          continue;
        }
      }
      if (std::getenv("SOVEREIGN_QP_TRACE")) {
        int violated = 0;
        for (long double value : slack) {
          if (value < -tol) ++violated;
        }
        std::cerr << std::setprecision(18)
                  << "[qp-structured-gi-guard] abort"
                  << " iteration=" << (iterations + 1)
                  << " guard_value=" << static_cast<double>(guard_value)
                  << " guard_threshold=" << static_cast<double>(guard_threshold)
                  << " threshold_C=" << static_cast<double>(guard_constant)
                  << " n_eff=" << static_cast<double>(n_eff)
                  << " threshold_state_scale="
                  << static_cast<double>(guard_state_scale)
                  << " threshold_formula=C*eps_ld*n_eff*max(1,r_inf,lambda_inf)"
                  << " r_inf=" << static_cast<double>(r_inf)
                  << " lambda_inf=" << static_cast<double>(lambda_inf)
                  << " np_norm=" << static_cast<double>(normal_norm)
                  << " np_inf=" << static_cast<double>(normal_inf)
                  << " support_residual="
                  << static_cast<double>(support_residual)
                  << " equation_residual="
                  << static_cast<double>(equation_residual)
                  << " W=" << std::count(active.begin(), active.end(),
                                           static_cast<char>(1))
                  << " violated=" << violated
                  << " min_s=" << static_cast<double>(min_slack)
                  << " last_recompute_drift="
                  << static_cast<double>(last_recompute_drift)
                  << " max_drift=" << static_cast<double>(max_drift) << "\n";
        std::cerr << "[qp-structured-gi-guard] last20_roundoff_ratios\n";
        for (const GuardSample& sample : guard_history) {
          std::cerr << "  iteration=" << sample.iteration
                    << " value=" << static_cast<double>(sample.value)
                    << " threshold=" << static_cast<double>(sample.threshold)
                    << " value_over_eps_n_scale="
                    << static_cast<double>(sample.roundoff_ratio)
                    << " value_over_guard_tol="
                    << static_cast<double>(sample.scaled_ratio)
                    << " r_inf=" << static_cast<double>(sample.r_inf)
                    << " lambda_inf=" << static_cast<double>(sample.lambda_inf)
                    << "\n";
        }
        std::cerr << "[qp-structured-gi-guard] trend_500\n";
        for (const TrendSample& sample : trend_history) {
          std::cerr << "  iteration=" << sample.iteration
                    << " violated=" << sample.violated
                    << " min_s=" << static_cast<double>(sample.min_slack)
                    << "\n";
        }
      }
      failure = "GI dual direction support/equation residual too large";
      break;
    }

    const long double t_full = -min_slack / apz;
    long double t_drop = std::numeric_limits<long double>::infinity();
    int drop = -1;
    for (int r = 0; r < m; ++r) {
      if (!active[static_cast<std::size_t>(r)]) continue;
      const long double rr = supported_direction[static_cast<std::size_t>(r)];
      if (rr <= 0.0L) continue;
      const long double ratio =
          lambda[static_cast<std::size_t>(r)] / rr;
      if (ratio < t_drop ||
          (ratio == t_drop && (drop < 0 || r < drop))) {
        t_drop = ratio;
        drop = r;
      }
    }
    const long double t = std::min(t_full, t_drop);
    if (!(t >= 0.0L) || !std::isfinite(t)) {
      failure = "GI produced an invalid step length";
      break;
    }

    if (!step_verified && data.rows <= 50 &&
        std::count(active.begin(), active.end(), static_cast<char>(1)) > 0) {
      long double difference = 0.0L;
      if (!verify_liswet_gi_step_dense(
              data, active, x, lambda, p, normal, z, supported_direction, t,
              difference) ||
          difference > 1e-10L) {
        failure = "dense GI step verification failed";
        break;
      }
      max_step_difference = difference;
      step_verified = true;
    }

    for (int r = 0; r < m; ++r) {
      if (active[static_cast<std::size_t>(r)]) {
        lambda[static_cast<std::size_t>(r)] -=
            t * supported_direction[static_cast<std::size_t>(r)];
      }
    }
    for (int j = 0; j < n; ++j) {
      x[static_cast<std::size_t>(j)] +=
          t * z[static_cast<std::size_t>(j)];
    }
    ++iterations;

    const bool drop_step = drop >= 0 && t_drop <= t_full;
    if (drop_step) {
      active[static_cast<std::size_t>(drop)] = 0;
      lambda[static_cast<std::size_t>(drop)] = 0.0L;
      ++drops;
      retry_p = p;
      if (!recompute_state()) {
        failure = "GI state recomputation after drop failed";
        break;
      }
    } else {
      active[static_cast<std::size_t>(p)] = 1;
      lambda[static_cast<std::size_t>(p)] += t;
      retry_p = -1;
    }
    if (!drop_step && iterations % 50 == 0) {
      if (!recompute_state()) {
        failure = "GI periodic state recomputation failed";
        break;
      }
    }

    if (std::getenv("SOVEREIGN_QP_TRACE") &&
        (n <= 1000 || iterations % 100 == 0 || drop_step)) {
      const long double max_violation = std::max(0.0L, -min_slack);
      std::cerr << std::setprecision(12)
                << "[qp-structured-gi] it=" << iterations
                << " active="
                << std::count(active.begin(), active.end(), static_cast<char>(1))
                << " p=" << p
                << " min_s=" << static_cast<double>(min_slack)
                << " max_violation=" << static_cast<double>(max_violation)
                << " t=" << static_cast<double>(t)
                << " t_full=" << static_cast<double>(t_full)
                << " t_drop=" << static_cast<double>(t_drop)
                << " drop=" << (drop_step ? drop : -1)
                << " support_residual=" << static_cast<double>(support_residual)
                << " drift=" << static_cast<double>(max_drift) << "\n";
    }
    if (std::getenv("SOVEREIGN_QP_TRACE") && iterations % 500 == 0) {
      int violated = 0;
      for (long double value : slack) {
        if (value < -tol) ++violated;
      }
      trend_history.push_back(
          TrendSample{iterations, violated, min_slack});
      std::cerr << "[qp-structured-gi-trend] iteration=" << iterations
                << " violated=" << violated
                << " min_s=" << static_cast<double>(min_slack) << "\n";
    }
  }
  if (iterations >= iteration_cap && failure.empty()) {
    failure = "GI iteration cap reached";
  }

  std::vector<long double> certified_lambda, certified_slack;
  const LiswetCertificate certificate =
      certify_liswet(data, x, certified_lambda, certified_slack);
  for (int j = 0; j < n; ++j) {
    result.primal[original.variables[static_cast<std::size_t>(j)].name] =
        static_cast<double>(x[static_cast<std::size_t>(j)]);
  }
  for (int r = 0; r < m; ++r) {
    result.dual["row_" + std::to_string(r)] =
        static_cast<double>(certified_lambda[static_cast<std::size_t>(r)]);
    result.slacks["row_" + std::to_string(r)] =
        static_cast<double>(certified_slack[static_cast<std::size_t>(r)]);
  }
  const long double objective = evaluate_liswet_objective(original, x);
  result.iterations = iterations;
  result.primal_residual = static_cast<double>(certificate.primal);
  result.dual_residual = static_cast<double>(certificate.stationarity);
  result.optimality_gap = static_cast<double>(certificate.complementarity);
  result.objective_value = static_cast<double>(objective);
  result.has_objective_value = certificate.ok && failure.empty();
  result.status = certificate.ok && failure.empty() ? SolverStatus::Optimal
                                                     : SolverStatus::Error;
  std::ostringstream message;
  message << (certificate.ok && failure.empty()
                  ? "Optimal"
                  : "Structured LISWET Goldfarb-Idnani failed certification")
          << "; certifier primal=" << std::setprecision(18)
          << static_cast<double>(certificate.primal)
          << " dual=" << static_cast<double>(certificate.dual)
          << " stationarity=" << static_cast<double>(certificate.stationarity)
          << " complementarity=" << static_cast<double>(certificate.complementarity)
          << " lambda_inf=" << static_cast<double>(certificate.lambda_inf)
          << " consistency=" << static_cast<double>(certificate.consistency)
          << " min_lambda=" << static_cast<double>(certificate.min_lambda)
          << "; gi_iterations=" << iterations
          << "/" << iteration_cap
          << "; drops=" << drops
          << "; refreshes=" << refreshes
          << "; guard_trips=" << guard_trips
          << "; max_guard_value_over_guard_tol="
          << static_cast<double>(max_guard_ratio)
          << "; guard_n_eff=n^2"
          << "; guard_C=16"
          << "; active=" << std::count(active.begin(), active.end(),
                                       static_cast<char>(1))
          << "; max_drift=" << static_cast<double>(max_drift)
          << "; max_support_residual="
          << static_cast<double>(max_support_residual)
          << "; max_step_difference="
          << static_cast<double>(max_step_difference)
          << "; step_verified=" << (step_verified ? 1 : 0)
          << "; objective=" << static_cast<double>(objective);
  if (!failure.empty()) message << "; trace_failure=" << failure;
  result.message = message.str();
  if (profile) {
    const double total_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now() - profile_start)
            .count();
    const double allocation_other_seconds = std::max(
        0.0, total_seconds - direction_projection_seconds -
                  recurrence_seconds - recompute_seconds);
    std::cerr << std::setprecision(12)
              << "[qp-structured-gi-profile] n=" << n
              << " total_seconds=" << total_seconds
              << " direction_projection_calls=" << direction_projection_calls
              << " direction_projection_seconds="
              << direction_projection_seconds
              << " refresh_projection_calls=" << refresh_projection_calls
              << " refresh_projection_seconds=" << refresh_projection_seconds
              << " recurrence_seconds=" << recurrence_seconds
              << " recompute_seconds=" << recompute_seconds
              << " allocation_other_proxy_seconds="
              << allocation_other_seconds << "\n";
  }
  return true;
}

bool polish_liswet_knots(const LiswetStructuredData& data,
                         const std::vector<long double>& target,
                         const std::vector<long double>& slack,
                         const std::vector<long double>& z,
                         std::vector<long double>& x,
                         LiswetCertificate& certificate,
                         int& passes) {
  std::vector<char> active(static_cast<std::size_t>(data.rows), 0);
  for (int r = 0; r < data.rows; ++r) {
    active[static_cast<std::size_t>(r)] =
        slack[static_cast<std::size_t>(r)] < z[static_cast<std::size_t>(r)];
  }
  for (passes = 0; passes < 50; ++passes) {
    std::vector<long double> candidate;
    if (!solve_liswet_knot_projection(data, active, target, candidate)) return false;
    if (data.rows <= 50) {
      long double difference = 0.0L;
      if (!verify_liswet_knot_dense(data, active, target, candidate, difference) ||
          difference > 1e-10L) {
        return false;
      }
    }
    std::vector<long double> lambda, candidate_slack;
    certificate = certify_liswet(data, candidate, lambda, candidate_slack);
    long double polish_scale = 1.0L;
    for (long double value : target) polish_scale = std::max(polish_scale, std::abs(value));
    for (long double value : candidate) polish_scale = std::max(polish_scale, std::abs(value));
    const long double update_tolerance = 1e-8L * polish_scale;
    int violations = 0;
    int negative_multipliers = 0;
    for (int r = 0; r < data.rows; ++r) {
      if (candidate_slack[static_cast<std::size_t>(r)] < -update_tolerance) ++violations;
      if (active[static_cast<std::size_t>(r)] &&
          lambda[static_cast<std::size_t>(r)] < -update_tolerance) {
        ++negative_multipliers;
      }
    }
    if (std::getenv("SOVEREIGN_QP_TRACE")) {
      int active_count = static_cast<int>(
          std::count(active.begin(), active.end(), static_cast<char>(1)));
      std::cerr << "[qp-structured] phase=polish pass=" << passes
                << " active=" << active_count
                << " violations=" << violations
                << " negative_multipliers=" << negative_multipliers << "\n";
    }
    if (certificate.ok) {
      x.swap(candidate);
      return true;
    }

    std::vector<char> next_active = active;
    int direct_changes = 0;
    int direct_complementarity_additions = 0;
    std::vector<std::pair<long double, int>> evidence_changes;
    for (int r = 0; r < data.rows; ++r) {
      const std::size_t rr = static_cast<std::size_t>(r);
      if (active[rr] && lambda[rr] < -update_tolerance) {
        next_active[rr] = 0;
        evidence_changes.emplace_back(
            std::abs(lambda[rr]), r);
      } else if (!active[rr] &&
                 (candidate_slack[rr] < -update_tolerance ||
                  (lambda[rr] > update_tolerance &&
                   lambda[rr] * candidate_slack[rr] > update_tolerance))) {
        next_active[rr] = 1;
        if (lambda[rr] * candidate_slack[rr] > update_tolerance) {
          ++direct_complementarity_additions;
        }
        evidence_changes.emplace_back(
            std::max(-candidate_slack[rr],
                     lambda[rr] * candidate_slack[rr]), r);
      }
      if (next_active[rr] != active[rr]) ++direct_changes;
    }
    const int max_changes_per_pass =
        std::min(1024, std::max(8, data.rows / 10));
    if (direct_changes > max_changes_per_pass) {
      std::sort(evidence_changes.begin(), evidence_changes.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      next_active = active;
      direct_complementarity_additions = 0;
      direct_changes = 0;
      for (int q = 0;
           q < max_changes_per_pass &&
           q < static_cast<int>(evidence_changes.size()); ++q) {
        const int r = evidence_changes[static_cast<std::size_t>(q)].second;
        const std::size_t rr = static_cast<std::size_t>(r);
        next_active[rr] = active[rr] ? 0 : 1;
        if (!active[rr] &&
            lambda[rr] * candidate_slack[rr] > update_tolerance) {
          ++direct_complementarity_additions;
        }
        ++direct_changes;
      }
    }
    if (std::getenv("SOVEREIGN_QP_TRACE")) {
      std::cerr << "[qp-structured] phase=polish_update pass=" << passes
                << " changes=" << direct_changes
                << " complementarity_additions="
                << direct_complementarity_additions
                << " limit=" << max_changes_per_pass << "\n";
    }
    if (direct_changes == 0) return false;
    active.swap(next_active);
    continue;
  }
  return false;
}

bool solve_liswet_structured_ipm(const OptimizationModel& original,
                                  const QpInteriorPointOptions& opt,
                                  SolverResult& result) {
  LiswetStructuredData data;
  if (!detect_liswet_structured(original, data)) return false;

  const int n = data.variables;
  const int m = data.rows;
  const int iteration_cap = std::max(1, opt.max_iterations);
  // The public QP defaults are appropriate for generic models, but the
  // structured acceptance contract compares the projection itself.  Use a
  // stricter internal stopping test; this only tightens, never loosens, the
  // caller's requested tolerances.
  const long double structured_target_tol =
      data.rows <= 50 ? 1e-12L : 1e-10L;
  const long double structured_feasibility_tol =
      std::min(static_cast<long double>(opt.feasibility_tol), structured_target_tol);
  const long double structured_optimality_tol =
      std::min(static_cast<long double>(opt.optimality_tol), structured_target_tol);
  const long double tau = 0.995L;
  std::vector<long double> x = data.target;
  std::vector<long double> slack;
  liswet_apply_a(data, x, slack);
  for (long double& value : slack) value = std::max(1.0L, value);
  std::vector<long double> z(static_cast<std::size_t>(m), 1.0L);
  for (int r = 0; r < m; ++r) z[static_cast<std::size_t>(r)] =
      1.0L / slack[static_cast<std::size_t>(r)];

  long double last_pivot = 0.0L;
  long double last_regularization = 0.0L;
  long double last_regularization_ratio = 0.0L;
  int regularization_count = 0;
  int completed_iterations = 0;
  bool converged = false;
  bool polish_requested = false;
  std::string failure;
#if defined(SOVEREIGN_HAS_FLOAT128)
  const bool high_precision =
      std::getenv("SOVEREIGN_QP_HIGH_PRECISION") != nullptr &&
      std::string(std::getenv("SOVEREIGN_QP_HIGH_PRECISION")) == "1";
#else
  const bool high_precision = false;
#endif
  auto max_abs_ld = [](const std::vector<long double>& values) {
    long double answer = 0.0L;
    for (long double value : values) answer = std::max(answer, std::abs(value));
    return answer;
  };
  auto positive_step = [](const std::vector<long double>& values,
                          const std::vector<long double>& direction) {
    long double answer = 1.0L;
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (direction[i] < 0.0L) answer = std::min(answer, -values[i] / direction[i]);
    }
    return std::isfinite(answer) ? std::max(0.0L, answer) : 0.0L;
  };

  for (int iteration = 0; iteration < iteration_cap; ++iteration) {
    std::vector<long double> ax, atz;
    liswet_apply_a(data, x, ax);
    liswet_apply_at(data, z, atz);
    std::vector<long double> rp(static_cast<std::size_t>(m), 0.0L);
    std::vector<long double> rd(static_cast<std::size_t>(n), 0.0L);
    for (int r = 0; r < m; ++r) {
      rp[static_cast<std::size_t>(r)] = ax[static_cast<std::size_t>(r)] -
                                         slack[static_cast<std::size_t>(r)];
    }
    for (int j = 0; j < n; ++j) {
      rd[static_cast<std::size_t>(j)] = x[static_cast<std::size_t>(j)] -
          data.target[static_cast<std::size_t>(j)] - atz[static_cast<std::size_t>(j)];
    }
    const long double mu = [&]() {
      long double value = 0.0L;
      for (int r = 0; r < m; ++r) value += slack[static_cast<std::size_t>(r)] *
                                              z[static_cast<std::size_t>(r)];
      return value / static_cast<long double>(m);
    }();
    const long double p_res = max_abs_ld(rp) / std::max(1.0L, max_abs_ld(ax));
    const long double d_res = max_abs_ld(rd) /
        std::max(1.0L, max_abs_ld(data.target) + max_abs_ld(atz));
    long double max_complementarity = 0.0L;
    for (int r = 0; r < m; ++r) {
      max_complementarity = std::max(
          max_complementarity,
          std::abs(slack[static_cast<std::size_t>(r)] *
                   z[static_cast<std::size_t>(r)]));
    }
    const long double comp_res = max_complementarity /
        std::max(1.0L, max_abs_ld(data.target));
    const bool finite_iter = std::isfinite(mu) && std::isfinite(p_res) &&
                             std::isfinite(d_res) && std::isfinite(comp_res);
    if (!finite_iter) {
      failure = "non-finite residual, mu, or iterate";
      break;
    }
    if (!high_precision && comp_res <= 1e-6L) {
      polish_requested = true;
      completed_iterations = iteration;
      break;
    }
    if (p_res <= structured_feasibility_tol &&
        d_res <= structured_feasibility_tol &&
        comp_res <= structured_optimality_tol) {
      converged = true;
      completed_iterations = iteration;
      if (std::getenv("SOVEREIGN_QP_TRACE")) {
        std::cerr << std::setprecision(12)
                  << "[qp-structured] it=" << iteration << " mu=" << static_cast<double>(mu)
                  << " p_res=" << static_cast<double>(p_res)
                  << " d_res=" << static_cast<double>(d_res)
                  << " comp_res=" << static_cast<double>(comp_res)
                  << " max_comp=" << static_cast<double>(max_complementarity)
                  << " alpha_aff_s=NA alpha_aff_z=NA alpha_s=NA alpha_z=NA"
                  << " min_pivot=" << static_cast<double>(last_pivot)
                  << " regularization=" << static_cast<double>(last_regularization)
                  << " regularizations=" << regularization_count
                  << " cap=" << iteration_cap << " nan_inf=0"
                  << " kkt=x-space pentadiagonal Cholesky\n";
      }
      break;
    }

    std::vector<long double> diagonal(static_cast<std::size_t>(n), 1.0L);
    std::vector<long double> sub1(static_cast<std::size_t>(n), 0.0L);
    std::vector<long double> sub2(static_cast<std::size_t>(n), 0.0L);
    std::vector<long double> d(static_cast<std::size_t>(m), 0.0L);
    long double min_d = std::numeric_limits<long double>::max();
    long double max_d = 0.0L;
    int constraints_s_lt_z = 0;
    for (int r = 0; r < m; ++r) {
      d[static_cast<std::size_t>(r)] =
          z[static_cast<std::size_t>(r)] / slack[static_cast<std::size_t>(r)];
      min_d = std::min(min_d, d[static_cast<std::size_t>(r)]);
      max_d = std::max(max_d, d[static_cast<std::size_t>(r)]);
      if (slack[static_cast<std::size_t>(r)] < z[static_cast<std::size_t>(r)]) {
        ++constraints_s_lt_z;
      }
      const auto& row = data.a[static_cast<std::size_t>(r)];
      for (int p = 0; p < 3; ++p) {
        const int i = r + p;
        for (int q = 0; q < 3; ++q) {
          const int j = r + q;
          const long double value = d[static_cast<std::size_t>(r)] *
                                    row[static_cast<std::size_t>(p)] *
                                    row[static_cast<std::size_t>(q)];
          if (i == j) diagonal[static_cast<std::size_t>(i)] += value;
          else if (i == j + 1) sub1[static_cast<std::size_t>(i)] += value;
          else if (i == j + 2) sub2[static_cast<std::size_t>(i)] += value;
        }
      }
    }
    const long double max_diag_m =
        *std::max_element(diagonal.begin(), diagonal.end());
    LiswetPentadiagonalCholesky factor;
#if defined(SOVEREIGN_HAS_FLOAT128)
    LiswetQuadPentadiagonalCholesky quad_factor;
#endif
    bool regularization_cap_hit = false;
    bool factor_ok = false;
#if defined(SOVEREIGN_HAS_FLOAT128)
    if (high_precision) {
      last_regularization = 0.0L;
      factor_ok = quad_factor.factor(diagonal, sub1, sub2);
    } else
#endif
    {
      factor_ok = factor.factor(diagonal, sub1, sub2, last_regularization,
                                regularization_cap_hit);
    }
    const long double factor_min_pivot =
#if defined(SOVEREIGN_HAS_FLOAT128)
        high_precision ? quad_factor.min_pivot() :
#endif
        factor.min_pivot();
    const long double regularization_cap =
#if defined(SOVEREIGN_HAS_FLOAT128)
        high_precision ? 0.0L :
#endif
        factor.regularization_cap();
    if (!factor_ok) {
      last_pivot = factor_min_pivot;
      last_regularization_ratio = last_regularization;
      if (regularization_cap_hit) {
        failure = "regularization cap hit before factorization could continue";
      } else if (high_precision) {
        failure = "float128 pentadiagonal Cholesky encountered a non-positive pivot";
      } else {
        failure = "pentadiagonal Cholesky factorization failed";
      }
      if (std::getenv("SOVEREIGN_QP_TRACE")) {
        std::cerr << std::setprecision(12)
                  << "[qp-structured] it=" << iteration
                  << " mu=" << static_cast<double>(mu)
                  << " lambda_inf=" << static_cast<double>(max_abs_ld(z))
                  << " max_D=" << static_cast<double>(max_d)
                  << " min_D=" << static_cast<double>(min_d)
                  << " max_diag_M=" << static_cast<double>(max_diag_m)
                  << " min_pivot=" << static_cast<double>(last_pivot)
                  << " regularization=" << static_cast<double>(last_regularization)
                  << " regularization_ratio=" << static_cast<double>(last_regularization_ratio)
                  << " regularization_cap=" << static_cast<double>(regularization_cap)
                  << " regularization_cap_hit=" << (regularization_cap_hit ? 1 : 0)
                  << " constraints_s_lt_z=" << constraints_s_lt_z
                  << " cap=" << iteration_cap << " nan_inf=0 phase=factorization_failed"
                  << " high_precision=" << (high_precision ? 1 : 0) << "\n";
      }
      break;
    }
    if (last_regularization > 0.0L) ++regularization_count;
    last_regularization_ratio = last_regularization;
    last_pivot = factor_min_pivot;
    auto solve_direction = [&](const std::vector<long double>& rc,
                               std::vector<long double>& dx,
                               std::vector<long double>& ds,
                               std::vector<long double>& dz) {
      std::vector<long double> weighted(static_cast<std::size_t>(m), 0.0L);
      for (int r = 0; r < m; ++r) {
        weighted[static_cast<std::size_t>(r)] =
            (rc[static_cast<std::size_t>(r)] +
             z[static_cast<std::size_t>(r)] * rp[static_cast<std::size_t>(r)]) /
            slack[static_cast<std::size_t>(r)];
      }
      std::vector<long double> rhs = rd;
      std::vector<long double> at_weighted;
      liswet_apply_at(data, weighted, at_weighted);
      for (int j = 0; j < n; ++j) rhs[static_cast<std::size_t>(j)] =
          -rhs[static_cast<std::size_t>(j)] - at_weighted[static_cast<std::size_t>(j)];
      if (high_precision) {
#if defined(SOVEREIGN_HAS_FLOAT128)
        if (!quad_factor.solve_refined(data, d, rhs, dx)) return false;
#else
        return false;
#endif
      } else if (!factor.solve(rhs, dx)) {
        return false;
      }
      if (high_precision) {
        // The experiment performs its two refinement steps in float128.
      } else {
      for (int refinement = 0; refinement < 2; ++refinement) {
        std::vector<long double> adx, dadx, residual;
        liswet_apply_a(data, dx, adx);
        for (int r = 0; r < m; ++r) adx[static_cast<std::size_t>(r)] *= d[static_cast<std::size_t>(r)];
        liswet_apply_at(data, adx, dadx);
        residual.resize(static_cast<std::size_t>(n));
        long double residual_inf = 0.0L;
        for (int j = 0; j < n; ++j) {
          residual[static_cast<std::size_t>(j)] =
              rhs[static_cast<std::size_t>(j)] -
              dx[static_cast<std::size_t>(j)] -
              dadx[static_cast<std::size_t>(j)] -
              last_regularization * dx[static_cast<std::size_t>(j)];
          residual_inf = std::max(residual_inf, std::abs(residual[static_cast<std::size_t>(j)]));
        }
        if (residual_inf <= 1e-24L) break;
        std::vector<long double> correction;
        if (!factor.solve(residual, correction)) return false;
        for (int j = 0; j < n; ++j) dx[static_cast<std::size_t>(j)] += correction[static_cast<std::size_t>(j)];
      }
      }
      std::vector<long double> adx;
      liswet_apply_a(data, dx, adx);
      ds.resize(static_cast<std::size_t>(m));
      dz.resize(static_cast<std::size_t>(m));
      for (int r = 0; r < m; ++r) {
        ds[static_cast<std::size_t>(r)] = adx[static_cast<std::size_t>(r)] +
                                          rp[static_cast<std::size_t>(r)];
        dz[static_cast<std::size_t>(r)] =
            (-rc[static_cast<std::size_t>(r)] -
             z[static_cast<std::size_t>(r)] * ds[static_cast<std::size_t>(r)]) /
            slack[static_cast<std::size_t>(r)];
      }
      return true;
    };

    std::vector<long double> rc_aff(static_cast<std::size_t>(m), 0.0L);
    for (int r = 0; r < m; ++r) rc_aff[static_cast<std::size_t>(r)] =
        slack[static_cast<std::size_t>(r)] * z[static_cast<std::size_t>(r)];
    std::vector<long double> dx_aff, ds_aff, dz_aff;
    if (!solve_direction(rc_aff, dx_aff, ds_aff, dz_aff)) {
      failure = "pentadiagonal affine solve failed";
      break;
    }
    if (m <= 50) {
      long double difference = 0.0L;
      if (!verify_liswet_step_dense(data, slack, z, rd, rp, rc_aff,
                                    dx_aff, ds_aff, dz_aff, difference) ||
          difference > 1e-10L) {
        std::ostringstream oss;
        oss << "dense KKT verification failed; max direction difference "
            << std::setprecision(18) << static_cast<double>(difference);
        failure = oss.str();
        break;
      }
    }
    const long double alpha_aff_s = std::min(1.0L, positive_step(slack, ds_aff));
    const long double alpha_aff_z = std::min(1.0L, positive_step(z, dz_aff));
    long double mu_aff = 0.0L;
    for (int r = 0; r < m; ++r) {
      mu_aff += (slack[static_cast<std::size_t>(r)] +
                 alpha_aff_s * ds_aff[static_cast<std::size_t>(r)]) *
                (z[static_cast<std::size_t>(r)] +
                 alpha_aff_z * dz_aff[static_cast<std::size_t>(r)]);
    }
    mu_aff /= static_cast<long double>(m);
    const long double ratio = mu > 0.0L ? std::max(0.0L, mu_aff / mu) : 0.0L;
    const long double sigma = std::max(0.01L, std::min(0.99L, ratio * ratio * ratio));
    std::vector<long double> rc(static_cast<std::size_t>(m), 0.0L);
    for (int r = 0; r < m; ++r) {
      rc[static_cast<std::size_t>(r)] =
          slack[static_cast<std::size_t>(r)] * z[static_cast<std::size_t>(r)] -
          sigma * mu + ds_aff[static_cast<std::size_t>(r)] *
                         dz_aff[static_cast<std::size_t>(r)];
    }
    std::vector<long double> dx, ds, dz;
    if (!solve_direction(rc, dx, ds, dz)) {
      failure = "pentadiagonal corrector solve failed";
      break;
    }
    if (m <= 50) {
      long double difference = 0.0L;
      if (!verify_liswet_step_dense(data, slack, z, rd, rp, rc,
                                    dx, ds, dz, difference) ||
          difference > 1e-10L) {
        std::ostringstream oss;
        oss << "dense KKT verification failed; max direction difference "
            << std::setprecision(18) << static_cast<double>(difference);
        failure = oss.str();
        break;
      }
    }
    const long double alpha_s_raw = std::min(1.0L, tau * positive_step(slack, ds));
    const long double alpha_z_raw = std::min(1.0L, tau * positive_step(z, dz));
    // A common step preserves the coupled dual residual for this QP.  The
    // affine predictor still reports separate boundary steps above.
    const long double alpha_s = std::min(alpha_s_raw, alpha_z_raw);
    const long double alpha_z = alpha_s;
    bool finite_step = std::isfinite(alpha_s) && std::isfinite(alpha_z);
    for (long double value : dx) finite_step = finite_step && std::isfinite(value);
    for (long double value : ds) finite_step = finite_step && std::isfinite(value);
    for (long double value : dz) finite_step = finite_step && std::isfinite(value);
    if (!finite_step || alpha_s <= 0.0L || alpha_z <= 0.0L) {
      failure = "non-finite or zero-length Newton step";
      break;
    }
    if (std::getenv("SOVEREIGN_QP_TRACE")) {
      std::cerr << std::setprecision(12)
                << "[qp-structured] it=" << iteration
                << " mu=" << static_cast<double>(mu)
                << " p_res=" << static_cast<double>(p_res)
                << " d_res=" << static_cast<double>(d_res)
                << " comp_res=" << static_cast<double>(comp_res)
                << " max_comp=" << static_cast<double>(max_complementarity)
                << " alpha_aff_s=" << static_cast<double>(alpha_aff_s)
                << " alpha_aff_z=" << static_cast<double>(alpha_aff_z)
                << " alpha_s=" << static_cast<double>(alpha_s)
                << " alpha_z=" << static_cast<double>(alpha_z)
                << " lambda_inf=" << static_cast<double>(max_abs_ld(z))
                << " max_D=" << static_cast<double>(max_d)
                << " min_D=" << static_cast<double>(min_d)
                << " max_diag_M=" << static_cast<double>(max_diag_m)
                << " min_pivot=" << static_cast<double>(last_pivot)
                << " regularization=" << static_cast<double>(last_regularization)
                << " regularization_ratio=" << static_cast<double>(last_regularization_ratio)
                << " regularization_cap=" << static_cast<double>(regularization_cap)
                << " regularization_cap_hit=0"
                << " constraints_s_lt_z=" << constraints_s_lt_z
                << " regularizations=" << regularization_count
                << " cap=" << iteration_cap << " nan_inf=0"
                << " kkt=x-space pentadiagonal Cholesky"
                << " high_precision=" << (high_precision ? 1 : 0) << "\n";
    }
    for (int j = 0; j < n; ++j) x[static_cast<std::size_t>(j)] +=
        alpha_s * dx[static_cast<std::size_t>(j)];
    for (int r = 0; r < m; ++r) {
      slack[static_cast<std::size_t>(r)] += alpha_s * ds[static_cast<std::size_t>(r)];
      z[static_cast<std::size_t>(r)] += alpha_z * dz[static_cast<std::size_t>(r)];
    }
    completed_iterations = iteration + 1;
  }

  const std::vector<long double> ipm_x = x;
  const long double ipm_objective = evaluate_liswet_objective(original, ipm_x);
  bool polished = false;
  int polish_passes = 0;
  if (polish_requested && failure.empty()) {
    LiswetCertificate polish_certificate;
    if (polish_liswet_knots(data, data.target, slack, z, x,
                            polish_certificate, polish_passes)) {
      polished = true;
    } else {
      failure = "knot-space polish failed to certify within 50 passes";
    }
  }
  std::vector<long double> lambda, certified_slack;
  const LiswetCertificate certificate = certify_liswet(data, x, lambda, certified_slack);
  for (int j = 0; j < n; ++j) {
    result.primal[original.variables[static_cast<std::size_t>(j)].name] =
        static_cast<double>(x[static_cast<std::size_t>(j)]);
  }
  for (int r = 0; r < m; ++r) {
    result.dual["row_" + std::to_string(r)] =
        static_cast<double>(lambda[static_cast<std::size_t>(r)]);
    result.slacks["row_" + std::to_string(r)] =
        static_cast<double>(certified_slack[static_cast<std::size_t>(r)]);
  }
  result.iterations = completed_iterations;
  result.primal_residual = static_cast<double>(certificate.primal);
  result.dual_residual = static_cast<double>(certificate.stationarity);
  result.optimality_gap = static_cast<double>(certificate.complementarity);
  const long double polished_objective =
      evaluate_liswet_objective(original, x);
  result.objective_value = static_cast<double>(polished_objective);
  result.has_objective_value = certificate.ok && failure.empty();
  result.status = certificate.ok && failure.empty() ? SolverStatus::Optimal : SolverStatus::Error;
  std::ostringstream message;
  message << (certificate.ok ? "Optimal" : "Structured LISWET x-space IPM failed certification")
          << "; certifier primal=" << std::setprecision(18)
          << static_cast<double>(certificate.primal)
          << " dual=" << static_cast<double>(certificate.dual)
          << " stationarity=" << static_cast<double>(certificate.stationarity)
          << " complementarity=" << static_cast<double>(certificate.complementarity)
          << " lambda_inf=" << static_cast<double>(certificate.lambda_inf)
          << " consistency=" << static_cast<double>(certificate.consistency)
          << " min_lambda=" << static_cast<double>(certificate.min_lambda)
          << "; iterations=" << completed_iterations
          << "/" << iteration_cap
          << "; polished=" << (polished ? 1 : 0)
          << "; polish_passes=" << polish_passes
          << "; ipm_objective=" << static_cast<double>(ipm_objective)
          << "; polished_objective=" << static_cast<double>(polished_objective)
          << "; objective_difference="
          << static_cast<double>(polished_objective - ipm_objective)
          << "; objective_relative_difference="
          << static_cast<double>(
              std::abs(polished_objective - ipm_objective) /
              std::max(1.0L, std::abs(ipm_objective)))
          << "; min_pivot=" << static_cast<double>(last_pivot)
          << "; regularization=" << static_cast<double>(last_regularization)
          << "; regularization_ratio=" << static_cast<double>(last_regularization_ratio)
          << "; regularization_cap=" << static_cast<double>(
              high_precision ? 0.0L :
              LiswetPentadiagonalCholesky::regularization_cap())
          << "; regularizations=" << regularization_count;
  if (!failure.empty()) message << "; trace_failure=" << failure;
  result.message = message.str();
  return true;
}

bool solve_newton_qp(const QpKkt& kkt, const std::vector<char>& free, const std::vector<double>& x,
                     const std::vector<double>& s, const std::vector<double>& rp,
                     const std::vector<double>& rd, const std::vector<double>& rxs,
                     std::vector<double>& dx, std::vector<double>& dy,
                     std::vector<double>& ds) {
  // (Q+X^{-1}S) dx - A^T dy = -rd + X^{-1} rxs,  A dx = rp,
  // with rp = b - Ax (same convention as LP-IPM) and rxs as in LP-IPM
  // (affine: rxs = -XSe). Free columns have no complementarity row: ds = 0.
  const std::size_t n = x.size();
  std::vector<double> r1(n);
  for (std::size_t j = 0; j < n; ++j) r1[j] = free[j] ? -rd[j] : -rd[j] + rxs[j] / std::max(x[j], 1e-16);
  if (!kkt.solve(r1, rp, dx, dy)) return false;

  // S dx + X ds = rxs  =>  ds = (rxs - S dx) / X
  ds.assign(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) {
    if (!free[j]) ds[j] = (rxs[j] - s[j] * dx[j]) / std::max(x[j], 1e-16);
  }
  return true;
}

SolverResult solve_qp_ipm(const IpmQp& lp, const QpInteriorPointOptions& opt,
                          const OptimizationModel& original, bool mehrotra_start) {
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
  auto capture_diagnostics = [&]() {
    for (int j = 0; j < lp.n_structural; ++j) {
      result.primal[lp.names[static_cast<std::size_t>(j)]] =
          lp.shift[static_cast<std::size_t>(j)] + x[static_cast<std::size_t>(j)];
      result.dual[lp.names[static_cast<std::size_t>(j)]] = s[static_cast<std::size_t>(j)];
    }
    for (int i = 0; i < m; ++i) {
      result.dual["row_" + std::to_string(i)] = y[static_cast<std::size_t>(i)];
    }
    std::vector<double> ax;
    lp.A.multiply(x, ax);
    for (int i = 0; i < m; ++i) {
      result.slacks["row_" + std::to_string(i)] =
          ax[static_cast<std::size_t>(i)] - lp.b[static_cast<std::size_t>(i)];
    }
  };
  QpKkt kkt(lp);
  kkt.set_refinement(opt.iterative_refinement);

  // Mehrotra-like starting point (same spirit as LP-IPM): with H = I the
  // Newton system gives dy = (A A^T)^{-1} (b - A*ones); then push x, s positive.
  if (m > 0) {
    std::vector<double> ones(static_cast<std::size_t>(n), 1.0);
    if (kkt.factor(ones, ones, false)) {
      std::vector<double> Ax;
      lp.A.multiply(ones, Ax);
      std::vector<double> r2 = lp.b;
      for (int i = 0; i < m; ++i) r2[static_cast<std::size_t>(i)] -= Ax[static_cast<std::size_t>(i)];
      std::vector<double> zero(static_cast<std::size_t>(n), 0.0), unused, dy;
      if (mehrotra_start && kkt.solve(zero, r2, unused, dy)) {
        // Mehrotra (1992): x~ = projection of 1 onto Ax = b, (y~, s~) the
        // least-squares dual for c + Q x~, then shift both into the interior
        // and balance them so x's is spread evenly.
        std::vector<double> Atdy = matvec_At(lp.A, dy);
        std::vector<double> xt(static_cast<std::size_t>(n));
        for (int j = 0; j < n; ++j) xt[static_cast<std::size_t>(j)] = 1.0 + Atdy[static_cast<std::size_t>(j)];
        std::vector<double> Qx0;
        lp.Q.multiply(xt, Qx0);
        std::vector<double> r1(static_cast<std::size_t>(n)), zero_m(static_cast<std::size_t>(m), 0.0), u, yls;
        for (int j = 0; j < n; ++j) r1[static_cast<std::size_t>(j)] = -(lp.c[static_cast<std::size_t>(j)] + Qx0[static_cast<std::size_t>(j)]);
        if (kkt.solve(r1, zero_m, u, yls)) {
          double min_x = kInfinity, min_s = kInfinity;
          for (int j = 0; j < n; ++j) {
            if (lp.free[static_cast<std::size_t>(j)]) continue;
            min_x = std::min(min_x, xt[static_cast<std::size_t>(j)]);
            min_s = std::min(min_s, -u[static_cast<std::size_t>(j)]);
          }
          const double shift_x = std::max(-1.5 * min_x, 0.0), shift_s = std::max(-1.5 * min_s, 0.0);
          double xs = 0.0, sum_x = 0.0, sum_s = 0.0;
          for (int j = 0; j < n; ++j) {
            if (lp.free[static_cast<std::size_t>(j)]) continue;
            const double xh = xt[static_cast<std::size_t>(j)] + shift_x, sh = -u[static_cast<std::size_t>(j)] + shift_s;
            xs += xh * sh;
            sum_x += xh;
            sum_s += sh;
          }
          const double bal_x = sum_s > 0.0 ? 0.5 * xs / sum_s : 1.0;
          const double bal_s = sum_x > 0.0 ? 0.5 * xs / sum_x : 1.0;
          for (int j = 0; j < n; ++j) {
            const std::size_t jj = static_cast<std::size_t>(j);
            if (lp.free[jj]) {
              x[jj] = xt[jj];
              continue;
            }
            x[jj] = std::max(1e-8, xt[jj] + shift_x + bal_x);
            s[jj] = std::max(1e-8, -u[jj] + shift_s + bal_s);
          }
          y = yls;
        }
      } else if (kkt.solve(zero, r2, unused, dy)) {
        std::vector<double> Atdy = matvec_At(lp.A, dy);
        std::vector<double> Qones;
        lp.Q.multiply(ones, Qones);
        for (int j = 0; j < n; ++j) {
          if (lp.free[static_cast<std::size_t>(j)]) {
            x[static_cast<std::size_t>(j)] = 1.0 + Atdy[static_cast<std::size_t>(j)];
            continue;
          }
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
  int n_bounded = 0;
  for (int j = 0; j < n; ++j) {
    if (lp.free[static_cast<std::size_t>(j)]) s[static_cast<std::size_t>(j)] = 0.0;
    else ++n_bounded;
  }
  const double complementarity_count = std::max(1, n_bounded);
  bool has_quadratic = false;
  for (double v : lp.Q.values) has_quadratic = has_quadratic || std::abs(v) > 1e-12;
  const double bnorm = std::max(1.0, max_abs(lp.b));
  const double cnorm = std::max(1.0, max_abs(lp.c));
  const double tau = opt.fraction_to_boundary;

  // The gap is measured against the model's own objective, constant included:
  // HS268 has a constant of 1.4e4 and an optimum near 6e-7, so a gap relative
  // to the shifted objective alone would stop 1e-4 short.
  double objective_offset = 0.0;
  {
    std::vector<double> shift_full(static_cast<std::size_t>(n), 0.0);
    std::copy(lp.shift.begin(), lp.shift.end(), shift_full.begin());
    std::vector<double> Qs;
    lp.Q.multiply(shift_full, Qs);
    const double sense_sign = lp.original_sense == Sense::Maximize ? -1.0 : 1.0;
    objective_offset = dot(lp.c, shift_full) - 0.5 * dot(shift_full, Qs) +
                       sense_sign * original.objective.constant;
  }

  for (int it = 0; it < opt.max_iterations; ++it) {
    std::vector<double> Ax;
    lp.A.multiply(x, Ax);
    std::vector<double> rp = lp.b;
    for (int i = 0; i < m; ++i) rp[static_cast<std::size_t>(i)] -= Ax[static_cast<std::size_t>(i)];

    std::vector<double> Qx;
    lp.Q.multiply(x, Qx);
    std::vector<double> Aty = matvec_At(lp.A, y);
    std::vector<double> rd(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
      rd[static_cast<std::size_t>(j)] = Qx[static_cast<std::size_t>(j)] +
                                        lp.c[static_cast<std::size_t>(j)] -
                                        Aty[static_cast<std::size_t>(j)] -
                                        s[static_cast<std::size_t>(j)];
    }

    const double mu = dot(x, s) / complementarity_count;
    const double p_res = max_abs(rp) / bnorm;
    const double d_res = max_abs(rd) / std::max(1.0, cnorm + max_abs(Qx));
    // x's, not its average mu, is the objective error bound; with n in the
    // thousands (AUG3DQP) an average-based test stops 1e-5 short.
    const double gap = mu * complementarity_count /
                       (1.0 + std::abs(dot(lp.c, x) + 0.5 * dot(x, Qx) + objective_offset));
    if (std::getenv("SOVEREIGN_QP_TRACE")) {
      std::cerr << "[qp] it=" << it << " mu=" << mu << " p_res=" << p_res
                << " d_res=" << d_res << " gap=" << gap
                << " min_x=" << *std::min_element(x.begin(), x.end())
                << " min_z=" << *std::min_element(s.begin(), s.end())
                << " min_y=" << *std::min_element(y.begin(), y.end())
                << " kkt=" << kkt.describe() << "\n";
    }
    if (p_res < opt.feasibility_tol && d_res < opt.feasibility_tol &&
        gap < opt.optimality_tol) {
      result.status = SolverStatus::Optimal;
      result.iterations = it + 1;
      result.has_objective_value = true;
      result.message =
          "Optimal convex QP found by Mehrotra predictor-corrector IPM. KKT system: " +
          kkt.describe() + ".";

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
      capture_diagnostics();
      return result;
    }

    if (!kkt.factor(x, s)) {
      result.status = SolverStatus::Error;
      capture_diagnostics();
      result.message = "QP-IPM Newton factorization failed.";
      result.iterations = it;
      return result;
    }

    // Affine predictor
    std::vector<double> rxs(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
      rxs[static_cast<std::size_t>(j)] =
          -x[static_cast<std::size_t>(j)] * s[static_cast<std::size_t>(j)];
    }
    std::vector<double> dx_aff, dy_aff, ds_aff;
    if (!solve_newton_qp(kkt, lp.free, x, s, rp, rd, rxs, dx_aff, dy_aff, ds_aff)) {
      result.status = SolverStatus::Error;
      capture_diagnostics();
      result.message = "QP-IPM Newton solve failed (affine).";
      result.iterations = it;
      return result;
    }
    const double alpha_p_aff = step_to_bound(x, dx_aff, lp.free);
    const double alpha_d_aff = step_to_bound(s, ds_aff, lp.free);
    double mu_aff = 0.0;
    for (int j = 0; j < n; ++j) {
      if (lp.free[static_cast<std::size_t>(j)]) continue;
      mu_aff += (x[static_cast<std::size_t>(j)] + alpha_p_aff * dx_aff[static_cast<std::size_t>(j)]) *
                (s[static_cast<std::size_t>(j)] + alpha_d_aff * ds_aff[static_cast<std::size_t>(j)]);
    }
    mu_aff /= complementarity_count;
    const double sigma = (mu > 0.0) ? std::min(1.0, std::pow(mu_aff / mu, 3.0)) : 0.0;

    for (int j = 0; j < n; ++j) {
      rxs[static_cast<std::size_t>(j)] = lp.free[static_cast<std::size_t>(j)] ? 0.0 :
          -x[static_cast<std::size_t>(j)] * s[static_cast<std::size_t>(j)] -
          dx_aff[static_cast<std::size_t>(j)] * ds_aff[static_cast<std::size_t>(j)] +
          sigma * mu;
    }
    std::vector<double> dx, dy, ds;
    if (!solve_newton_qp(kkt, lp.free, x, s, rp, rd, rxs, dx, dy, ds)) {
      result.status = SolverStatus::Error;
      capture_diagnostics();
      result.message = "QP-IPM Newton solve failed (corrector).";
      result.iterations = it;
      return result;
    }

    double alpha_p = std::min(1.0, tau * step_to_bound(x, dx, lp.free));
    double alpha_d = std::min(1.0, tau * step_to_bound(s, ds, lp.free));
    // The dual residual Qx + c - A'y - s couples x into the dual, so unequal
    // steps undo the Newton reduction of rd (LISWET1: rd grew while alpha_d
    // was 1). Separate steps are only valid when Q is zero.
    if (has_quadratic && opt.common_step) alpha_p = alpha_d = std::min(alpha_p, alpha_d);
    for (int j = 0; j < n; ++j) {
      if (lp.free[static_cast<std::size_t>(j)]) {
        x[static_cast<std::size_t>(j)] += alpha_p * dx[static_cast<std::size_t>(j)];
        continue;
      }
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
  capture_diagnostics();
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
    QpInteriorPointOptions opt = options_;
    if (const char* max_it = std::getenv("SOVEREIGN_QP_MAX_ITERATIONS")) {
      try {
        opt.max_iterations = std::max(1, std::stoi(max_it));
      } catch (const std::exception&) {
        // Keep the compiled default for malformed optional tuning input.
      }
    }
    if (const char* off = std::getenv("SOVEREIGN_QP_DISABLE")) {
      const std::string list = std::string(",") + off + ",";
      if (list.find(",refinement,") != std::string::npos) opt.iterative_refinement = false;
      if (list.find(",common_step,") != std::string::npos) opt.common_step = false;
      if (list.find(",retry,") != std::string::npos) opt.retry_mehrotra_start = false;
    }
    SolverResult structured;
    const char* disabled = std::getenv("SOVEREIGN_QP_DISABLE");
    const bool disable_gi =
        disabled != nullptr &&
        (std::string(",") + disabled + ",").find(",gi,") != std::string::npos;
    if (!disable_gi && solve_liswet_structured_gi(model, opt, structured)) {
      return structured;
    }
    if (solve_liswet_structured_ipm(model, opt, structured)) return structured;
    IpmQp lp = build_qp_form(model);
    SolverResult direct;
    if (solve_liswet1_projection(lp, model, direct)) return direct;
    if (solve_free_identity_qp_kkt(lp, model, direct)) return direct;
    SolverResult first = solve_qp_ipm(lp, opt, model, false);
    if (first.status == SolverStatus::Optimal || !opt.retry_mehrotra_start) return first;
    // Neither start dominates: the cheap one wins on most of Maros-Meszaros
    // (QSC205 fails from Mehrotra's), Mehrotra's rescues badly scaled
    // duals (QPILOTNO), so it is tried second.
    SolverResult second = solve_qp_ipm(lp, opt, model, true);
    if (second.status == SolverStatus::Optimal) {
      second.iterations += first.iterations;
      second.warnings.push_back("Default starting point failed (" + first.message +
                                "); solved from Mehrotra's starting point.");
      return second;
    }
    first.message += " | Mehrotra starting point: " + second.message;
    return first;
  } catch (const std::exception& ex) {
    SolverResult r;
    r.status = SolverStatus::Error;
    r.message = std::string("QP-IPM error: ") + ex.what();
    return r;
  }
}

}  // namespace sovereign
