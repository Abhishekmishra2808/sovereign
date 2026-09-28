#include "sovereign/qp_interior_point.hpp"

#include "sovereign/dense_lu.hpp"
#include "sovereign/gpu_spmv.hpp"
#include "sovereign/sparse_ldlt.hpp"
#include "sovereign/sparse_matrix.hpp"
#include "sovereign/sparse_symmetric.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
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
      for (std::size_t j = 0; j < n_; ++j) h_[j] = std::max(s[j], 1e-16) / std::max(x[j], 1e-16);
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
    for (int step = 0; step < kRefinementSteps && err > 0.0; ++step) {
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

bool solve_newton_qp(const QpKkt& kkt, const std::vector<double>& x,
                     const std::vector<double>& s, const std::vector<double>& rp,
                     const std::vector<double>& rd, const std::vector<double>& rxs,
                     std::vector<double>& dx, std::vector<double>& dy,
                     std::vector<double>& ds) {
  // (Q+X^{-1}S) dx - A^T dy = -rd + X^{-1} rxs,  A dx = rp,
  // with rp = b - Ax (same convention as LP-IPM) and rxs as in LP-IPM
  // (affine: rxs = -XSe).
  const std::size_t n = x.size();
  std::vector<double> r1(n);
  for (std::size_t j = 0; j < n; ++j) r1[j] = -rd[j] + rxs[j] / std::max(x[j], 1e-16);
  if (!kkt.solve(r1, rp, dx, dy)) return false;

  // S dx + X ds = rxs  =>  ds = (rxs - S dx) / X
  ds.assign(n, 0.0);
  for (std::size_t j = 0; j < n; ++j) ds[j] = (rxs[j] - s[j] * dx[j]) / std::max(x[j], 1e-16);
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
  QpKkt kkt(lp);

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
      if (kkt.solve(zero, r2, unused, dy)) {
        std::vector<double> Atdy = matvec_At(lp.A, dy);
        std::vector<double> Qones;
        lp.Q.multiply(ones, Qones);
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
    lp.Q.multiply(x, Qx);
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
      return result;
    }

    if (!kkt.factor(x, s)) {
      result.status = SolverStatus::Error;
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
    if (!solve_newton_qp(kkt, x, s, rp, rd, rxs, dx_aff, dy_aff, ds_aff)) {
      result.status = SolverStatus::Error;
      result.message = "QP-IPM Newton solve failed (affine).";
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
    if (!solve_newton_qp(kkt, x, s, rp, rd, rxs, dx, dy, ds)) {
      result.status = SolverStatus::Error;
      result.message = "QP-IPM Newton solve failed (corrector).";
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
