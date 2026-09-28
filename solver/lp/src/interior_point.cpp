#include "sovereign/interior_point.hpp"

#include "sovereign/dense_lu.hpp"
#include "sovereign/gpu_spmv.hpp"
#include "sovereign/sparse_ldlt.hpp"
#include "sovereign/sparse_matrix.hpp"
#include "sovereign/sparse_symmetric.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace sovereign {
namespace {

struct IpmLp {
  SparseMatrixCSC A;  // m x n
  std::vector<double> b;
  std::vector<double> c;  // minimization
  std::vector<std::string> names;
  std::vector<int> structural_index;
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

void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
  for (std::size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

std::vector<double> matvec_At(const SparseMatrixCSC& A, const std::vector<double>& y) {
  std::vector<double> out;
  A.multiply_transpose(y, out);
  return out;
}

void build_normal_eq(const SparseMatrixCSC& A, const std::vector<double>& d,
                     std::vector<double>& M_col_major) {
  // M = A diag(d) A^T , column-major m x m
  const int m = static_cast<int>(A.nrows);
  const int n = static_cast<int>(A.ncols);
  M_col_major.assign(static_cast<std::size_t>(m) * static_cast<std::size_t>(m), 0.0);
  for (int j = 0; j < n; ++j) {
    const double dj = d[static_cast<std::size_t>(j)];
    if (dj == 0.0) continue;
    for (int p = A.col_ptr[static_cast<std::size_t>(j)];
         p < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const int r = A.row_idx[static_cast<std::size_t>(p)];
      const double ar = A.values[static_cast<std::size_t>(p)] * dj;
      for (int q = A.col_ptr[static_cast<std::size_t>(j)];
           q < A.col_ptr[static_cast<std::size_t>(j) + 1]; ++q) {
        const int c = A.row_idx[static_cast<std::size_t>(q)];
        const double ac = A.values[static_cast<std::size_t>(q)];
        M_col_major[static_cast<std::size_t>(c) * static_cast<std::size_t>(m) +
                    static_cast<std::size_t>(r)] += ar * ac;
      }
    }
  }
  // Symmetrize / stabilize diagonal
  for (int i = 0; i < m; ++i) {
    M_col_major[static_cast<std::size_t>(i) * static_cast<std::size_t>(m) +
                static_cast<std::size_t>(i)] += 1e-12;
  }
}

constexpr double kNormalEqRegularization = 1e-12;
// Above this many rows the dense m x m system no longer fits comfortably in a
// 32-bit process (8000^2 doubles = 512 MB), so only the sparse path is tried.
constexpr std::size_t kDenseMaxRows = 8000;

std::size_t env_size(const char* name, std::size_t fallback) {
  const char* v = std::getenv(name);
  if (!v || !*v) return fallback;
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(v, &end, 10);
  return (end && *end == '\0') ? static_cast<std::size_t>(parsed) : fallback;
}

// Factors M = A D A^T once per interior-point iteration.
//
// SOVEREIGN_IPM_NORMAL_EQUATIONS selects the method: "dense" always uses
// DenseLU (which runs on the GPU when SOVEREIGN_DEVICE allows), "sparse"
// always uses the minimum-degree-ordered sparse LDL^T, and "auto" (default)
// picks per model from at least SOVEREIGN_IPM_SPARSE_MIN_ROWS rows:
//   - an explicit CUDA request keeps the dense GPU factorization while it fits;
//   - on the CPU, sparse LDL^T beat the dense LU on every model measured, even
//     at 65% fill, so it is used unless the factor is essentially dense;
//   - with a usable GPU under SOVEREIGN_DEVICE=auto, the dense GPU LU takes
//     over once one sparse factorization exceeds SOVEREIGN_IPM_GPU_MIN_FLOPS
//     multiply-adds. On an RTX 2050 the two tied at 1.6e8 and the GPU was 1.5x
//     faster at 5.1e8, hence the 2e8 default.
class NormalEquations {
 public:
  NormalEquations(const SparseMatrixCSC& A, bool force_sparse) : A_(A), m_(A.nrows) {
    const char* raw = std::getenv("SOVEREIGN_IPM_NORMAL_EQUATIONS");
    const std::string mode = force_sparse ? "sparse" : raw && *raw ? raw : "auto";
    if (mode == "dense" || m_ == 0) return;
    const double dense_entries = 0.5 * static_cast<double>(m_) * static_cast<double>(m_ + 1);
    const bool dense_fits = m_ <= kDenseMaxRows;
    std::size_t limit = 0;
    bool gpu_dense = false;
    if (mode != "sparse" && dense_fits) {
      if (m_ < env_size("SOVEREIGN_IPM_SPARSE_MIN_ROWS", 200)) return;
      const std::string device = requested_device();
      if (device == "cuda") return;
      // Each column with k nonzeros couples k(k+1)/2 entries of M, so a few
      // dense columns make the pattern itself as large as the dense system.
      double pairs = 0.0;
      for (std::size_t j = 0; j < A_.ncols; ++j) {
        const double k = static_cast<double>(A_.col_ptr[j + 1] - A_.col_ptr[j]);
        pairs += 0.5 * k * (k + 1.0);
      }
      if (pairs > 4.0 * dense_entries) return;
      limit = static_cast<std::size_t>(0.9 * dense_entries);
      gpu_dense = device == "auto" && m_ >= env_size("SOVEREIGN_GPU_DENSE_MIN", 400) && gpu_available();
    }
    pattern_ = build_normal_eq_pattern(A_);
    use_sparse_ = ldlt_.symbolic_analyze(pattern_, limit);
    if (use_sparse_ && gpu_dense &&
        ldlt_.factor_flops() > static_cast<double>(env_size("SOVEREIGN_IPM_GPU_MIN_FLOPS", 200000000))) {
      use_sparse_ = false;
    }
    ldlt_.set_tiny_pivot_threshold(1e-13);
  }

  bool factor(const std::vector<double>& d) {
    last_sparse_ = false;
    if (use_sparse_) {
      build_normal_eq_values(A_, d, pattern_, values_);
      if (ldlt_.numeric_factor(values_, kNormalEqRegularization)) {
        last_sparse_ = true;
        ++sparse_factorizations_;
        replaced_pivots_ = std::max(replaced_pivots_, ldlt_.replaced_pivots());
        return true;
      }
      // A breakdown near convergence is retried with the pivoting dense LU.
      if (m_ > kDenseMaxRows) return false;
    }
    std::vector<double> M;
    build_normal_eq(A_, d, M);
    ++dense_factorizations_;
    return lu_.factorize(std::move(M), m_);
  }

  bool solve(std::vector<double>& rhs) const { return last_sparse_ ? ldlt_.solve(rhs) : lu_.solve(rhs); }

  std::string describe() const {
    std::ostringstream oss;
    if (!use_sparse_) {
      oss << "dense LU (" << m_ << " x " << m_ << ")";
    } else {
      oss << "sparse LDL^T with minimum-degree ordering (nnz(L) = " << ldlt_.factor_nnz()
          << ", dense triangle " << static_cast<std::size_t>(0.5 * static_cast<double>(m_) * static_cast<double>(m_ + 1))
          << ")";
      if (replaced_pivots_ > 0) oss << ", up to " << replaced_pivots_ << " dependent rows dropped per iteration";
      if (dense_factorizations_ > 0) oss << ", " << dense_factorizations_ << " dense LU retries";
    }
    return oss.str();
  }

 private:
  const SparseMatrixCSC& A_;
  std::size_t m_;
  bool use_sparse_ = false;
  bool last_sparse_ = false;
  int sparse_factorizations_ = 0;
  int dense_factorizations_ = 0;
  int replaced_pivots_ = 0;
  SparseSymmetricPattern pattern_;
  SparseLDLT ldlt_;
  std::vector<double> values_;
  DenseLU lu_;
};

double step_to_bound(const std::vector<double>& x, const std::vector<double>& dx) {
  double alpha = 1.0;
  for (std::size_t i = 0; i < x.size(); ++i) {
    if (dx[i] < 0.0) {
      alpha = std::min(alpha, -x[i] / dx[i]);
    }
  }
  return alpha;
}

IpmLp build_ipm_form(const OptimizationModel& model, bool enable_scaling) {
  IpmLp lp;
  lp.original_sense = model.sense;
  lp.n_structural = static_cast<int>(model.variables.size());
  const int m0 = static_cast<int>(model.constraints.size());

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

  const int m_total = m0 + static_cast<int>(ub_cons.size());
  std::vector<double> rhs(static_cast<std::size_t>(m_total), 0.0);
  std::vector<std::vector<std::pair<int, double>>> rows(
      static_cast<std::size_t>(m_total));

  auto add_coeff = [&](int row, int col, double val) {
    if (val == 0.0) return;
    rows[static_cast<std::size_t>(row)].push_back({col, val});
  };

  std::unordered_map<std::string, int> var_index;
  var_index.reserve(static_cast<std::size_t>(lp.n_structural) * 2);
  for (int i = 0; i < lp.n_structural; ++i) {
    var_index[model.variables[static_cast<std::size_t>(i)].name] = i;
  }

  for (int r = 0; r < m0; ++r) {
    const Constraint& cons = model.constraints[static_cast<std::size_t>(r)];
    double b = cons.rhs;
    for (const auto& kv : cons.linear) {
      auto it = var_index.find(kv.first);
      if (it == var_index.end()) {
        throw std::runtime_error("Unknown variable in constraint: " + kv.first);
      }
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

  // IPM form: slacks only (no artificials). Ge: Ax - surplus = b; Eq: Ax = b.
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
  lp.structural_index.assign(static_cast<std::size_t>(n_total), -1);

  const double sense_sign = (model.sense == Sense::Maximize) ? -1.0 : 1.0;
  for (int i = 0; i < lp.n_structural; ++i) {
    lp.names[static_cast<std::size_t>(i)] = model.variables[static_cast<std::size_t>(i)].name;
    lp.structural_index[static_cast<std::size_t>(i)] = i;
    double coef = 0.0;
    auto it = model.objective.linear.find(lp.names[static_cast<std::size_t>(i)]);
    if (it != model.objective.linear.end()) coef = it->second;
    lp.c[static_cast<std::size_t>(i)] = sense_sign * coef;
  }

  std::vector<std::vector<std::pair<int, double>>> cols(
      static_cast<std::size_t>(n_total));
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

  lp.col_scale.assign(static_cast<std::size_t>(n_total), 1.0);
  if (enable_scaling && m_total > 0 && n_total > 0) {
    for (int pass = 0; pass < 2; ++pass) {
      std::vector<double> row_max(static_cast<std::size_t>(m_total), 0.0);
      for (std::size_t p = 0; p < lp.A.values.size(); ++p) {
        const int r = lp.A.row_idx[p];
        row_max[static_cast<std::size_t>(r)] =
            std::max(row_max[static_cast<std::size_t>(r)], std::abs(lp.A.values[p]));
      }
      std::vector<double> row_scale(static_cast<std::size_t>(m_total), 1.0);
      for (int i = 0; i < m_total; ++i) {
        if (row_max[static_cast<std::size_t>(i)] > 0.0) {
          row_scale[static_cast<std::size_t>(i)] =
              1.0 / std::sqrt(row_max[static_cast<std::size_t>(i)]);
        }
        lp.b[static_cast<std::size_t>(i)] *= row_scale[static_cast<std::size_t>(i)];
      }
      for (std::size_t p = 0; p < lp.A.values.size(); ++p) {
        lp.A.values[p] *= row_scale[static_cast<std::size_t>(lp.A.row_idx[p])];
      }
      for (int j = 0; j < n_total; ++j) {
        double maxv = 0.0;
        for (int p = lp.A.col_ptr[static_cast<std::size_t>(j)];
             p < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
          maxv = std::max(maxv, std::abs(lp.A.values[static_cast<std::size_t>(p)]));
        }
        const double s = (maxv > 0.0) ? 1.0 / std::sqrt(maxv) : 1.0;
        for (int p = lp.A.col_ptr[static_cast<std::size_t>(j)];
             p < lp.A.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
          lp.A.values[static_cast<std::size_t>(p)] *= s;
        }
        lp.c[static_cast<std::size_t>(j)] *= s;
        lp.col_scale[static_cast<std::size_t>(j)] *= s;
      }
    }
  }
  return lp;
}

std::vector<double> scaling_diagonal(const std::vector<double>& x, const std::vector<double>& s) {
  std::vector<double> d(x.size());
  for (std::size_t j = 0; j < x.size(); ++j) d[j] = std::max(x[j], 1e-16) / std::max(s[j], 1e-16);
  return d;
}

// Solves the Newton system with normal equations already factored for d = x/s.
bool solve_newton(const IpmLp& lp, const NormalEquations& normal, const std::vector<double>& d,
                  const std::vector<double>& s, const std::vector<double>& rp,
                  const std::vector<double>& rd, const std::vector<double>& rxs,
                  std::vector<double>& dx, std::vector<double>& dy,
                  std::vector<double>& ds) {
  const int n = lp.n;
  std::vector<double> tmp(static_cast<std::size_t>(n), 0.0);
  for (int j = 0; j < n; ++j) {
    const double sj = std::max(s[static_cast<std::size_t>(j)], 1e-16);
    tmp[static_cast<std::size_t>(j)] =
        d[static_cast<std::size_t>(j)] * rd[static_cast<std::size_t>(j)] -
        rxs[static_cast<std::size_t>(j)] / sj;
  }

  std::vector<double> Atmp;
  lp.A.multiply(tmp, Atmp);
  std::vector<double> rhs = rp;
  axpy(1.0, Atmp, rhs);

  dy = rhs;
  if (!normal.solve(dy)) return false;

  std::vector<double> Atdy = matvec_At(lp.A, dy);
  dx.assign(static_cast<std::size_t>(n), 0.0);
  ds.assign(static_cast<std::size_t>(n), 0.0);
  for (int j = 0; j < n; ++j) {
    dx[static_cast<std::size_t>(j)] =
        d[static_cast<std::size_t>(j)] *
            (Atdy[static_cast<std::size_t>(j)] - rd[static_cast<std::size_t>(j)]) +
        rxs[static_cast<std::size_t>(j)] / std::max(s[static_cast<std::size_t>(j)], 1e-16);
    ds[static_cast<std::size_t>(j)] =
        rd[static_cast<std::size_t>(j)] - Atdy[static_cast<std::size_t>(j)];
  }
  return true;
}

SolverResult solve_ipm(const IpmLp& lp, const InteriorPointOptions& opt,
                       const OptimizationModel& original) {
  SolverResult result;
  const int m = lp.m;
  const int n = lp.n;
  if (n <= 0) {
    result.status = SolverStatus::Error;
    result.message = "IPM: empty model.";
    return result;
  }
  if (m <= 0) {
    // Unconstrained: same logic as simplex path
    result.status = SolverStatus::Optimal;
    result.has_objective_value = true;
    double obj = original.objective.constant;
    for (int i = 0; i < lp.n_structural; ++i) {
      const double x = lp.shift[static_cast<std::size_t>(i)];
      result.primal[lp.names[static_cast<std::size_t>(i)]] = x;
      auto it = original.objective.linear.find(lp.names[static_cast<std::size_t>(i)]);
      if (it != original.objective.linear.end()) obj += it->second * x;
    }
    result.objective_value = obj;
    result.message = "Optimal (no constraints) via IPM path.";
    return result;
  }

  std::vector<double> x(static_cast<std::size_t>(n), 1.0);
  std::vector<double> s(static_cast<std::size_t>(n), 1.0);
  std::vector<double> y(static_cast<std::size_t>(m), 0.0);
  NormalEquations normal(lp.A, opt.use_sparse_normal_equations);

  // Mehrotra-like starting point from least-squares residual push
  {
    std::vector<double> ones(static_cast<std::size_t>(n), 1.0);
    if (normal.factor(ones)) {
      std::vector<double> Ax;
      lp.A.multiply(ones, Ax);
      std::vector<double> dy = lp.b;
      for (int i = 0; i < m; ++i) dy[static_cast<std::size_t>(i)] -= Ax[static_cast<std::size_t>(i)];
      if (normal.solve(dy)) {
        std::vector<double> Atdy = matvec_At(lp.A, dy);
        for (int j = 0; j < n; ++j) {
          x[static_cast<std::size_t>(j)] = std::max(1.0, std::abs(Atdy[static_cast<std::size_t>(j)]));
          s[static_cast<std::size_t>(j)] =
              std::max(1.0, std::abs(lp.c[static_cast<std::size_t>(j)] - Atdy[static_cast<std::size_t>(j)]));
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

    std::vector<double> Aty = matvec_At(lp.A, y);
    std::vector<double> rd = lp.c;
    for (int j = 0; j < n; ++j) {
      rd[static_cast<std::size_t>(j)] -=
          Aty[static_cast<std::size_t>(j)] + s[static_cast<std::size_t>(j)];
    }

    const double mu = dot(x, s) / static_cast<double>(n);
    const double p_res = max_abs(rp) / bnorm;
    const double d_res = max_abs(rd) / cnorm;

    // The duality gap is the *sum* of complementarity products, x's, NOT their
    // average mu = x's/n.
    //
    // Derivation for min c'x s.t. Ax = b, x >= 0 with dual A'y + s = c, s >= 0:
    //   c'x - b'y = c'x - (Ax)'y = x'(c - A'y) = x'(rd + s) = x'rd + x's
    // so once the dual residual rd is small, c'x - b'y and x's agree. Dividing
    // by n makes the termination test n times too lenient: on a wide sparse LP
    // (say n = 10200 transport variables) x's/n collapses below the tolerance
    // while the real gap is still ~1e-4 relative, and the solver reports
    // OPTIMAL on a measurably suboptimal point. That is exactly what happened
    // on transport_50x50 (504.604 vs 504.600) and transport_100x100
    // (1009.0399 vs 1009.0000).
    const double complementarity = dot(x, s);
    const double gap = std::abs(complementarity) / (1.0 + std::abs(dot(lp.c, x)));

    if (p_res < opt.feasibility_tol && d_res < opt.feasibility_tol &&
        gap < opt.optimality_tol) {
      result.status = SolverStatus::Optimal;
      result.iterations = it + 1;
      result.has_objective_value = true;
      result.duality_gap = gap;
      result.primal_residual = p_res;
      result.dual_residual = d_res;
      result.message = "Optimal solution found by primal-dual interior-point (Mehrotra). Normal equations: " +
                       normal.describe() + ".";

      // Map structural solution back
      double obj = original.objective.constant;
      for (int j = 0; j < lp.n_structural; ++j) {
        const double yj =
            lp.col_scale[static_cast<std::size_t>(j)] * x[static_cast<std::size_t>(j)];
        const double xv = lp.shift[static_cast<std::size_t>(j)] + yj;
        result.primal[lp.names[static_cast<std::size_t>(j)]] = xv;
        auto itc = original.objective.linear.find(lp.names[static_cast<std::size_t>(j)]);
        if (itc != original.objective.linear.end()) obj += itc->second * xv;
      }
      result.objective_value = obj;
      return result;
    }

    // Affine scaling predictor: rxs = -XSe
    std::vector<double> rxs(static_cast<std::size_t>(n), 0.0);
    for (int j = 0; j < n; ++j) {
      rxs[static_cast<std::size_t>(j)] =
          -x[static_cast<std::size_t>(j)] * s[static_cast<std::size_t>(j)];
    }
    const std::vector<double> d = scaling_diagonal(x, s);
    std::vector<double> dx_aff, dy_aff, ds_aff;
    if (!normal.factor(d) || !solve_newton(lp, normal, d, s, rp, rd, rxs, dx_aff, dy_aff, ds_aff)) {
      result.status = SolverStatus::NumericalError;
      result.message =
          "IPM Newton normal-equations factorization is singular on the affine predictor "
          "(the basis is numerically dependent). The iterate is not trustworthy, so falling "
          "back to the simplex path is the correct response.";
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
    const double sigma =
        (mu > 0.0) ? std::min(1.0, std::pow(mu_aff / mu, 3.0)) : 0.0;

    // Corrector: rxs = -XSe - dx_aff.*ds_aff + sigma*mu*e
    for (int j = 0; j < n; ++j) {
      rxs[static_cast<std::size_t>(j)] =
          -x[static_cast<std::size_t>(j)] * s[static_cast<std::size_t>(j)] -
          dx_aff[static_cast<std::size_t>(j)] * ds_aff[static_cast<std::size_t>(j)] +
          sigma * mu;
    }
    std::vector<double> dx, dy, ds;
    if (!solve_newton(lp, normal, d, s, rp, rd, rxs, dx, dy, ds)) {
      result.status = SolverStatus::NumericalError;
      result.message =
          "IPM Newton normal-equations factorization is singular on the corrector "
          "(the basis is numerically dependent). The iterate is not trustworthy, so "
          "falling back to the simplex path is the correct response.";
      result.iterations = it;
      return result;
    }

    double alpha_p = tau * step_to_bound(x, dx);
    double alpha_d = tau * step_to_bound(s, ds);
    alpha_p = std::min(1.0, alpha_p);
    alpha_d = std::min(1.0, alpha_d);

    for (int j = 0; j < n; ++j) {
      x[static_cast<std::size_t>(j)] += alpha_p * dx[static_cast<std::size_t>(j)];
      s[static_cast<std::size_t>(j)] += alpha_d * ds[static_cast<std::size_t>(j)];
      x[static_cast<std::size_t>(j)] = std::max(x[static_cast<std::size_t>(j)], 1e-14);
      s[static_cast<std::size_t>(j)] = std::max(s[static_cast<std::size_t>(j)], 1e-14);
    }
    for (int i = 0; i < m; ++i) {
      y[static_cast<std::size_t>(i)] += alpha_d * dy[static_cast<std::size_t>(i)];
    }

    result.iterations = it + 1;
  }

  // Iteration budget exhausted. Report the actual residuals so the caller can
  // see *why* it stopped, and deliberately do NOT attach the last iterate: it
  // is an interior point that has not met the gap test, so its objective is not
  // a valid answer. Returning it with has_objective_value=true is how a
  // suboptimal number escapes as if it were proven.
  {
    std::vector<double> Ax;
    lp.A.multiply(x, Ax);
    std::vector<double> rp = lp.b;
    for (int i = 0; i < m; ++i) rp[static_cast<std::size_t>(i)] -= Ax[static_cast<std::size_t>(i)];
    std::vector<double> Aty = matvec_At(lp.A, y);
    std::vector<double> rd = lp.c;
    for (int j = 0; j < n; ++j) {
      rd[static_cast<std::size_t>(j)] -=
          Aty[static_cast<std::size_t>(j)] + s[static_cast<std::size_t>(j)];
    }
    const double complementarity = dot(x, s);
    const double gap = std::abs(complementarity) / (1.0 + std::abs(dot(lp.c, x)));

    result.status = SolverStatus::IterationLimit;
    result.duality_gap = gap;
    result.primal_residual = max_abs(rp) / bnorm;
    result.dual_residual = max_abs(rd) / cnorm;
    std::ostringstream oss;
    oss << "IPM did not converge in " << opt.max_iterations
        << " iterations. Relative duality gap = " << gap
        << " (tolerance " << opt.optimality_tol << "), primal residual = "
        << result.primal_residual << ", dual residual = " << result.dual_residual
        << ". No solution returned; the caller should fall back to the simplex path.";
    result.message = oss.str();
  }
  return result;
}

}  // namespace

InteriorPointSolver::InteriorPointSolver(InteriorPointOptions options)
    : options_(std::move(options)) {}

SolverResult InteriorPointSolver::solve(const OptimizationModel& model) const {
  try {
    IpmLp lp = build_ipm_form(model, options_.enable_scaling);
    return solve_ipm(lp, options_, model);
  } catch (const std::exception& ex) {
    SolverResult r;
    r.status = SolverStatus::Error;
    r.message = std::string("IPM error: ") + ex.what();
    return r;
  }
}

}  // namespace sovereign
