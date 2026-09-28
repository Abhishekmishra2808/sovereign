#include "mini_test.hpp"

#include "sovereign/sparse_lu.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

using namespace sovereign;

namespace {

struct Csc {
  std::size_t n = 0;
  std::vector<int> ptr{0};
  std::vector<int> idx;
  std::vector<double> val;
  void add(int row, double v) {
    idx.push_back(row);
    val.push_back(v);
  }
  void end_column() { ptr.push_back(static_cast<int>(idx.size())); }
};

struct Lcg {
  std::uint64_t s;
  double uniform() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(s >> 11) / 9007199254740992.0;
  }
  int below(int n) { return static_cast<int>(uniform() * n) % n; }
};

double residual(const Csc& a, const std::vector<double>& x, const std::vector<double>& b,
                bool transpose) {
  std::vector<double> ax(a.n, 0.0);
  for (std::size_t j = 0; j < a.n; ++j) {
    for (int p = a.ptr[j]; p < a.ptr[j + 1]; ++p) {
      const std::size_t i = static_cast<std::size_t>(a.idx[static_cast<std::size_t>(p)]);
      if (transpose) ax[j] += a.val[static_cast<std::size_t>(p)] * x[i];
      else ax[i] += a.val[static_cast<std::size_t>(p)] * x[j];
    }
  }
  double r = 0.0;
  for (std::size_t i = 0; i < a.n; ++i) r = std::max(r, std::abs(ax[i] - b[i]));
  return r;
}

// Random nonsingular sparse matrix: a permuted, scaled diagonal plus noise, so
// the pivot order is neither the identity nor the natural column order.
Csc random_matrix(Lcg& rng, std::size_t n, double density) {
  std::vector<int> perm(n);
  for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<int>(i);
  for (std::size_t i = n; i > 1; --i) std::swap(perm[i - 1], perm[static_cast<std::size_t>(rng.below(static_cast<int>(i)))]);
  Csc a;
  a.n = n;
  for (std::size_t j = 0; j < n; ++j) {
    std::vector<double> col(n, 0.0);
    col[static_cast<std::size_t>(perm[j])] = (rng.uniform() < 0.5 ? -1.0 : 1.0) * (2.0 + 4.0 * rng.uniform());
    for (std::size_t i = 0; i < n; ++i) {
      if (rng.uniform() < density) col[i] += 2.0 * rng.uniform() - 1.0;
    }
    for (std::size_t i = 0; i < n; ++i) {
      if (col[i] != 0.0) a.add(static_cast<int>(i), col[i]);
    }
    a.end_column();
  }
  return a;
}

}  // namespace

TEST(SparseLuTest, RandomMatricesSolveBothWays) {
  Lcg rng{12345};
  for (int trial = 0; trial < 60; ++trial) {
    const std::size_t n = 1 + static_cast<std::size_t>(rng.below(80));
    const Csc a = random_matrix(rng, n, trial % 3 == 0 ? 0.3 : 0.05);
    SparseLU lu;
    EXPECT_TRUE(lu.factorize(a.n, a.ptr, a.idx, a.val));
    std::vector<double> b(n);
    for (double& v : b) v = 10.0 * rng.uniform() - 5.0;
    std::vector<double> x = b;
    EXPECT_TRUE(lu.solve(x));
    EXPECT_TRUE(residual(a, x, b, false) < 1e-9);
    std::vector<double> y = b;
    EXPECT_TRUE(lu.solve_transpose(y));
    EXPECT_TRUE(residual(a, y, b, true) < 1e-9);
  }
}

TEST(SparseLuTest, SimplexStyleBasisHasNoFill) {
  // Slack columns (-e_i) mixed with a few structural columns: the sparsest-first
  // order pivots every slack on its own row, so L and U hold only A's entries.
  const std::size_t n = 50;
  Csc a;
  a.n = n;
  for (std::size_t j = 0; j < n; ++j) {
    if (j % 5 == 0) {
      a.add(static_cast<int>(j), 2.0);
      if (j + 1 < n) a.add(static_cast<int>(j + 1), 1.0);
      if (j + 2 < n) a.add(static_cast<int>(j + 2), -1.0);
    } else {
      a.add(static_cast<int>(j), -1.0);
    }
    a.end_column();
  }
  SparseLU lu;
  EXPECT_TRUE(lu.factorize(a.n, a.ptr, a.idx, a.val));
  EXPECT_TRUE(lu.factor_nnz() <= a.idx.size() + n);
  std::vector<double> b(n, 1.0);
  std::vector<double> x = b;
  EXPECT_TRUE(lu.solve(x));
  EXPECT_TRUE(residual(a, x, b, false) < 1e-12);
  std::vector<double> y = b;
  EXPECT_TRUE(lu.solve_transpose(y));
  EXPECT_TRUE(residual(a, y, b, true) < 1e-12);
}

TEST(SparseLuTest, SingularMatricesAreRejected) {
  // Two identical columns.
  Csc dup;
  dup.n = 3;
  for (int j = 0; j < 3; ++j) {
    if (j < 2) {
      dup.add(0, 1.0);
      dup.add(1, 2.0);
    } else {
      dup.add(2, 1.0);
    }
    dup.end_column();
  }
  SparseLU lu;
  EXPECT_FALSE(lu.factorize(dup.n, dup.ptr, dup.idx, dup.val));
  EXPECT_FALSE(lu.ok());

  // An empty column.
  Csc empty;
  empty.n = 2;
  empty.add(0, 1.0);
  empty.end_column();
  empty.end_column();
  EXPECT_FALSE(lu.factorize(empty.n, empty.ptr, empty.idx, empty.val));
}

TEST(SparseLuTest, PivotsAwayFromZeroDiagonal) {
  // [[0, 1], [1, 0]] needs a row interchange.
  Csc a;
  a.n = 2;
  a.add(1, 1.0);
  a.end_column();
  a.add(0, 1.0);
  a.end_column();
  SparseLU lu;
  EXPECT_TRUE(lu.factorize(a.n, a.ptr, a.idx, a.val));
  std::vector<double> x = {3.0, 7.0};
  EXPECT_TRUE(lu.solve(x));
  EXPECT_NEAR(x[0], 7.0, 1e-12);
  EXPECT_NEAR(x[1], 3.0, 1e-12);
}
