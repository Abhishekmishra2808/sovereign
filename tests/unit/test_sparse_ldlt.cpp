#include "mini_test.hpp"

#include "sovereign/sparse_ldlt.hpp"
#include "sovereign/sparse_symmetric.hpp"
#include "sovereign/dense_lu.hpp"

#include <cmath>
#include <vector>

using namespace sovereign;

namespace {

// Helper: Compute M * x for a symmetric matrix stored in CSC lower triangle format
std::vector<double> multiply_symmetric(
    const SparseSymmetricPattern& pattern,
    const std::vector<double>& values,
    const std::vector<double>& x,
    double regularization = 0.0
) {
  const int n = static_cast<int>(pattern.n);
  std::vector<double> result(n, 0.0);

  // Process lower triangle
  for (int j = 0; j < n; ++j) {
    const int col_start = pattern.col_ptr[static_cast<std::size_t>(j)];
    const int col_end = pattern.col_ptr[static_cast<std::size_t>(j) + 1];

    for (int p = col_start; p < col_end; ++p) {
      const int i = pattern.row_idx[static_cast<std::size_t>(p)];
      const double val = values[static_cast<std::size_t>(p)];

      // M[i,j] contributes to result[i] from x[j]
      result[static_cast<std::size_t>(i)] += val * x[static_cast<std::size_t>(j)];

      // By symmetry, M[i,j] = M[j,i] contributes to result[j] from x[i]
      if (i != j) {
        result[static_cast<std::size_t>(j)] += val * x[static_cast<std::size_t>(i)];
      }
    }
  }

  // Add regularization: (M + λI) * x
  for (int i = 0; i < n; ++i) {
    result[static_cast<std::size_t>(i)] += regularization * x[static_cast<std::size_t>(i)];
  }

  return result;
}

// Helper: Compute ||v||_2
double vector_norm(const std::vector<double>& v) {
  double sum = 0.0;
  for (double val : v) {
    sum += val * val;
  }
  return std::sqrt(sum);
}

// Helper: Compute relative residual ||M x - b|| / (||M|| ||x|| + ||b||)
// For simplicity, approximate ||M|| by max diagonal entry
double compute_residual(
    const SparseSymmetricPattern& pattern,
    const std::vector<double>& values,
    const std::vector<double>& x,
    const std::vector<double>& b,
    double regularization = 0.0
) {
  std::vector<double> mx = multiply_symmetric(pattern, values, x, regularization);

  double residual_norm = 0.0;
  for (std::size_t i = 0; i < mx.size(); ++i) {
    double diff = mx[i] - b[i];
    residual_norm += diff * diff;
  }
  residual_norm = std::sqrt(residual_norm);

  double x_norm = vector_norm(x);
  double b_norm = vector_norm(b);

  // Approximate ||M|| by max absolute value
  double m_norm = 0.0;
  for (double val : values) {
    m_norm = std::max(m_norm, std::abs(val));
  }
  m_norm += regularization;

  double denom = m_norm * x_norm + b_norm;
  if (denom < 1e-30) denom = 1.0;

  return residual_norm / denom;
}

}  // namespace

TEST(SparseLDLT, EmptyMatrix) {
  SparseSymmetricPattern pattern;
  pattern.n = 0;
  pattern.col_ptr = {0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.ok());
  EXPECT_EQ(ldlt.n(), 0);
  EXPECT_EQ(ldlt.factor_nnz(), 0);
}

TEST(SparseLDLT, IdentityMatrix) {
  // Identity matrix: only diagonal entries
  // [1 0 0]
  // [0 1 0]
  // [0 0 1]
  // L should be empty (no off-diagonal), D = [1,1,1]

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 1, 2, 3};
  pattern.row_idx = {0, 1, 2};  // Only diagonals

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.ok());
  EXPECT_EQ(ldlt.n(), 3);
  EXPECT_EQ(ldlt.factor_nnz(), 0);  // No off-diagonal entries in L
  EXPECT_TRUE(ldlt.symbolic_time_seconds() >= 0.0);
}

TEST(SparseLDLT, DiagonalMatrix) {
  // Diagonal matrix with different values
  // [2 0 0 0]
  // [0 3 0 0]
  // [0 0 5 0]
  // [0 0 0 7]

  SparseSymmetricPattern pattern;
  pattern.n = 4;
  pattern.col_ptr = {0, 1, 2, 3, 4};
  pattern.row_idx = {0, 1, 2, 3};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.ok());
  EXPECT_EQ(ldlt.n(), 4);
  EXPECT_EQ(ldlt.factor_nnz(), 0);  // Pure diagonal -> no L fill
}

TEST(SparseLDLT, SmallSPDMatrix) {
  // Small SPD matrix:
  // [2  -1   0]
  // [-1  2  -1]
  // [0  -1   2]
  //
  // Lower triangle in CSC:
  // col 0: [0, 1]     -> M[0,0] = 2, M[1,0] = -1
  // col 1: [1, 2]     -> M[1,1] = 2, M[2,1] = -1
  // col 2: [2]        -> M[2,2] = 2

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 2, 4, 5};
  pattern.row_idx = {0, 1,  1, 2,  2};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.ok());
  EXPECT_EQ(ldlt.n(), 3);

  // This tridiagonal structure should have minimal fill
  // Exact nnz(L) depends on AMD ordering, but should be small
  std::size_t nnz_l = ldlt.factor_nnz();
  EXPECT_TRUE(nnz_l <= 3);  // At most 3 off-diagonal entries
}

TEST(SparseLDLT, MatrixWithStructuralZeros) {
  // Matrix with structural zeros:
  // [4  1  0  0]
  // [1  4  1  0]
  // [0  1  4  1]
  // [0  0  1  4]
  //
  // Tridiagonal pattern

  SparseSymmetricPattern pattern;
  pattern.n = 4;
  pattern.col_ptr = {0, 2, 4, 6, 7};
  pattern.row_idx = {
    0, 1,     // col 0: M[0,0], M[1,0]
    1, 2,     // col 1: M[1,1], M[2,1]
    2, 3,     // col 2: M[2,2], M[3,2]
    3         // col 3: M[3,3]
  };

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.ok());
  EXPECT_EQ(ldlt.n(), 4);

  // Check that symbolic analysis completed
  std::size_t nnz_l = ldlt.factor_nnz();
  EXPECT_TRUE(nnz_l > 0);  // Should have some fill
  EXPECT_TRUE(nnz_l <= 6);  // But not dense (dense would be 6 off-diagonals)
}

TEST(SparseLDLT, FillReductionComparison) {
  // Compare two orderings on a matrix where AMD should reduce fill
  // Arrow matrix:
  // [2  1  1  1]
  // [1  2  0  0]
  // [1  0  2  0]
  // [1  0  0  2]
  //
  // Natural ordering: node 0 has degree 3, eliminates last -> more fill
  // AMD ordering: should eliminate degree-1 nodes first -> less fill

  SparseSymmetricPattern pattern;
  pattern.n = 4;
  pattern.col_ptr = {0, 4, 5, 6, 7};
  pattern.row_idx = {
    0, 1, 2, 3,   // col 0: M[0,0], M[1,0], M[2,0], M[3,0]
    1,            // col 1: M[1,1]
    2,            // col 2: M[2,2]
    3             // col 3: M[3,3]
  };

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.ok());
  EXPECT_EQ(ldlt.n(), 4);

  // Input has 7 entries (4 diagonal + 3 off-diagonal in lower triangle)
  // AMD should eliminate degree-1 nodes (1, 2, 3) first, then node 0
  // This ordering produces ZERO fill - each degree-1 node is only connected to node 0
  std::size_t nnz_l = ldlt.factor_nnz();
  double fill = ldlt.fill_ratio();

  // Check that fill ratio is computed
  EXPECT_TRUE(fill >= 0.0);

  // AMD should produce minimal fill for this arrow matrix
  // In fact, optimal ordering gives zero fill
  EXPECT_TRUE(nnz_l <= 3);  // At most the 3 original off-diagonals
}

// ============================================================================
// Phase 4: Numerical Factorization Tests
// ============================================================================

TEST(SparseLDLT, NumericalIdentityMatrix) {
  // Identity matrix: M = I
  // Expected: L = I (no off-diagonals), D = I (all ones)

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 1, 2, 3};
  pattern.row_idx = {0, 1, 2};

  std::vector<double> values = {1.0, 1.0, 1.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  // Check diagnostics
  EXPECT_TRUE(ldlt.min_pivot() >= 0.99);
  EXPECT_TRUE(ldlt.min_pivot() <= 1.01);
  EXPECT_TRUE(ldlt.max_pivot() >= 0.99);
  EXPECT_TRUE(ldlt.max_pivot() <= 1.01);
  EXPECT_EQ(ldlt.regularization_used(), 0.0);
}

TEST(SparseLDLT, NumericalDiagonalMatrix) {
  // Diagonal matrix: M = diag(2, 3, 5, 7)
  // Expected: L has no off-diagonals, D = [2, 3, 5, 7]

  SparseSymmetricPattern pattern;
  pattern.n = 4;
  pattern.col_ptr = {0, 1, 2, 3, 4};
  pattern.row_idx = {0, 1, 2, 3};

  std::vector<double> values = {2.0, 3.0, 5.0, 7.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  EXPECT_TRUE(ldlt.min_pivot() >= 1.99);
  EXPECT_TRUE(ldlt.min_pivot() <= 2.01);
  EXPECT_TRUE(ldlt.max_pivot() >= 6.99);
  EXPECT_TRUE(ldlt.max_pivot() <= 7.01);
}

TEST(SparseLDLT, NumericalSmallKnownMatrix) {
  // Small 2x2 SPD matrix with known factorization
  // M = [4  2]
  //     [2  3]
  //
  // After AMD permutation, pivots may differ from natural ordering
  // Just verify factorization succeeds and pivots are positive

  SparseSymmetricPattern pattern;
  pattern.n = 2;
  pattern.col_ptr = {0, 2, 3};
  pattern.row_idx = {0, 1,  1};

  std::vector<double> values = {4.0, 2.0,  3.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  // Matrix is SPD, all pivots should be positive
  EXPECT_TRUE(ldlt.min_pivot() > 0.0);
  EXPECT_TRUE(ldlt.max_pivot() > 0.0);
  EXPECT_TRUE(ldlt.min_pivot() <= ldlt.max_pivot());
}

TEST(SparseLDLT, NumericalTridiagonalSPD) {
  // Tridiagonal SPD matrix:
  // M = [2  -1   0]
  //     [-1  2  -1]
  //     [0  -1   2]

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 2, 4, 5};
  pattern.row_idx = {0, 1,  1, 2,  2};

  std::vector<double> values = {2.0, -1.0,  2.0, -1.0,  2.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  // Matrix is SPD, factorization should succeed
  EXPECT_TRUE(ldlt.min_pivot() > 0.0);
  EXPECT_TRUE(ldlt.max_pivot() > 0.0);
  EXPECT_TRUE(ldlt.numeric_time_seconds() >= 0.0);
}

TEST(SparseLDLT, NumericalWithRegularization) {
  // Test regularization: M + λI
  // Start with near-singular diagonal matrix

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 1, 2, 3};
  pattern.row_idx = {0, 1, 2};

  std::vector<double> values = {1e-6, 1e-6, 1e-6};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));

  // Without regularization, should fail or have tiny pivots
  // With regularization, should succeed
  const double lambda = 1e-8;
  EXPECT_TRUE(ldlt.numeric_factor(values, lambda));

  // Pivots should be approximately lambda + 1e-6
  EXPECT_TRUE(ldlt.min_pivot() > lambda * 0.5);
  EXPECT_EQ(ldlt.regularization_used(), lambda);
}

TEST(SparseLDLT, RejectsNonPositiveDefinite) {
  // Matrix that is NOT positive definite
  // M = [1   2]
  //     [2   1]
  // Eigenvalues: 3 and -1 (not SPD!)

  SparseSymmetricPattern pattern;
  pattern.n = 2;
  pattern.col_ptr = {0, 2, 3};
  pattern.row_idx = {0, 1,  1};

  std::vector<double> values = {1.0, 2.0,  1.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));

  // Factorization should fail (negative or zero pivot)
  EXPECT_FALSE(ldlt.numeric_factor(values, 0.0));
}

TEST(SparseLDLT, ReconstructionTest) {
  // Verify that L D L^T reconstructs P M P^T accurately
  // Use a small tridiagonal matrix

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 2, 4, 5};
  pattern.row_idx = {0, 1,  1, 2,  2};

  std::vector<double> values = {4.0, -1.0,  4.0, -1.0,  4.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  // We can't easily extract L and D from the opaque class,
  // but we can verify properties:
  // 1. All pivots positive (SPD)
  // 2. Factorization succeeded
  // 3. Timing recorded

  EXPECT_TRUE(ldlt.min_pivot() > 0.0);
  EXPECT_TRUE(ldlt.max_pivot() > 0.0);
  EXPECT_TRUE(ldlt.numeric_time_seconds() >= 0.0);

  // Actual reconstruction will be done via solve test:
  // if solve(b) gives x, then we verify ||M x - b|| is small
}

// ============================================================================
// Phase 5: Sparse Triangular Solve Tests
// ============================================================================

TEST(SparseLDLT, SolveIdentity) {
  // M = I, solve I x = b, expect x = b

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 1, 2, 3};
  pattern.row_idx = {0, 1, 2};

  std::vector<double> values = {1.0, 1.0, 1.0};
  std::vector<double> b = {2.0, 3.0, 5.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  std::vector<double> x = b;
  EXPECT_TRUE(ldlt.solve(x));

  // For identity matrix, x should equal b
  EXPECT_TRUE(std::abs(x[0] - 2.0) < 1e-10);
  EXPECT_TRUE(std::abs(x[1] - 3.0) < 1e-10);
  EXPECT_TRUE(std::abs(x[2] - 5.0) < 1e-10);

  // Verify residual
  double residual = compute_residual(pattern, values, x, b);
  EXPECT_TRUE(residual < 1e-10);
}

TEST(SparseLDLT, SolveDiagonal) {
  // M = diag(2, 3, 5), analytical solution x[i] = b[i] / M[i,i]

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 1, 2, 3};
  pattern.row_idx = {0, 1, 2};

  std::vector<double> values = {2.0, 3.0, 5.0};
  std::vector<double> b = {4.0, 9.0, 15.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  std::vector<double> x = b;
  EXPECT_TRUE(ldlt.solve(x));

  // Expected: x = [2, 3, 3]
  EXPECT_TRUE(std::abs(x[0] - 2.0) < 1e-10);
  EXPECT_TRUE(std::abs(x[1] - 3.0) < 1e-10);
  EXPECT_TRUE(std::abs(x[2] - 3.0) < 1e-10);

  // Verify residual
  double residual = compute_residual(pattern, values, x, b);
  EXPECT_TRUE(residual < 1e-10);
}

TEST(SparseLDLT, SolveSmallSPD) {
  // Small 2x2 SPD matrix, compare against analytical solution
  // M = [4  2]
  //     [2  3]

  SparseSymmetricPattern pattern;
  pattern.n = 2;
  pattern.col_ptr = {0, 2, 3};
  pattern.row_idx = {0, 1,  1};

  std::vector<double> values = {4.0, 2.0,  3.0};
  std::vector<double> b = {10.0, 7.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  std::vector<double> x = b;
  EXPECT_TRUE(ldlt.solve(x));

  // Verify residual against ORIGINAL matrix
  double residual = compute_residual(pattern, values, x, b);
  EXPECT_TRUE(residual < 1e-10);

  // Also verify solution satisfies M x = b within tolerance
  std::vector<double> mx = multiply_symmetric(pattern, values, x);
  EXPECT_TRUE(std::abs(mx[0] - b[0]) < 1e-10);
  EXPECT_TRUE(std::abs(mx[1] - b[1]) < 1e-10);
}

TEST(SparseLDLT, SolveTridiagonalMultipleRHS) {
  // Tridiagonal SPD matrix with multiple RHS vectors
  // Factorization should be reused

  SparseSymmetricPattern pattern;
  pattern.n = 4;
  pattern.col_ptr = {0, 2, 4, 6, 7};
  pattern.row_idx = {0, 1,  1, 2,  2, 3,  3};

  std::vector<double> values = {4.0, -1.0,  4.0, -1.0,  4.0, -1.0,  4.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  // Solve for multiple RHS
  std::vector<std::vector<double>> rhs_vectors = {
    {1.0, 0.0, 0.0, 0.0},
    {0.0, 1.0, 0.0, 0.0},
    {1.0, 1.0, 1.0, 1.0},
    {1.0, 2.0, 3.0, 4.0}
  };

  for (const auto& b : rhs_vectors) {
    std::vector<double> x = b;
    EXPECT_TRUE(ldlt.solve(x));

    double residual = compute_residual(pattern, values, x, b);
    EXPECT_TRUE(residual < 1e-2);  // Reasonable tolerance for 4x4 system
  }
}

TEST(SparseLDLT, SolveWithRegularization) {
  // Solve (M + λI) x = b

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 2, 4, 5};
  pattern.row_idx = {0, 1,  1, 2,  2};

  std::vector<double> values = {2.0, -1.0,  2.0, -1.0,  2.0};
  std::vector<double> b = {1.0, 2.0, 3.0};

  const double lambda = 1e-8;

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, lambda));

  std::vector<double> x = b;
  EXPECT_TRUE(ldlt.solve(x));

  // Verify residual against regularized system
  double residual = compute_residual(pattern, values, x, b, lambda);
  EXPECT_TRUE(residual < 1e-2);
}

TEST(SparseLDLT, SolveAMDPermuted) {
  // Arrow matrix where AMD produces nontrivial permutation
  // [2  1  1  1]
  // [1  2  0  0]
  // [1  0  2  0]
  // [1  0  0  2]

  SparseSymmetricPattern pattern;
  pattern.n = 4;
  pattern.col_ptr = {0, 4, 5, 6, 7};
  pattern.row_idx = {0, 1, 2, 3,  1,  2,  3};

  std::vector<double> values = {2.0, 1.0, 1.0, 1.0,  2.0,  2.0,  2.0};
  std::vector<double> b = {5.0, 3.0, 3.0, 3.0};

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  std::vector<double> x = b;
  EXPECT_TRUE(ldlt.solve(x));

  // Verify residual - permutation must be handled correctly
  double residual = compute_residual(pattern, values, x, b);
  EXPECT_TRUE(residual < 1e-2);
}

TEST(SparseLDLT, SolveTiming) {
  // Verify solve timing is recorded

  SparseSymmetricPattern pattern;
  pattern.n = 10;
  pattern.col_ptr.resize(11);
  pattern.row_idx.resize(10);
  std::vector<double> values(10);

  for (int i = 0; i <= 10; ++i) {
    pattern.col_ptr[static_cast<std::size_t>(i)] = i;
  }
  for (int i = 0; i < 10; ++i) {
    pattern.row_idx[static_cast<std::size_t>(i)] = i;
    values[static_cast<std::size_t>(i)] = 2.0 + i * 0.1;
  }

  std::vector<double> b(10, 1.0);

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.numeric_factor(values, 0.0));

  std::vector<double> x = b;
  EXPECT_TRUE(ldlt.solve(x));

  EXPECT_TRUE(ldlt.solve_time_seconds() >= 0.0);

  double residual = compute_residual(pattern, values, x, b);
  EXPECT_TRUE(residual < 1e-9);
}

TEST(SparseLDLT, PredictedNonzeroCount) {
  // Verify that symbolic analysis predicts nnz(L) correctly
  // Use a simple banded matrix where we can predict fill

  // Banded matrix (bandwidth 1):
  // [2 -1  0  0  0]
  // [-1 2 -1  0  0]
  // [0 -1  2 -1  0]
  // [0  0 -1  2 -1]
  // [0  0  0 -1  2]

  SparseSymmetricPattern pattern;
  pattern.n = 5;
  pattern.col_ptr = {0, 2, 4, 6, 8, 9};
  pattern.row_idx = {
    0, 1,        // col 0: M[0,0], M[1,0]
    1, 2,        // col 1: M[1,1], M[2,1]
    2, 3,        // col 2: M[2,2], M[3,2]
    3, 4,        // col 3: M[3,3], M[4,3]
    4            // col 4: M[4,4]
  };

  SparseLDLT ldlt;
  EXPECT_TRUE(ldlt.symbolic_analyze(pattern));
  EXPECT_TRUE(ldlt.ok());
  EXPECT_EQ(ldlt.n(), 5);

  // Tridiagonal matrix: minimal fill expected
  std::size_t nnz_l = ldlt.factor_nnz();

  // For tridiagonal, L should match input structure (4 off-diagonals)
  EXPECT_TRUE(nnz_l <= 4);

  // Verify fill ratio is reasonable
  double fill = ldlt.fill_ratio();
  EXPECT_TRUE(fill <= 1.0);  // Fill ratio should not exceed 1.0 for tridiagonal
}
