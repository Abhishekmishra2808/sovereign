#include "mini_test.hpp"

#include "sovereign/sparse_symmetric.hpp"
#include "sovereign/sparse_matrix.hpp"

#include <cmath>

using namespace sovereign;

TEST(SparseSymmetric, PatternValidation) {
  SparseSymmetricPattern p;
  p.n = 3;
  p.col_ptr = {0, 1, 3, 5};
  p.row_idx = {0, 1, 2, 2};  // Not sorted in column 2

  EXPECT_FALSE(p.is_valid());  // Should fail: row 2 appears before checking sorted

  // Fix: sort column 2
  p.row_idx = {0, 1, 2, 2};  // Still duplicate
  // Actually make it valid
  p.row_idx = {0, 1, 2, 2};  // col 0: [0], col 1: [1,2], col 2: [2]

  // Correct pattern
  p.row_idx = {0, 1, 2, 2};
  p.col_ptr = {0, 1, 3, 4};
  EXPECT_TRUE(p.is_valid());
}

TEST(SparseSymmetric, SmallPatternConstruction) {
  // A = [1 0]
  //     [1 1]
  // (2x2 matrix)
  //
  // M = A D A^T (assume D = I for structure)
  // M = [1 1]  (lower triangle: M[0,0]=1, M[1,0]=1, M[1,1]=2)
  //     [1 2]

  SparseMatrixCSC A;
  A.nrows = 2;
  A.ncols = 2;
  A.col_ptr = {0, 2, 3};  // col 0: 2 entries, col 1: 1 entry
  A.row_idx = {0, 1, 1};  // col 0: rows [0,1], col 1: row [1]
  A.values = {1.0, 1.0, 1.0};

  SparseSymmetricPattern pattern = build_normal_eq_pattern(A);

  EXPECT_EQ(pattern.n, 2);
  EXPECT_TRUE(pattern.is_valid());

  // Expected pattern (lower triangle):
  // col 0: row 0 (diagonal), row 1
  // col 1: row 1 (diagonal)
  // Total: 3 entries

  EXPECT_EQ(pattern.nnz(), 3);

  // Check column 0
  EXPECT_EQ(pattern.col_ptr[0], 0);
  EXPECT_EQ(pattern.col_ptr[1], 2);  // 2 entries in column 0

  // Check column 1
  EXPECT_EQ(pattern.col_ptr[2], 3);  // 1 entry in column 1
}

TEST(SparseSymmetric, DiagonalAlwaysPresent) {
  // A = [1 0]
  //     [0 0]
  // Second row is all zeros

  SparseMatrixCSC A;
  A.nrows = 2;
  A.ncols = 2;
  A.col_ptr = {0, 1, 1};  // col 0: 1 entry, col 1: 0 entries
  A.row_idx = {0};
  A.values = {1.0};

  SparseSymmetricPattern pattern = build_normal_eq_pattern(A);

  // Even though A[1,:] is zero, diagonal M[1,1] should be in pattern
  // (needed for regularization)

  EXPECT_EQ(pattern.n, 2);

  // Column 0: row 0
  // Column 1: row 1 (diagonal only)
  EXPECT_EQ(pattern.nnz(), 2);

  // Verify M[1,1] is present
  EXPECT_EQ(pattern.col_ptr[1], 1);
  EXPECT_EQ(pattern.col_ptr[2], 2);
  EXPECT_EQ(pattern.row_idx[1], 1);
}

TEST(SparseSymmetric, ValueAssemblySmall) {
  // A = [1 0]
  //     [1 1]
  //
  // D = diag([2, 3])
  //
  // M = A D A^T
  //   = [1 0] [2 0] [1 1]
  //     [1 1] [0 3] [0 1]
  //   = [2 0] [1 1]
  //     [2 3] [0 1]
  //   = [2  2]
  //     [2  5]

  SparseMatrixCSC A;
  A.nrows = 2;
  A.ncols = 2;
  A.col_ptr = {0, 2, 3};
  A.row_idx = {0, 1, 1};
  A.values = {1.0, 1.0, 1.0};

  std::vector<double> d = {2.0, 3.0};

  SparseSymmetricPattern pattern = build_normal_eq_pattern(A);
  std::vector<double> values;

  build_normal_eq_values(A, d, pattern, values);

  EXPECT_EQ(values.size(), pattern.nnz());

  // Expected values (lower triangle):
  // M[0,0] = 2
  // M[1,0] = 2
  // M[1,1] = 5

  // Column 0: M[0,0], M[1,0]
  EXPECT_NEAR(values[0], 2.0, 1e-14);
  EXPECT_NEAR(values[1], 2.0, 1e-14);

  // Column 1: M[1,1]
  EXPECT_NEAR(values[2], 5.0, 1e-14);
}

TEST(SparseSymmetric, ValueAssemblyWithZeroD) {
  // Test that D[j] = 0 is handled (skip that column of A)

  SparseMatrixCSC A;
  A.nrows = 2;
  A.ncols = 2;
  A.col_ptr = {0, 2, 3};
  A.row_idx = {0, 1, 1};
  A.values = {1.0, 1.0, 1.0};

  std::vector<double> d = {2.0, 0.0};  // D[1] = 0

  SparseSymmetricPattern pattern = build_normal_eq_pattern(A);
  std::vector<double> values;

  build_normal_eq_values(A, d, pattern, values);

  // With D[1] = 0, only column 0 of A contributes
  // M = A[:,0] * 2.0 * A[:,0]^T
  //   = [1]   * 2 * [1 1]
  //     [1]
  //   = [2  2]
  //     [2  2]

  EXPECT_NEAR(values[0], 2.0, 1e-14);  // M[0,0]
  EXPECT_NEAR(values[1], 2.0, 1e-14);  // M[1,0]
  EXPECT_NEAR(values[2], 2.0, 1e-14);  // M[1,1]
}

TEST(SparseSymmetric, LargerMatrix) {
  // A = [1 0 0]
  //     [1 1 0]
  //     [0 1 1]
  //     [0 0 1]
  // (4x3 matrix, banded structure)

  SparseMatrixCSC A;
  A.nrows = 4;
  A.ncols = 3;
  A.col_ptr = {0, 2, 4, 6};
  A.row_idx = {0, 1,    1, 2,    2, 3};
  A.values = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};

  SparseSymmetricPattern pattern = build_normal_eq_pattern(A);

  EXPECT_EQ(pattern.n, 4);
  EXPECT_TRUE(pattern.is_valid());

  // M should have banded structure too
  // Check that nnz(M) is reasonable (not dense)
  EXPECT_TRUE(pattern.nnz() <= 10);  // 4x4 lower triangle = 10 max
  EXPECT_TRUE(pattern.nnz() >= 4);   // At least diagonal

  // Assemble with D = I
  std::vector<double> d = {1.0, 1.0, 1.0};
  std::vector<double> values;

  build_normal_eq_values(A, d, pattern, values);

  // All values should be non-negative (M is positive semi-definite)
  for (double v : values) {
    EXPECT_TRUE(v >= 0.0);
  }
}
