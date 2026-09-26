#include "mini_test.hpp"

#include "sovereign/sparse_matrix.hpp"

using namespace sovereign;

TEST(SparseTest, MultiplyIdentityLike) {
  SparseMatrixCSC a;
  a.resize(2, 2);
  a.push_back(0, 1.0);
  a.finish_column(0);
  a.push_back(1, 2.0);
  a.finish_column(1);

  std::vector<double> x = {3.0, 4.0};
  std::vector<double> y;
  a.multiply(x, y);
  EXPECT_EQ(y.size(), 2u);
  EXPECT_NEAR(y[0], 3.0, 1e-12);
  EXPECT_NEAR(y[1], 8.0, 1e-12);
}

TEST(SparseTest, ExtractBasis) {
  SparseMatrixCSC a;
  a.resize(2, 3);
  a.push_back(0, 1.0);
  a.finish_column(0);
  a.push_back(1, 1.0);
  a.finish_column(1);
  a.push_back(0, 2.0);
  a.push_back(1, 3.0);
  a.finish_column(2);

  std::vector<int> basis = {0, 1};
  std::vector<double> dense;
  a.extract_dense_basis(basis, dense);
  EXPECT_EQ(dense.size(), 4u);
  EXPECT_NEAR(dense[0], 1.0, 1e-12);  // col0 row0
  EXPECT_NEAR(dense[1], 0.0, 1e-12);  // col0 row1
  EXPECT_NEAR(dense[2], 0.0, 1e-12);  // col1 row0
  EXPECT_NEAR(dense[3], 1.0, 1e-12);  // col1 row1
}
