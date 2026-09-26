#include "mini_test.hpp"

#include "sovereign/dense_lu.hpp"

using namespace sovereign;

TEST(DenseLuTest, SolveTwoByTwo) {
  // A = [[2,1],[0,3]] column-major: 2,0,1,3
  std::vector<double> a = {2.0, 0.0, 1.0, 3.0};
  DenseLU lu;
  EXPECT_TRUE(lu.factorize(a, 2));
  std::vector<double> x = {5.0, 6.0};  // b
  EXPECT_TRUE(lu.solve(x));
  // 2x+y=5, 3y=6 => y=2, x=1.5
  EXPECT_NEAR(x[0], 1.5, 1e-9);
  EXPECT_NEAR(x[1], 2.0, 1e-9);
}
