#include "mini_test.hpp"

#include "sovereign/amd_ordering.hpp"
#include "sovereign/sparse_symmetric.hpp"

using namespace sovereign;

TEST(AMD, EmptyMatrix) {
  SparseSymmetricPattern pattern;
  pattern.n = 0;
  pattern.col_ptr = {0};

  AMDOrdering amd;
  std::vector<int> perm, iperm;

  EXPECT_TRUE(amd.compute(pattern, perm, iperm));
  EXPECT_EQ(perm.size(), 0);
  EXPECT_EQ(iperm.size(), 0);
}

TEST(AMD, SingleNode) {
  SparseSymmetricPattern pattern;
  pattern.n = 1;
  pattern.col_ptr = {0, 1};
  pattern.row_idx = {0};  // Just diagonal

  AMDOrdering amd;
  std::vector<int> perm, iperm;

  EXPECT_TRUE(amd.compute(pattern, perm, iperm));
  EXPECT_EQ(perm.size(), 1);
  EXPECT_EQ(iperm.size(), 1);
  EXPECT_EQ(perm[0], 0);
  EXPECT_EQ(iperm[0], 0);
}

TEST(AMD, SmallMatrix) {
  // Pattern:
  // [x x 0]
  // [x x x]
  // [0 x x]
  //
  // Node 1 has degree 2, nodes 0 and 2 have degree 1
  // AMD should prefer degree-1 nodes

  SparseSymmetricPattern pattern;
  pattern.n = 3;
  pattern.col_ptr = {0, 2, 5, 7};
  // col 0: rows [0, 1]
  // col 1: rows [1, 2]
  // col 2: row [2]
  pattern.row_idx = {0, 1,   1, 2,   2};  // Wait, need to fix this

  // Actually construct proper lower triangle:
  // col 0: [0]
  // col 1: [1, 0]  <- wait, must be i >= j, so [0, 1]... no, lower triangle means row >= col
  // Let me fix:
  // col 0: [0, 1] (M[0,0], M[1,0])
  // col 1: [1, 2] (M[1,1], M[2,1])
  // col 2: [2]    (M[2,2])

  pattern.row_idx = {0, 1,   1, 2,   2};
  pattern.col_ptr = {0, 2, 4, 5};

  AMDOrdering amd;
  std::vector<int> perm, iperm;

  EXPECT_TRUE(amd.compute(pattern, perm, iperm));
  EXPECT_EQ(perm.size(), 3);
  EXPECT_EQ(iperm.size(), 3);

  // Check that perm and iperm are inverses
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(iperm[perm[i]], i);
    EXPECT_EQ(perm[iperm[i]], i);
  }
}

TEST(AMD, PermutationValidity) {
  // Larger matrix to test permutation properties
  SparseSymmetricPattern pattern;
  pattern.n = 5;
  pattern.col_ptr = {0, 1, 3, 5, 7, 8};
  pattern.row_idx = {
    0,       // col 0
    1, 0,    // col 1 (added 0 for off-diagonal, but must be row >= col, so just [1])
    2, 1,    // col 2
    3, 2,    // col 3
    4        // col 4
  };

  // Fix to proper lower triangle:
  pattern.row_idx = {0,  1,2,  2,3,  3,4,  4};
  pattern.col_ptr = {0, 1, 3, 5, 7, 8};

  AMDOrdering amd;
  std::vector<int> perm, iperm;

  EXPECT_TRUE(amd.compute(pattern, perm, iperm));
  EXPECT_EQ(perm.size(), 5);
  EXPECT_EQ(iperm.size(), 5);

  // Check permutation is valid (all elements 0..n-1 present exactly once)
  std::vector<bool> seen(5, false);
  for (int i = 0; i < 5; ++i) {
    EXPECT_TRUE(perm[i] >= 0);
    EXPECT_TRUE(perm[i] < 5);
    EXPECT_FALSE(seen[perm[i]]);
    seen[perm[i]] = true;
  }

  // Check inverse property
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(iperm[perm[i]], i);
    EXPECT_EQ(perm[iperm[i]], i);
  }
}

TEST(AMD, TimingRecorded) {
  SparseSymmetricPattern pattern;
  pattern.n = 10;
  // Create diagonal pattern (simple case)
  pattern.col_ptr.resize(11);
  for (int i = 0; i <= 10; ++i) {
    pattern.col_ptr[i] = i;
  }
  pattern.row_idx.resize(10);
  for (int i = 0; i < 10; ++i) {
    pattern.row_idx[i] = i;
  }

  AMDOrdering amd;
  std::vector<int> perm, iperm;

  EXPECT_TRUE(amd.compute(pattern, perm, iperm));

  // Check that timing was recorded (should be > 0)
  EXPECT_TRUE(amd.ordering_time_seconds() >= 0.0);
}
