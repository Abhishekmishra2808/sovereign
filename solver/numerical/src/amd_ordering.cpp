#include "sovereign/amd_ordering.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <vector>

namespace sovereign {

namespace {

// Simple degree-based ordering (not full AMD, but a correct fill-reducing heuristic)
// This is a simplified implementation suitable for first milestone.
// Full AMD with aggressive absorption and mass elimination can be added later.

struct Node {
  int degree;         // Current approximate degree
  int id;             // Original node id
  bool eliminated;    // Has this node been eliminated?
};

}  // namespace

bool AMDOrdering::compute(
    const SparseSymmetricPattern& pattern,
    std::vector<int>& perm,
    std::vector<int>& iperm
) {
  auto start = std::chrono::steady_clock::now();

  const int n = static_cast<int>(pattern.n);
  if (n == 0) {
    perm.clear();
    iperm.clear();
    return true;
  }

  perm.resize(static_cast<std::size_t>(n));
  iperm.resize(static_cast<std::size_t>(n));

  // Build adjacency lists from symmetric pattern
  // For each nonzero M[i,j] (i != j), add edge i <-> j
  std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));

  for (int j = 0; j < n; ++j) {
    const int start = pattern.col_ptr[static_cast<std::size_t>(j)];
    const int end = pattern.col_ptr[static_cast<std::size_t>(j) + 1];
    for (int p = start; p < end; ++p) {
      const int i = pattern.row_idx[static_cast<std::size_t>(p)];
      if (i != j) {
        adj[static_cast<std::size_t>(j)].push_back(i);
        adj[static_cast<std::size_t>(i)].push_back(j);
      }
    }
  }

  // Remove duplicates from adjacency lists
  for (int i = 0; i < n; ++i) {
    std::sort(adj[static_cast<std::size_t>(i)].begin(), adj[static_cast<std::size_t>(i)].end());
    auto last = std::unique(adj[static_cast<std::size_t>(i)].begin(), adj[static_cast<std::size_t>(i)].end());
    adj[static_cast<std::size_t>(i)].erase(last, adj[static_cast<std::size_t>(i)].end());
  }

  // Initialize nodes
  std::vector<Node> nodes(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    nodes[static_cast<std::size_t>(i)].degree = static_cast<int>(adj[static_cast<std::size_t>(i)].size());
    nodes[static_cast<std::size_t>(i)].id = i;
    nodes[static_cast<std::size_t>(i)].eliminated = false;
  }

  // Greedy elimination: repeatedly pick minimum degree node
  for (int k = 0; k < n; ++k) {
    // Find minimum degree node not yet eliminated
    int min_deg = n + 1;
    int min_node = -1;
    for (int i = 0; i < n; ++i) {
      if (!nodes[static_cast<std::size_t>(i)].eliminated &&
          nodes[static_cast<std::size_t>(i)].degree < min_deg) {
        min_deg = nodes[static_cast<std::size_t>(i)].degree;
        min_node = i;
      }
    }

    if (min_node < 0) break;  // No more nodes

    // Eliminate this node
    perm[static_cast<std::size_t>(k)] = min_node;
    iperm[static_cast<std::size_t>(min_node)] = k;
    nodes[static_cast<std::size_t>(min_node)].eliminated = true;

    // Update degrees of neighbors (simplified update)
    // Full AMD would form element and do aggressive absorption here
    const std::vector<int>& neighbors = adj[static_cast<std::size_t>(min_node)];

    for (int v : neighbors) {
      if (!nodes[static_cast<std::size_t>(v)].eliminated) {
        // Add edges between neighbors (clique formation)
        for (int w : neighbors) {
          if (w != v && !nodes[static_cast<std::size_t>(w)].eliminated) {
            // Check if edge v-w already exists
            auto& adj_v = adj[static_cast<std::size_t>(v)];
            if (std::find(adj_v.begin(), adj_v.end(), w) == adj_v.end()) {
              adj_v.push_back(w);
              nodes[static_cast<std::size_t>(v)].degree++;
            }
          }
        }
        // Remove min_node from v's adjacency (it's eliminated)
        auto& adj_v = adj[static_cast<std::size_t>(v)];
        adj_v.erase(std::remove(adj_v.begin(), adj_v.end(), min_node), adj_v.end());
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  ordering_time_ = std::chrono::duration<double>(end - start).count();

  return true;
}

}  // namespace sovereign
