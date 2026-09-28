#include "sovereign/amd_ordering.hpp"

#include <algorithm>
#include <chrono>
#include <set>
#include <utility>
#include <vector>

namespace sovereign {

bool AMDOrdering::compute(
    const SparseSymmetricPattern& pattern,
    std::vector<int>& perm,
    std::vector<int>& iperm
) {
  auto start = std::chrono::steady_clock::now();
  fill_limit_exceeded_ = false;
  predicted_factor_nnz_ = 0;

  const int n = static_cast<int>(pattern.n);
  perm.assign(static_cast<std::size_t>(n), -1);
  iperm.assign(static_cast<std::size_t>(n), -1);
  if (n == 0) {
    ordering_time_ = 0.0;
    return true;
  }

  std::vector<std::vector<int>> adj(static_cast<std::size_t>(n));
  for (int j = 0; j < n; ++j) {
    for (int p = pattern.col_ptr[static_cast<std::size_t>(j)]; p < pattern.col_ptr[static_cast<std::size_t>(j) + 1]; ++p) {
      const int i = pattern.row_idx[static_cast<std::size_t>(p)];
      if (i != j) {
        adj[static_cast<std::size_t>(j)].push_back(i);
        adj[static_cast<std::size_t>(i)].push_back(j);
      }
    }
  }
  for (auto& list : adj) {
    std::sort(list.begin(), list.end());
    list.erase(std::unique(list.begin(), list.end()), list.end());
  }

  std::set<std::pair<int, int>> queue;  // (degree, node)
  for (int i = 0; i < n; ++i) queue.insert({static_cast<int>(adj[static_cast<std::size_t>(i)].size()), i});

  std::vector<int> mark(static_cast<std::size_t>(n), -1);
  int stamp = 0;

  for (int k = 0; k < n; ++k) {
    const int node = queue.begin()->second;
    queue.erase(queue.begin());
    perm[static_cast<std::size_t>(k)] = node;
    iperm[static_cast<std::size_t>(node)] = k;

    std::vector<int> clique;
    clique.swap(adj[static_cast<std::size_t>(node)]);
    predicted_factor_nnz_ += clique.size();
    if (fill_limit_ > 0 && predicted_factor_nnz_ > fill_limit_) {
      fill_limit_exceeded_ = true;
      ordering_time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      return false;
    }

    // Eliminating node joins its remaining neighbours into a clique.
    for (int v : clique) {
      auto& list = adj[static_cast<std::size_t>(v)];
      queue.erase({static_cast<int>(list.size()), v});
      ++stamp;
      std::size_t keep = 0;
      for (int w : list) {
        if (w == node) continue;
        mark[static_cast<std::size_t>(w)] = stamp;
        list[keep++] = w;
      }
      list.resize(keep);
      mark[static_cast<std::size_t>(v)] = stamp;
      for (int w : clique) {
        if (mark[static_cast<std::size_t>(w)] != stamp) list.push_back(w);
      }
      queue.insert({static_cast<int>(list.size()), v});
    }
  }

  ordering_time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return true;
}

}  // namespace sovereign
