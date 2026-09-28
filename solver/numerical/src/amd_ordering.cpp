#include "sovereign/amd_ordering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
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

  // Dense nodes (a budget row touching every variable, say) are taken out of
  // the graph and ordered last, as in AMD: kept in, each elimination would
  // rewrite their adjacency lists and the ordering would cost O(n^2).
  const double dense_degree = std::max(16.0, 10.0 * std::sqrt(static_cast<double>(n)));
  std::vector<char> dense(static_cast<std::size_t>(n), 0);
  std::vector<std::pair<int, int>> dense_nodes;  // (degree, node)
  for (int i = 0; i < n; ++i) {
    const auto degree = adj[static_cast<std::size_t>(i)].size();
    if (static_cast<double>(degree) > dense_degree) {
      dense[static_cast<std::size_t>(i)] = 1;
      dense_nodes.push_back({static_cast<int>(degree), i});
    }
  }
  if (!dense_nodes.empty()) {
    for (int i = 0; i < n; ++i) {
      auto& list = adj[static_cast<std::size_t>(i)];
      if (dense[static_cast<std::size_t>(i)]) {
        std::vector<int>().swap(list);
        continue;
      }
      list.erase(std::remove_if(list.begin(), list.end(),
                                [&](int w) { return dense[static_cast<std::size_t>(w)] != 0; }),
                 list.end());
    }
    std::sort(dense_nodes.begin(), dense_nodes.end());
  }
  const int n_sparse = n - static_cast<int>(dense_nodes.size());

  std::set<std::pair<int, int>> queue;  // (degree, node)
  for (int i = 0; i < n; ++i) {
    if (!dense[static_cast<std::size_t>(i)]) queue.insert({static_cast<int>(adj[static_cast<std::size_t>(i)].size()), i});
  }

  // Among the sparse nodes, fill is exactly the clique sizes below (dense
  // nodes come later, so they create no fill between earlier nodes); the
  // dense rows add to that, so this count is a lower bound on nnz(L) and the
  // exact figure comes from the symbolic analysis.
  const std::size_t n_dense = dense_nodes.size();
  predicted_factor_nnz_ = n_dense * (n_dense - (n_dense > 0 ? 1 : 0)) / 2;

  std::vector<int> mark(static_cast<std::size_t>(n), -1);
  int stamp = 0;

  for (int k = n_sparse; k < n; ++k) {
    const int node = dense_nodes[static_cast<std::size_t>(k - n_sparse)].second;
    perm[static_cast<std::size_t>(k)] = node;
    iperm[static_cast<std::size_t>(node)] = k;
  }

  for (int k = 0; k < n_sparse; ++k) {
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
