#include "sovereign/gpu_spmv.hpp"
#include "sovereign/sparse_matrix.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

namespace sovereign {
namespace benchmark {

/**
 * Generate a random sparse matrix in CSC format
 */
void generate_random_csc(std::size_t nrows, std::size_t ncols, double density,
                        std::vector<int>& col_ptr, std::vector<int>& row_idx,
                        std::vector<double>& values) {
  std::mt19937 gen(42);
  std::uniform_real_distribution<double> dist_val(-1.0, 1.0);
  std::uniform_int_distribution<int> dist_row(0, static_cast<int>(nrows) - 1);

  col_ptr.resize(ncols + 1);
  col_ptr[0] = 0;

  std::vector<std::vector<int>> rows_per_col(ncols);
  std::vector<std::vector<double>> vals_per_col(ncols);

  for (std::size_t j = 0; j < ncols; ++j) {
    int nnz_in_col = static_cast<int>(nrows * density);
    for (int i = 0; i < nnz_in_col; ++i) {
      int row = dist_row(gen);
      rows_per_col[j].push_back(row);
      vals_per_col[j].push_back(dist_val(gen));
    }
    col_ptr[j + 1] = col_ptr[j] + nnz_in_col;
  }

  int total_nnz = col_ptr[ncols];
  row_idx.resize(total_nnz);
  values.resize(total_nnz);

  int idx = 0;
  for (std::size_t j = 0; j < ncols; ++j) {
    for (std::size_t i = 0; i < rows_per_col[j].size(); ++i) {
      row_idx[idx] = rows_per_col[j][i];
      values[idx] = vals_per_col[j][i];
      ++idx;
    }
  }
}

/**
 * Benchmark SpMV performance
 */
void benchmark_spmv(std::size_t nrows, std::size_t ncols, double density, int iterations) {
  std::cout << "=== SpMV Benchmark ===" << std::endl;
  std::cout << "Matrix size: " << nrows << " x " << ncols << std::endl;
  std::cout << "Density: " << density << std::endl;
  std::cout << "Iterations: " << iterations << std::endl;
  std::cout << std::endl;

  // Generate random matrix
  std::vector<int> col_ptr, row_idx;
  std::vector<double> values;
  generate_random_csc(nrows, ncols, density, col_ptr, row_idx, values);

  std::vector<double> x(ncols);
  std::vector<double> y_cpu(nrows);
  std::vector<double> y_gpu(nrows);

  // Initialize x vector
  for (std::size_t i = 0; i < ncols; ++i) {
    x[i] = static_cast<double>(i) / ncols;
  }

  // CPU benchmark
  std::cout << "CPU SpMV..." << std::endl;
  auto cpu_start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    spmv_csc_cpu(nrows, ncols, col_ptr, row_idx, values, x, y_cpu);
  }
  auto cpu_end = std::chrono::high_resolution_clock::now();
  double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
  std::cout << "  Total time: " << cpu_time << " ms" << std::endl;
  std::cout << "  Avg time: " << (cpu_time / iterations) << " ms" << std::endl;

  // GPU benchmark
  if (gpu_available()) {
    std::cout << std::endl << "GPU SpMV..." << std::endl;
    auto gpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
      auto result = spmv_csc_auto(nrows, ncols, col_ptr, row_idx, values, x, y_gpu);
      if (!result.used_gpu) {
        throw std::runtime_error("GPU benchmark did not execute a CUDA kernel: " + result.message);
      }
    }
    auto gpu_end = std::chrono::high_resolution_clock::now();
    double gpu_time = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    std::cout << "  Total time: " << gpu_time << " ms" << std::endl;
    std::cout << "  Avg time: " << (gpu_time / iterations) << " ms" << std::endl;

    // Calculate speedup
    double speedup = cpu_time / gpu_time;
    std::cout << std::endl << "Speedup: " << speedup << "x" << std::endl;

    // Verify results match
    double max_diff = 0.0;
    for (std::size_t i = 0; i < nrows; ++i) {
      double diff = std::abs(y_cpu[i] - y_gpu[i]);
      max_diff = std::max(max_diff, diff);
    }
    std::cout << "Max difference: " << max_diff << std::endl;
    if (max_diff > 1e-8) throw std::runtime_error("CPU and GPU SpMV results differ.");
  } else {
    throw std::runtime_error("GPU not available for GPU benchmark.");
  }
}

void verify_cache_invalidation() {
  // Reuse the same host vectors but change both x and A between calls. A
  // pointer-only device cache would silently return the stale matrix result.
  std::vector<int> ptr{0, 2, 3};
  std::vector<int> rows{0, 1, 1};
  std::vector<double> values{1.0, 2.0, 3.0};
  std::vector<double> x{4.0, 5.0};
  std::vector<double> cpu, gpu;
  for (int pass = 0; pass < 3; ++pass) {
    spmv_csc_cpu(2, 2, ptr, rows, values, x, cpu);
    if (!spmv_csc_auto(2, 2, ptr, rows, values, x, gpu).used_gpu) {
      throw std::runtime_error("Cache check did not execute CUDA.");
    }
    for (std::size_t i = 0; i < cpu.size(); ++i) {
      if (std::abs(cpu[i] - gpu[i]) > 1e-12) {
        throw std::runtime_error("CUDA matrix cache returned stale data.");
      }
    }
    x[0] += 1.0;
    values[0] += 1.0;
  }
  std::cout << "CUDA cache invalidation: PASS" << std::endl;
}

} // namespace benchmark
} // namespace sovereign

int main() {
  using namespace sovereign::benchmark;
#if defined(_WIN32)
  _putenv_s("SOVEREIGN_DEVICE", "cuda");
#else
  setenv("SOVEREIGN_DEVICE", "cuda", 1);
#endif

  std::cout << "Sovereign GPU Benchmark" << std::endl;
  std::cout << "=======================" << std::endl;
  std::cout << std::endl;

  verify_cache_invalidation();
  std::cout << std::endl;

  // Small problem
  benchmark_spmv(1000, 1000, 0.01, 100);
  std::cout << std::endl;

  // Medium problem
  benchmark_spmv(10000, 10000, 0.001, 50);
  std::cout << std::endl;

  // Large problem
  benchmark_spmv(50000, 50000, 0.0005, 10);
  std::cout << std::endl;

  return 0;
}
