#include "sovereign/gpu_spmv.hpp"

#if defined(SOVEREIGN_USE_CUDA)
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace sovereign {
namespace {

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t err = (call);                                             \
    if (err != cudaSuccess) {                                                   \
      throw std::runtime_error(std::string("CUDA error: ") +                     \
                               cudaGetErrorString(err));                       \
    }                                                                           \
  } while (0)

__global__ void spmv_csr_kernel(int num_rows, const int* row_ptr,
                                const int* col_idx, const double* values,
                                const double* x, double* y) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;
  double sum = 0.0;
  for (int i = row_ptr[row]; i < row_ptr[row + 1]; ++i) {
    sum += values[i] * x[col_idx[i]];
  }
  y[row] = sum;
}

// An LP's matrix is unchanged across its many interior-point SpMV calls.
// Retain one exact matrix copy and its device buffers per calling thread.
// Strong branching solves LPs on multiple threads, so a process-global cache
// would race and let one node overwrite another node's device matrix.
struct DeviceMatrixCache {
  std::size_t nrows = 0;
  std::size_t ncols = 0;
  std::vector<int> col_ptr;
  std::vector<int> row_idx;
  std::vector<double> values;
  int* d_row_ptr = nullptr;
  int* d_col_idx = nullptr;
  double* d_values = nullptr;
  double* d_x = nullptr;
  double* d_y = nullptr;

  ~DeviceMatrixCache() { clear(); }

  void clear() {
    if (d_row_ptr) cudaFree(d_row_ptr);
    if (d_col_idx) cudaFree(d_col_idx);
    if (d_values) cudaFree(d_values);
    if (d_x) cudaFree(d_x);
    if (d_y) cudaFree(d_y);
    d_row_ptr = nullptr;
    d_col_idx = nullptr;
    d_values = nullptr;
    d_x = nullptr;
    d_y = nullptr;
    nrows = ncols = 0;
    col_ptr.clear();
    row_idx.clear();
    values.clear();
  }

  bool matches(std::size_t rows, std::size_t cols,
               const std::vector<int>& ptr, const std::vector<int>& idx,
               const std::vector<double>& val) const {
    // Exact equality is intentional. Address-only caching is unsafe because
    // temporary node LPs can reuse an address with different coefficients.
    return d_values && rows == nrows && cols == ncols &&
           ptr == col_ptr && idx == row_idx && val == values;
  }

  void upload(std::size_t rows, std::size_t cols,
              const std::vector<int>& ptr, const std::vector<int>& idx,
              const std::vector<double>& val) {
    clear();
    nrows = rows;
    ncols = cols;
    col_ptr = ptr;
    row_idx = idx;
    values = val;
    const int nnz = ptr[cols];

    std::vector<int> row_ptr(rows + 1, 0);
    std::vector<int> col_idx(static_cast<std::size_t>(nnz));
    std::vector<double> csr_values(static_cast<std::size_t>(nnz));
    for (std::size_t j = 0; j < cols; ++j) {
      for (int p = ptr[j]; p < ptr[j + 1]; ++p) {
        ++row_ptr[static_cast<std::size_t>(idx[static_cast<std::size_t>(p)]) + 1];
      }
    }
    for (std::size_t i = 0; i < rows; ++i) row_ptr[i + 1] += row_ptr[i];
    std::vector<int> next = row_ptr;
    for (std::size_t j = 0; j < cols; ++j) {
      for (int p = ptr[j]; p < ptr[j + 1]; ++p) {
        const int dest = next[static_cast<std::size_t>(idx[static_cast<std::size_t>(p)])]++;
        col_idx[static_cast<std::size_t>(dest)] = static_cast<int>(j);
        csr_values[static_cast<std::size_t>(dest)] = val[static_cast<std::size_t>(p)];
      }
    }

    CUDA_CHECK(cudaMalloc(&d_row_ptr, (rows + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_col_idx, static_cast<std::size_t>(nnz) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_values, static_cast<std::size_t>(nnz) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_x, cols * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y, rows * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_row_ptr, row_ptr.data(), (rows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_col_idx, col_idx.data(),
                          static_cast<std::size_t>(nnz) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, csr_values.data(),
                          static_cast<std::size_t>(nnz) * sizeof(double),
                          cudaMemcpyHostToDevice));
  }
};

thread_local DeviceMatrixCache matrix_cache;

}  // namespace

bool spmv_csc_cuda(std::size_t nrows, std::size_t ncols,
                   const std::vector<int>& col_ptr,
                   const std::vector<int>& row_idx,
                   const std::vector<double>& values,
                   const std::vector<double>& x,
                   std::vector<double>& y) {
  try {
    DeviceMatrixCache& cache = matrix_cache;
    if (!cache.matches(nrows, ncols, col_ptr, row_idx, values)) {
      cache.upload(nrows, ncols, col_ptr, row_idx, values);
    }
    y.resize(nrows);
    CUDA_CHECK(cudaMemcpy(cache.d_x, x.data(), ncols * sizeof(double),
                          cudaMemcpyHostToDevice));
    const int threads = 256;
    const int blocks = (static_cast<int>(nrows) + threads - 1) / threads;
    spmv_csr_kernel<<<blocks, threads>>>(static_cast<int>(nrows),
                                          cache.d_row_ptr, cache.d_col_idx,
                                          cache.d_values, cache.d_x, cache.d_y);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(y.data(), cache.d_y, nrows * sizeof(double),
                          cudaMemcpyDeviceToHost));
    return true;
  } catch (const std::exception&) {
    matrix_cache.clear();
    return false;
  }
}

}  // namespace sovereign
#endif  // SOVEREIGN_USE_CUDA
