#include "sovereign/gpu_spmv.hpp"
#include <atomic>
#include <cstdlib>
#include <stdexcept>

#if defined(SOVEREIGN_USE_CUDA)
#include <cuda_runtime.h>
// CUDA implementation is in cuda_spmv.cu
namespace sovereign { bool spmv_csc_cuda(std::size_t nrows, std::size_t ncols,
                          const std::vector<int>& col_ptr,
                          const std::vector<int>& row_idx,
                          const std::vector<double>& values,
                          const std::vector<double>& x,
                          std::vector<double>& y); }
#endif

namespace sovereign {
namespace { std::atomic<unsigned long long> operation_count{0}; }
unsigned long long gpu_operations() { return operation_count.load(); }
void reset_gpu_operations() { operation_count.store(0); }

void spmv_csc_cpu(std::size_t nrows, std::size_t ncols,
                  const std::vector<int>& col_ptr,
                  const std::vector<int>& row_idx,
                  const std::vector<double>& values,
                  const std::vector<double>& x,
                  std::vector<double>& y) {
  (void)ncols;
  y.assign(nrows, 0.0);
  for (std::size_t j = 0; j < ncols; ++j) {
    const double xj = x[j];
    if (xj == 0.0) continue;
    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      y[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(p)])] +=
          values[static_cast<std::size_t>(p)] * xj;
    }
  }
}

bool gpu_available() {
#if defined(SOVEREIGN_USE_CUDA)
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  return err == cudaSuccess && device_count > 0;
#else
  return false;
#endif
}

GpuSpmvResult spmv_csc_auto(std::size_t nrows, std::size_t ncols,
                            const std::vector<int>& col_ptr,
                            const std::vector<int>& row_idx,
                            const std::vector<double>& values,
                            const std::vector<double>& x,
                            std::vector<double>& y) {
  GpuSpmvResult r;
  const char* requested = std::getenv("SOVEREIGN_DEVICE");
  const std::string device = requested ? requested : "cpu";

#if defined(SOVEREIGN_USE_CUDA)
  // Try GPU first if available and problem is large enough
  if (device != "cpu" && gpu_available() && nrows > 0 && ncols > 0 &&
      !values.empty() && (device == "cuda" || values.size() >= 50000)) {
    if (spmv_csc_cuda(nrows, ncols, col_ptr, row_idx, values, x, y)) {
      r.used_gpu = true;
      ++operation_count;
      r.message = "GPU SpMV (CUDA)";
      return r;
    }
    if (device == "cuda") throw std::runtime_error("CUDA matrix operation failed; job was not silently run on CPU.");
  }
#endif

  // Fall back to CPU
  spmv_csc_cpu(nrows, ncols, col_ptr, row_idx, values, x, y);
  r.used_gpu = false;
  r.message = gpu_available()
                  ? "CPU SpMV (GPU available but not used for this problem size)"
                  : "CPU SpMV (GPU not enabled at build time)";
  return r;
}

}  // namespace sovereign
