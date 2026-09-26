#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sovereign {

// Phase 10: optional GPU acceleration hooks. CPU path is always available.
struct GpuSpmvResult {
  bool used_gpu = false;
  std::string message;
};

// y = A_csc * x  (CPU reference; GPU used only if SOVEREIGN_USE_CUDA compiled and beneficial)
void spmv_csc_cpu(std::size_t nrows, std::size_t ncols,
                  const std::vector<int>& col_ptr,
                  const std::vector<int>& row_idx,
                  const std::vector<double>& values,
                  const std::vector<double>& x,
                  std::vector<double>& y);

GpuSpmvResult spmv_csc_auto(std::size_t nrows, std::size_t ncols,
                            const std::vector<int>& col_ptr,
                            const std::vector<int>& row_idx,
                            const std::vector<double>& values,
                            const std::vector<double>& x,
                            std::vector<double>& y);

bool gpu_available();

}  // namespace sovereign
