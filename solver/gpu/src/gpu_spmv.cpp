#include "sovereign/gpu_spmv.hpp"

namespace sovereign {

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
  return true;  // actual device query would go here
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
  // Evidence-driven policy: without measured GPU benefit, always use CPU.
  spmv_csc_cpu(nrows, ncols, col_ptr, row_idx, values, x, y);
  r.used_gpu = false;
  r.message = gpu_available()
                  ? "CUDA build enabled but CPU SpMV retained (no proven speedup yet)."
                  : "CPU SpMV (GPU not enabled at build time).";
  return r;
}

}  // namespace sovereign
