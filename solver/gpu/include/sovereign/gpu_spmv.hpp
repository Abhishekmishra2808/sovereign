#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sovereign {

// Optional GPU acceleration. The CPU path is always available; CUDA is used
// through the NVIDIA driver at run time (see cuda_driver.hpp).
//
// SOVEREIGN_DEVICE selects the policy:
//   cpu  (default) never use the GPU;
//   cuda use the GPU for every supported operation and fail loudly if it breaks;
//   auto use the GPU only where the operation is large enough to benefit.
struct GpuSpmvResult {
  bool used_gpu = false;
  std::string message;
};

// y = A_csc * x on the CPU.
void spmv_csc_cpu(std::size_t nrows, std::size_t ncols,
                  const std::vector<int>& col_ptr,
                  const std::vector<int>& row_idx,
                  const std::vector<double>& values,
                  const std::vector<double>& x,
                  std::vector<double>& y);

// y = A x (or A^T x when transpose is set), on the GPU when the policy allows.
GpuSpmvResult spmv_csc_auto(std::size_t nrows, std::size_t ncols,
                            const std::vector<int>& col_ptr,
                            const std::vector<int>& row_idx,
                            const std::vector<double>& values,
                            const std::vector<double>& x,
                            std::vector<double>& y,
                            bool transpose = false);

// Dense LU (DenseLU's layout) on the GPU when the policy allows. Returns false
// when the caller should factorize on the CPU; otherwise `nonsingular` holds
// the outcome.
bool gpu_dense_lu(std::size_t n, std::vector<double>& a, std::vector<int>& piv,
                  bool& nonsingular);

std::string requested_device();
bool gpu_available();
// Per-process counts, used by the worker CLI to report actual GPU work.
unsigned long long gpu_operations();
unsigned long long gpu_factorizations();
void reset_gpu_operations();

}  // namespace sovereign
