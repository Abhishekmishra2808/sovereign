#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace sovereign {
namespace cuda {

// CUDA through the NVIDIA driver library (nvcuda.dll / libcuda.so.1), loaded at
// run time. No CUDA toolkit is needed to build the engine, and the same binary
// runs on machines without a GPU. Kernels ship as PTX and are compiled by the
// driver for whatever GPU is installed.
struct DeviceInfo {
  bool available = false;
  std::string name;
  int compute_major = 0;
  int compute_minor = 0;
  std::size_t memory_bytes = 0;
  int driver_version = 0;  // e.g. 12050 for CUDA 12.5
  // Why CUDA is unusable, or empty when available.
  std::string reason;
};

// Probes once per process: loads the driver, creates the context, compiles the
// kernels and checks them against the CPU on a small problem.
const DeviceInfo& device_info();

enum class Status { Ok, Singular, Error };

// Dense LU with partial pivoting of an n x n column-major matrix, in place.
// Produces the same layout and pivot choice as DenseLU: unit-lower L below the
// diagonal, U on and above it, and piv as the row permutation.
Status dense_lu(std::size_t n, std::vector<double>& a, std::vector<int>& piv);

// y = A x, or y = A^T x when transpose is set, for an nrows x ncols CSC matrix.
// The matrix stays on the device between calls with identical contents.
bool spmv_csc(std::size_t nrows, std::size_t ncols, const std::vector<int>& col_ptr,
              const std::vector<int>& row_idx, const std::vector<double>& values,
              const std::vector<double>& x, std::vector<double>& y, bool transpose);

}  // namespace cuda
}  // namespace sovereign
