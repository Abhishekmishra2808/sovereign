#include "sovereign/gpu_spmv.hpp"

#include "sovereign/cuda_driver.hpp"

#include <atomic>
#include <cstdlib>
#include <stdexcept>

namespace sovereign {
namespace {
std::atomic<unsigned long long> operation_count{0};
std::atomic<unsigned long long> factorization_count{0};

std::size_t env_size(const char* name, std::size_t fallback) {
  const char* raw = std::getenv(name);
  if (!raw || !*raw) return fallback;
  const long long v = std::atoll(raw);
  return v > 0 ? static_cast<std::size_t>(v) : fallback;
}
}  // namespace

unsigned long long gpu_operations() { return operation_count.load(); }
unsigned long long gpu_factorizations() { return factorization_count.load(); }
void reset_gpu_operations() {
  operation_count.store(0);
  factorization_count.store(0);
}

std::string requested_device() {
  const char* requested = std::getenv("SOVEREIGN_DEVICE");
  return requested && *requested ? requested : "cpu";
}

bool gpu_available() { return cuda::device_info().available; }

void spmv_csc_cpu(std::size_t nrows, std::size_t ncols,
                  const std::vector<int>& col_ptr,
                  const std::vector<int>& row_idx,
                  const std::vector<double>& values,
                  const std::vector<double>& x,
                  std::vector<double>& y) {
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

namespace {
void spmv_csc_transpose_cpu(std::size_t ncols, const std::vector<int>& col_ptr,
                            const std::vector<int>& row_idx,
                            const std::vector<double>& values,
                            const std::vector<double>& x, std::vector<double>& y) {
  y.assign(ncols, 0.0);
  for (std::size_t j = 0; j < ncols; ++j) {
    double sum = 0.0;
    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      sum += values[static_cast<std::size_t>(p)] *
             x[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(p)])];
    }
    y[j] = sum;
  }
}
}  // namespace

GpuSpmvResult spmv_csc_auto(std::size_t nrows, std::size_t ncols,
                            const std::vector<int>& col_ptr,
                            const std::vector<int>& row_idx,
                            const std::vector<double>& values,
                            const std::vector<double>& x,
                            std::vector<double>& y,
                            bool transpose) {
  GpuSpmvResult r;
  const std::string device = requested_device();
  // A lone product moves its vectors over PCIe for little arithmetic, so auto
  // mode only sends matrices large enough to amortize the transfer.
  const bool wanted = device == "cuda" ||
      (device == "auto" && values.size() >= env_size("SOVEREIGN_GPU_SPMV_MIN_NNZ", 50000));
  if (wanted && nrows > 0 && ncols > 0 && !values.empty() && gpu_available()) {
    if (cuda::spmv_csc(nrows, ncols, col_ptr, row_idx, values, x, y, transpose)) {
      r.used_gpu = true;
      ++operation_count;
      r.message = "GPU SpMV (CUDA)";
      return r;
    }
    if (device == "cuda") throw std::runtime_error("CUDA matrix operation failed; job was not silently run on CPU.");
  }
  if (transpose) {
    spmv_csc_transpose_cpu(ncols, col_ptr, row_idx, values, x, y);
  } else {
    spmv_csc_cpu(nrows, ncols, col_ptr, row_idx, values, x, y);
  }
  // Only consult the driver when the GPU was wanted: probing it costs a CUDA
  // context, which CPU-only solves must not pay.
  r.message = !wanted ? "CPU SpMV" : gpu_available() ? "CPU SpMV (the GPU product failed for this matrix)"
                                                     : "CPU SpMV (" + cuda::device_info().reason + ")";
  return r;
}

bool gpu_dense_lu(std::size_t n, std::vector<double>& a, std::vector<int>& piv,
                  bool& nonsingular) {
  const std::string device = requested_device();
  // The GPU factorization is O(n^3) work for O(n^2) transfer. Measured on an
  // RTX 2050 against the CPU path it loses at 300 rows and wins from about 400.
  const bool wanted = n >= 2 && (device == "cuda" ||
      (device == "auto" && n >= env_size("SOVEREIGN_GPU_DENSE_MIN", 400)));
  if (!wanted || !gpu_available()) return false;
  const cuda::DeviceInfo& info = cuda::device_info();
  const bool fits = static_cast<double>(n) * static_cast<double>(n) * sizeof(double) <
                    0.8 * static_cast<double>(info.memory_bytes);
  const cuda::Status status = fits ? cuda::dense_lu(n, a, piv) : cuda::Status::Error;
  if (status == cuda::Status::Error) {
    if (device == "cuda") {
      throw std::runtime_error(fits ? "CUDA dense factorization failed; job was not silently run on CPU."
                                    : "The dense factorization does not fit in GPU memory.");
    }
    return false;
  }
  ++operation_count;
  ++factorization_count;
  nonsingular = status == cuda::Status::Ok;
  return true;
}

}  // namespace sovereign
