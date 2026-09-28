#include "mini_test.hpp"

#include "sovereign/cuda_driver.hpp"
#include "sovereign/dense_lu.hpp"
#include "sovereign/gpu_spmv.hpp"

#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace sovereign;

#if defined(_WIN32) && !defined(_MSC_VER)
// MinGW's stdlib.h hides this in strict ISO mode; every Windows C runtime exports it.
extern "C" int _putenv(const char*);
#endif

namespace {

void set_device(const char* device) {
#if defined(_WIN32)
  _putenv((std::string("SOVEREIGN_DEVICE=") + device).c_str());
#else
  setenv("SOVEREIGN_DEVICE", device, 1);
#endif
}

// Deterministic nonsymmetric matrix whose largest entries sit off the
// diagonal, so partial pivoting has to swap rows.
std::vector<double> test_matrix(std::size_t n) {
  std::vector<double> a(n * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      a[j * n + i] = std::sin(0.37 * static_cast<double>(i * n + j) + 1.0) +
                     ((i + 1) % n == j ? 3.0 : 0.0);
    }
  }
  return a;
}

std::vector<double> solve_with(const char* device, const std::vector<double>& a, std::size_t n,
                               bool& ok) {
  set_device(device);
  DenseLU lu;
  std::vector<double> x(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = 1.0 + static_cast<double>(i % 7);
  ok = lu.factorize(a, n) && lu.solve(x);
  set_device("cpu");
  return x;
}

}  // namespace

TEST(GpuBackendTest, DetectionExplainsItself) {
  const cuda::DeviceInfo& info = cuda::device_info();
  if (info.available) {
    EXPECT_FALSE(info.name.empty());
    EXPECT_TRUE(info.reason.empty());
  } else {
    EXPECT_FALSE(info.reason.empty());
  }
}

TEST(GpuBackendTest, DenseLuMatchesCpu) {
  if (!gpu_available()) return;
  const std::size_t n = 150;
  const std::vector<double> a = test_matrix(n);
  bool cpu_ok = false, gpu_ok = false;
  const std::vector<double> cpu = solve_with("cpu", a, n, cpu_ok);
  reset_gpu_operations();
  const std::vector<double> gpu = solve_with("cuda", a, n, gpu_ok);
  EXPECT_TRUE(cpu_ok);
  EXPECT_TRUE(gpu_ok);
  EXPECT_EQ(gpu_factorizations(), 1ULL);
  for (std::size_t i = 0; i < n; ++i) EXPECT_NEAR(gpu[i], cpu[i], 1e-9);
}

TEST(GpuBackendTest, DenseLuReportsSingular) {
  if (!gpu_available()) return;
  const std::size_t n = 40;
  std::vector<double> a = test_matrix(n);
  for (std::size_t i = 0; i < n; ++i) a[5 * n + i] = 2.0 * a[3 * n + i];
  bool ok = true;
  solve_with("cuda", a, n, ok);
  EXPECT_FALSE(ok);
}

TEST(GpuBackendTest, SparseProductsMatchCpu) {
  if (!gpu_available()) return;
  const std::size_t rows = 90, cols = 130;
  std::vector<int> col_ptr(1, 0), row_idx;
  std::vector<double> values;
  for (std::size_t j = 0; j < cols; ++j) {
    for (std::size_t i = j % 5; i < rows; i += 7 + j % 3) {
      row_idx.push_back(static_cast<int>(i));
      values.push_back(std::cos(static_cast<double>(i + 3 * j)));
    }
    col_ptr.push_back(static_cast<int>(values.size()));
  }
  std::vector<double> x(cols), z(rows);
  for (std::size_t j = 0; j < cols; ++j) x[j] = std::sin(static_cast<double>(j));
  for (std::size_t i = 0; i < rows; ++i) z[i] = 0.5 - static_cast<double>(i % 4);

  std::vector<double> ax_cpu, atz_cpu, ax_gpu, atz_gpu;
  spmv_csc_auto(rows, cols, col_ptr, row_idx, values, x, ax_cpu);
  spmv_csc_auto(rows, cols, col_ptr, row_idx, values, z, atz_cpu, true);
  set_device("cuda");
  EXPECT_TRUE(spmv_csc_auto(rows, cols, col_ptr, row_idx, values, x, ax_gpu).used_gpu);
  EXPECT_TRUE(spmv_csc_auto(rows, cols, col_ptr, row_idx, values, z, atz_gpu, true).used_gpu);
  set_device("cpu");
  ASSERT_EQ(ax_gpu.size(), rows);
  ASSERT_EQ(atz_gpu.size(), cols);
  for (std::size_t i = 0; i < rows; ++i) EXPECT_NEAR(ax_gpu[i], ax_cpu[i], 1e-12);
  for (std::size_t j = 0; j < cols; ++j) EXPECT_NEAR(atz_gpu[j], atz_cpu[j], 1e-12);
}
