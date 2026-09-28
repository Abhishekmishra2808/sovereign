#include "sovereign/cuda_driver.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef LOAD_LIBRARY_SEARCH_SYSTEM32
#define LOAD_LIBRARY_SEARCH_SYSTEM32 0x00000800
#endif
#define SOVEREIGN_CUDAAPI __stdcall
#else
#include <dlfcn.h>
#define SOVEREIGN_CUDAAPI
#endif

namespace sovereign {
namespace cuda {
namespace {

// Driver API types, mirrored from cuda.h so no toolkit is needed to build.
typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
#if defined(_WIN64) || defined(__LP64__)
typedef unsigned long long CUdeviceptr;
#else
typedef unsigned int CUdeviceptr;
#endif
const CUresult CUDA_SUCCESS = 0;
const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75;
const int CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76;
const int CU_JIT_ERROR_LOG_BUFFER = 5;
const int CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6;

struct Driver {
  CUresult(SOVEREIGN_CUDAAPI* cuInit)(unsigned int);
  CUresult(SOVEREIGN_CUDAAPI* cuDriverGetVersion)(int*);
  CUresult(SOVEREIGN_CUDAAPI* cuDeviceGetCount)(int*);
  CUresult(SOVEREIGN_CUDAAPI* cuDeviceGet)(CUdevice*, int);
  CUresult(SOVEREIGN_CUDAAPI* cuDeviceGetName)(char*, int, CUdevice);
  CUresult(SOVEREIGN_CUDAAPI* cuDeviceGetAttribute)(int*, int, CUdevice);
  CUresult(SOVEREIGN_CUDAAPI* cuDeviceTotalMem)(std::size_t*, CUdevice);
  CUresult(SOVEREIGN_CUDAAPI* cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
  CUresult(SOVEREIGN_CUDAAPI* cuCtxSetCurrent)(CUcontext);
  CUresult(SOVEREIGN_CUDAAPI* cuCtxSynchronize)();
  CUresult(SOVEREIGN_CUDAAPI* cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, int*, void**);
  CUresult(SOVEREIGN_CUDAAPI* cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
  CUresult(SOVEREIGN_CUDAAPI* cuMemAlloc)(CUdeviceptr*, std::size_t);
  CUresult(SOVEREIGN_CUDAAPI* cuMemFree)(CUdeviceptr);
  CUresult(SOVEREIGN_CUDAAPI* cuMemcpyHtoD)(CUdeviceptr, const void*, std::size_t);
  CUresult(SOVEREIGN_CUDAAPI* cuMemcpyDtoH)(void*, CUdeviceptr, std::size_t);
  CUresult(SOVEREIGN_CUDAAPI* cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                                              unsigned int, unsigned int, unsigned int, unsigned int,
                                              CUstream, void**, void**);
  CUresult(SOVEREIGN_CUDAAPI* cuGetErrorName)(CUresult, const char**);
};

// Kernels in PTX. `$P` is the pointer type, `$AS` the address size and `$W`
// the index-to-byte-offset multiply, filled in for 32- or 64-bit processes.
// Dense matrices are column-major with element (i, j) at j * n + i.
const char* const kPtxTemplate = R"PTX(
.version 6.0
.target sm_50
.address_size $AS

.visible .entry row_dot(
  .param .u32 rd_rows, .param .$P rd_ptr, .param .$P rd_idx,
  .param .$P rd_val, .param .$P rd_x, .param .$P rd_y)
{
  .reg .pred %p<4>;
  .reg .b32 %r<12>;
  .reg .f64 %d<4>;
  .reg .$P %a<16>;
  ld.param.u32 %r1, [rd_rows];
  mov.u32 %r2, %ctaid.x;
  mov.u32 %r3, %ntid.x;
  mov.u32 %r4, %tid.x;
  mad.lo.u32 %r5, %r2, %r3, %r4;
  setp.ge.u32 %p1, %r5, %r1;
  @%p1 bra RD_DONE;
  ld.param.$P %a1, [rd_ptr];
  ld.param.$P %a2, [rd_idx];
  ld.param.$P %a3, [rd_val];
  ld.param.$P %a4, [rd_x];
  ld.param.$P %a5, [rd_y];
  cvta.to.global.$P %a1, %a1;
  cvta.to.global.$P %a2, %a2;
  cvta.to.global.$P %a3, %a3;
  cvta.to.global.$P %a4, %a4;
  cvta.to.global.$P %a5, %a5;
  $W %a6, %r5, 4;
  add.$P %a7, %a1, %a6;
  ld.global.u32 %r6, [%a7];
  ld.global.u32 %r7, [%a7+4];
  mov.f64 %d1, 0d0000000000000000;
  setp.ge.u32 %p2, %r6, %r7;
  @%p2 bra RD_STORE;
RD_LOOP:
  $W %a8, %r6, 4;
  add.$P %a9, %a2, %a8;
  ld.global.u32 %r8, [%a9];
  $W %a10, %r6, 8;
  add.$P %a11, %a3, %a10;
  ld.global.f64 %d2, [%a11];
  $W %a12, %r8, 8;
  add.$P %a13, %a4, %a12;
  ld.global.f64 %d3, [%a13];
  fma.rn.f64 %d1, %d2, %d3, %d1;
  add.u32 %r6, %r6, 1;
  setp.lt.u32 %p3, %r6, %r7;
  @%p3 bra RD_LOOP;
RD_STORE:
  $W %a14, %r5, 8;
  add.$P %a15, %a5, %a14;
  st.global.f64 [%a15], %d1;
RD_DONE:
  ret;
}

// One block. Finds the first row i >= k with the largest |a(i, k)|, the same
// choice as DenseLU, and flags the matrix singular when that is below 1e-14.
.visible .entry lu_pivot(
  .param .u32 lp_n, .param .u32 lp_k, .param .$P lp_a,
  .param .$P lp_piv, .param .$P lp_status)
{
  .shared .align 8 .b8 lp_sbest[2048];
  .shared .align 4 .b8 lp_sidx[1024];
  .reg .pred %p<8>;
  .reg .b32 %r<24>;
  .reg .f64 %d<8>;
  .reg .$P %a<12>;
  ld.param.u32 %r1, [lp_n];
  ld.param.u32 %r2, [lp_k];
  ld.param.$P %a1, [lp_a];
  ld.param.$P %a2, [lp_piv];
  ld.param.$P %a3, [lp_status];
  cvta.to.global.$P %a1, %a1;
  cvta.to.global.$P %a2, %a2;
  cvta.to.global.$P %a3, %a3;
  mov.u32 %r3, %tid.x;
  mov.u32 %r4, %ntid.x;
  ld.global.u32 %r5, [%a3];
  setp.ne.u32 %p1, %r5, 0;
  @%p1 bra LP_DONE;
  mul.lo.u32 %r6, %r2, %r1;
  mov.f64 %d1, 0dBFF0000000000000;
  mov.u32 %r7, %r1;
  add.u32 %r8, %r2, %r3;
LP_SCAN:
  setp.ge.u32 %p2, %r8, %r1;
  @%p2 bra LP_REDUCE;
  add.u32 %r9, %r6, %r8;
  $W %a4, %r9, 8;
  add.$P %a5, %a1, %a4;
  ld.global.f64 %d2, [%a5];
  abs.f64 %d2, %d2;
  setp.gt.f64 %p3, %d2, %d1;
  @%p3 mov.f64 %d1, %d2;
  @%p3 mov.u32 %r7, %r8;
  add.u32 %r8, %r8, %r4;
  bra LP_SCAN;
LP_REDUCE:
  mov.u32 %r10, lp_sbest;
  mov.u32 %r11, lp_sidx;
  shl.b32 %r12, %r3, 3;
  add.u32 %r13, %r10, %r12;
  shl.b32 %r14, %r3, 2;
  add.u32 %r15, %r11, %r14;
  st.shared.f64 [%r13], %d1;
  st.shared.u32 [%r15], %r7;
  bar.sync 0;
  shr.u32 %r16, %r4, 1;
LP_TREE:
  setp.eq.u32 %p4, %r16, 0;
  @%p4 bra LP_WRITE;
  setp.ge.u32 %p5, %r3, %r16;
  @%p5 bra LP_SYNC;
  shl.b32 %r17, %r16, 3;
  add.u32 %r18, %r13, %r17;
  ld.shared.f64 %d3, [%r18];
  shl.b32 %r19, %r16, 2;
  add.u32 %r20, %r15, %r19;
  ld.shared.u32 %r21, [%r20];
  ld.shared.f64 %d4, [%r13];
  ld.shared.u32 %r22, [%r15];
  setp.gt.f64 %p6, %d3, %d4;
  setp.eq.f64 %p7, %d3, %d4;
  setp.lt.and.u32 %p7, %r21, %r22, %p7;
  or.pred %p6, %p6, %p7;
  @%p6 st.shared.f64 [%r13], %d3;
  @%p6 st.shared.u32 [%r15], %r21;
LP_SYNC:
  bar.sync 0;
  shr.u32 %r16, %r16, 1;
  bra LP_TREE;
LP_WRITE:
  setp.ne.u32 %p1, %r3, 0;
  @%p1 bra LP_DONE;
  ld.shared.f64 %d5, [%r10];
  ld.shared.u32 %r23, [%r11];
  $W %a6, %r2, 4;
  add.$P %a7, %a2, %a6;
  st.global.u32 [%a7], %r23;
  setp.ge.f64 %p2, %d5, 0d3D06849B86A12B9B;
  setp.lt.and.u32 %p2, %r23, %r1, %p2;
  @%p2 bra LP_DONE;
  mov.u32 %r5, 1;
  st.global.u32 [%a3], %r5;
LP_DONE:
  ret;
}

// Swaps rows k and piv[k] across every column j.
.visible .entry lu_swap(
  .param .u32 ls_n, .param .u32 ls_k, .param .$P ls_a,
  .param .$P ls_piv, .param .$P ls_status)
{
  .reg .pred %p<4>;
  .reg .b32 %r<16>;
  .reg .f64 %d<4>;
  .reg .$P %a<12>;
  ld.param.u32 %r1, [ls_n];
  ld.param.u32 %r2, [ls_k];
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mov.u32 %r5, %tid.x;
  mad.lo.u32 %r6, %r3, %r4, %r5;
  setp.ge.u32 %p1, %r6, %r1;
  @%p1 bra LS_DONE;
  ld.param.$P %a1, [ls_a];
  ld.param.$P %a2, [ls_piv];
  ld.param.$P %a3, [ls_status];
  cvta.to.global.$P %a1, %a1;
  cvta.to.global.$P %a2, %a2;
  cvta.to.global.$P %a3, %a3;
  ld.global.u32 %r7, [%a3];
  setp.ne.u32 %p2, %r7, 0;
  @%p2 bra LS_DONE;
  $W %a4, %r2, 4;
  add.$P %a5, %a2, %a4;
  ld.global.u32 %r8, [%a5];
  setp.eq.u32 %p3, %r8, %r2;
  @%p3 bra LS_DONE;
  mul.lo.u32 %r9, %r6, %r1;
  add.u32 %r10, %r9, %r2;
  add.u32 %r11, %r9, %r8;
  $W %a6, %r10, 8;
  add.$P %a7, %a1, %a6;
  $W %a8, %r11, 8;
  add.$P %a9, %a1, %a8;
  ld.global.f64 %d1, [%a7];
  ld.global.f64 %d2, [%a9];
  st.global.f64 [%a7], %d2;
  st.global.f64 [%a9], %d1;
LS_DONE:
  ret;
}

// a(i, k) /= a(k, k) for i > k.
.visible .entry lu_scale(
  .param .u32 lc_n, .param .u32 lc_k, .param .$P lc_a, .param .$P lc_status)
{
  .reg .pred %p<4>;
  .reg .b32 %r<16>;
  .reg .f64 %d<4>;
  .reg .$P %a<8>;
  ld.param.u32 %r1, [lc_n];
  ld.param.u32 %r2, [lc_k];
  mov.u32 %r3, %ctaid.x;
  mov.u32 %r4, %ntid.x;
  mov.u32 %r5, %tid.x;
  mad.lo.u32 %r6, %r3, %r4, %r5;
  add.u32 %r7, %r2, 1;
  add.u32 %r7, %r7, %r6;
  setp.ge.u32 %p1, %r7, %r1;
  @%p1 bra LC_DONE;
  ld.param.$P %a1, [lc_a];
  ld.param.$P %a2, [lc_status];
  cvta.to.global.$P %a1, %a1;
  cvta.to.global.$P %a2, %a2;
  ld.global.u32 %r8, [%a2];
  setp.ne.u32 %p2, %r8, 0;
  @%p2 bra LC_DONE;
  mul.lo.u32 %r9, %r2, %r1;
  add.u32 %r10, %r9, %r2;
  add.u32 %r11, %r9, %r7;
  $W %a3, %r10, 8;
  add.$P %a4, %a1, %a3;
  $W %a5, %r11, 8;
  add.$P %a6, %a1, %a5;
  ld.global.f64 %d1, [%a4];
  ld.global.f64 %d2, [%a6];
  div.rn.f64 %d2, %d2, %d1;
  st.global.f64 [%a6], %d2;
LC_DONE:
  ret;
}

// a(i, j) -= a(i, k) * a(k, j) for i > k and j > k. x runs down rows so
// neighbouring threads touch neighbouring memory.
.visible .entry lu_update(
  .param .u32 lu_n, .param .u32 lu_k, .param .$P lu_a, .param .$P lu_status)
{
  .reg .pred %p<4>;
  .reg .b32 %r<20>;
  .reg .f64 %d<6>;
  .reg .$P %a<10>;
  ld.param.u32 %r1, [lu_n];
  ld.param.u32 %r2, [lu_k];
  add.u32 %r3, %r2, 1;
  mov.u32 %r4, %ctaid.x;
  mov.u32 %r5, %ntid.x;
  mov.u32 %r6, %tid.x;
  mad.lo.u32 %r7, %r4, %r5, %r6;
  add.u32 %r7, %r7, %r3;
  mov.u32 %r8, %ctaid.y;
  mov.u32 %r9, %ntid.y;
  mov.u32 %r10, %tid.y;
  mad.lo.u32 %r11, %r8, %r9, %r10;
  add.u32 %r11, %r11, %r3;
  setp.ge.u32 %p1, %r7, %r1;
  setp.ge.or.u32 %p1, %r11, %r1, %p1;
  @%p1 bra LU_DONE;
  ld.param.$P %a1, [lu_a];
  ld.param.$P %a2, [lu_status];
  cvta.to.global.$P %a1, %a1;
  cvta.to.global.$P %a2, %a2;
  ld.global.u32 %r12, [%a2];
  setp.ne.u32 %p2, %r12, 0;
  @%p2 bra LU_DONE;
  mul.lo.u32 %r13, %r2, %r1;
  add.u32 %r14, %r13, %r7;
  mul.lo.u32 %r15, %r11, %r1;
  add.u32 %r16, %r15, %r2;
  add.u32 %r17, %r15, %r7;
  $W %a3, %r14, 8;
  add.$P %a4, %a1, %a3;
  $W %a5, %r16, 8;
  add.$P %a6, %a1, %a5;
  $W %a7, %r17, 8;
  add.$P %a8, %a1, %a7;
  ld.global.f64 %d1, [%a4];
  ld.global.f64 %d2, [%a6];
  ld.global.f64 %d3, [%a8];
  neg.f64 %d1, %d1;
  fma.rn.f64 %d3, %d1, %d2, %d3;
  st.global.f64 [%a8], %d3;
LU_DONE:
  ret;
}
)PTX";

std::string replace_all(std::string text, const std::string& from, const std::string& to) {
  std::size_t pos = 0;
  while ((pos = text.find(from, pos)) != std::string::npos) {
    text.replace(pos, from.size(), to);
    pos += to.size();
  }
  return text;
}

std::string ptx_source() {
  const bool wide = sizeof(CUdeviceptr) == 8;
  std::string ptx = replace_all(kPtxTemplate, "$AS", wide ? "64" : "32");
  ptx = replace_all(ptx, "$W", wide ? "mul.wide.u32" : "mul.lo.u32");
  return replace_all(ptx, "$P", wide ? "u64" : "u32");
}

struct Backend {
  Driver f{};
  CUcontext ctx = nullptr;
  CUfunction row_dot = nullptr;
  CUfunction lu_pivot = nullptr;
  CUfunction lu_swap = nullptr;
  CUfunction lu_scale = nullptr;
  CUfunction lu_update = nullptr;
  DeviceInfo info;
};

Backend& backend() {
  static Backend b;
  return b;
}

std::string error_text(const Driver& f, CUresult code) {
  const char* name = nullptr;
  if (f.cuGetErrorName && f.cuGetErrorName(code, &name) == CUDA_SUCCESS && name) return name;
  return "CUDA error " + std::to_string(code);
}

void* load_driver() {
#if defined(_WIN32)
  return reinterpret_cast<void*>(LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32));
#else
  void* handle = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  return handle ? handle : dlopen("libcuda.so", RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* lib, const char* name) {
#if defined(_WIN32)
  return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(lib), name));
#else
  return dlsym(lib, name);
#endif
}

template <typename Fn>
bool bind(void* lib, const char* name, Fn& out, std::string& missing) {
  void* p = symbol(lib, name);
  if (!p) {
    missing = name;
    return false;
  }
  out = reinterpret_cast<Fn>(p);
  return true;
}

bool launch(CUfunction fn, unsigned gx, unsigned gy, unsigned bx, unsigned by, void** args) {
  return backend().f.cuLaunchKernel(fn, gx, gy, 1, bx, by, 1, 0, nullptr, args, nullptr) == CUDA_SUCCESS;
}

unsigned blocks_for(std::size_t count, unsigned per_block) {
  return static_cast<unsigned>((count + per_block - 1) / per_block);
}

// Frees device buffers when a call returns, including on early exits.
struct DeviceBuffers {
  std::vector<CUdeviceptr> ptrs;
  ~DeviceBuffers() {
    for (CUdeviceptr p : ptrs) backend().f.cuMemFree(p);
  }
  bool alloc(CUdeviceptr& out, std::size_t bytes) {
    if (backend().f.cuMemAlloc(&out, bytes == 0 ? 1 : bytes) != CUDA_SUCCESS) return false;
    ptrs.push_back(out);
    return true;
  }
};

std::size_t max_dense_order() {
  // Kernels index elements with 32-bit integers; 32-bit processes also use
  // 32-bit byte offsets.
  return sizeof(CUdeviceptr) == 8 ? 65535 : 20000;
}

Status run_dense_lu(std::size_t n, std::vector<double>& a, std::vector<int>& piv) {
  Backend& b = backend();
  if (a.size() != n * n || n == 0 || n > max_dense_order()) return Status::Error;
  if (b.f.cuCtxSetCurrent(b.ctx) != CUDA_SUCCESS) return Status::Error;
  DeviceBuffers buf;
  CUdeviceptr d_a = 0, d_piv = 0, d_status = 0;
  if (!buf.alloc(d_a, n * n * sizeof(double)) || !buf.alloc(d_piv, n * sizeof(int)) ||
      !buf.alloc(d_status, sizeof(unsigned))) {
    return Status::Error;
  }
  const unsigned zero = 0;
  if (b.f.cuMemcpyHtoD(d_a, a.data(), n * n * sizeof(double)) != CUDA_SUCCESS ||
      b.f.cuMemcpyHtoD(d_status, &zero, sizeof(zero)) != CUDA_SUCCESS) {
    return Status::Error;
  }
  unsigned un = static_cast<unsigned>(n);
  for (unsigned k = 0; k < un; ++k) {
    void* pivot_args[] = {&un, &k, &d_a, &d_piv, &d_status};
    if (!launch(b.lu_pivot, 1, 1, 256, 1, pivot_args)) return Status::Error;
    if (!launch(b.lu_swap, blocks_for(n, 256), 1, 256, 1, pivot_args)) return Status::Error;
    const std::size_t rest = n - k - 1;
    if (rest == 0) continue;
    void* args[] = {&un, &k, &d_a, &d_status};
    if (!launch(b.lu_scale, blocks_for(rest, 256), 1, 256, 1, args)) return Status::Error;
    if (!launch(b.lu_update, blocks_for(rest, 32), blocks_for(rest, 8), 32, 8, args)) return Status::Error;
  }
  if (b.f.cuCtxSynchronize() != CUDA_SUCCESS) return Status::Error;
  unsigned status = 0;
  if (b.f.cuMemcpyDtoH(&status, d_status, sizeof(status)) != CUDA_SUCCESS) return Status::Error;
  if (status != 0) return Status::Singular;
  std::vector<int> pivot_rows(n);
  if (b.f.cuMemcpyDtoH(a.data(), d_a, n * n * sizeof(double)) != CUDA_SUCCESS ||
      b.f.cuMemcpyDtoH(pivot_rows.data(), d_piv, n * sizeof(int)) != CUDA_SUCCESS) {
    return Status::Error;
  }
  piv.resize(n);
  for (std::size_t i = 0; i < n; ++i) piv[i] = static_cast<int>(i);
  for (std::size_t k = 0; k < n; ++k) {
    std::swap(piv[k], piv[static_cast<std::size_t>(pivot_rows[k])]);
  }
  return Status::Ok;
}

// An LP's matrix is unchanged across its many interior-point products, so
// each thread keeps one exact copy on the device per direction. Comparing
// contents, not addresses, because temporary node LPs can reuse an address.
// The device copy is laid out by output entry: CSR of A for y = A x, and the
// CSC arrays themselves for y = A^T x.
struct MatrixCache {
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<int> col_ptr;
  std::vector<int> row_idx;
  std::vector<double> values;
  std::size_t out_size = 0;
  CUdeviceptr d_ptr = 0, d_idx = 0, d_val = 0, d_x = 0, d_y = 0;

  void clear() {
    Driver& f = backend().f;
    CUdeviceptr* all[] = {&d_ptr, &d_idx, &d_val, &d_x, &d_y};
    for (CUdeviceptr* p : all) {
      if (*p) f.cuMemFree(*p);
      *p = 0;
    }
    rows = cols = out_size = 0;
    col_ptr.clear();
    row_idx.clear();
    values.clear();
  }
  bool matches(std::size_t r, std::size_t c, const std::vector<int>& p,
               const std::vector<int>& i, const std::vector<double>& v) const {
    return d_val && r == rows && c == cols && p == col_ptr && i == row_idx && v == values;
  }
  bool upload(std::size_t r, std::size_t c, const std::vector<int>& p,
              const std::vector<int>& i, const std::vector<double>& v, bool transpose) {
    clear();
    Driver& f = backend().f;
    const std::size_t nnz = v.size();
    std::vector<int> ptr, idx;
    std::vector<double> val;
    if (transpose) {
      ptr = p;
      idx = i;
      val = v;
    } else {
      ptr.assign(r + 1, 0);
      idx.resize(nnz);
      val.resize(nnz);
      for (std::size_t k = 0; k < nnz; ++k) ++ptr[static_cast<std::size_t>(i[k]) + 1];
      for (std::size_t row = 0; row < r; ++row) ptr[row + 1] += ptr[row];
      std::vector<int> next(ptr.begin(), ptr.end() - 1);
      for (std::size_t col = 0; col < c; ++col) {
        for (int k = p[col]; k < p[col + 1]; ++k) {
          const int dest = next[static_cast<std::size_t>(i[static_cast<std::size_t>(k)])]++;
          idx[static_cast<std::size_t>(dest)] = static_cast<int>(col);
          val[static_cast<std::size_t>(dest)] = v[static_cast<std::size_t>(k)];
        }
      }
    }
    const std::size_t out = transpose ? c : r;
    const std::size_t in = transpose ? r : c;
    auto alloc = [&](CUdeviceptr& dst, std::size_t bytes) {
      return f.cuMemAlloc(&dst, bytes == 0 ? 1 : bytes) == CUDA_SUCCESS;
    };
    if (!alloc(d_ptr, (out + 1) * sizeof(int)) || !alloc(d_idx, nnz * sizeof(int)) ||
        !alloc(d_val, nnz * sizeof(double)) || !alloc(d_x, in * sizeof(double)) ||
        !alloc(d_y, out * sizeof(double)) ||
        f.cuMemcpyHtoD(d_ptr, ptr.data(), (out + 1) * sizeof(int)) != CUDA_SUCCESS ||
        (nnz && f.cuMemcpyHtoD(d_idx, idx.data(), nnz * sizeof(int)) != CUDA_SUCCESS) ||
        (nnz && f.cuMemcpyHtoD(d_val, val.data(), nnz * sizeof(double)) != CUDA_SUCCESS)) {
      clear();
      return false;
    }
    rows = r;
    cols = c;
    out_size = out;
    col_ptr = p;
    row_idx = i;
    values = v;
    return true;
  }
};

bool run_spmv(std::size_t nrows, std::size_t ncols, const std::vector<int>& col_ptr,
              const std::vector<int>& row_idx, const std::vector<double>& values,
              const std::vector<double>& x, std::vector<double>& y, bool transpose) {
  Backend& b = backend();
  const std::size_t in = transpose ? nrows : ncols;
  const std::size_t out = transpose ? ncols : nrows;
  if (out == 0 || col_ptr.size() != ncols + 1 || x.size() != in) return false;
  if (b.f.cuCtxSetCurrent(b.ctx) != CUDA_SUCCESS) return false;
  // Never destroyed: at process exit the driver may already be unloaded, and
  // freeing device memory then crashes. The driver reclaims it with the process.
  static thread_local MatrixCache* caches = new MatrixCache[2];
  MatrixCache& cache = caches[transpose ? 1 : 0];
  if (!cache.matches(nrows, ncols, col_ptr, row_idx, values) &&
      !cache.upload(nrows, ncols, col_ptr, row_idx, values, transpose)) {
    return false;
  }
  if (in && b.f.cuMemcpyHtoD(cache.d_x, x.data(), in * sizeof(double)) != CUDA_SUCCESS) {
    cache.clear();
    return false;
  }
  unsigned count = static_cast<unsigned>(out);
  void* args[] = {&count, &cache.d_ptr, &cache.d_idx, &cache.d_val, &cache.d_x, &cache.d_y};
  y.resize(out);
  if (!launch(b.row_dot, blocks_for(out, 256), 1, 256, 1, args) ||
      b.f.cuMemcpyDtoH(y.data(), cache.d_y, out * sizeof(double)) != CUDA_SUCCESS) {
    cache.clear();
    return false;
  }
  return true;
}

// Checks the kernels on a small problem so "available" means results are right.
std::string self_test() {
  const std::size_t n = 5;
  std::vector<double> a(n * n);
  for (std::size_t j = 0; j < n; ++j) {
    for (std::size_t i = 0; i < n; ++i) {
      a[j * n + i] = std::sin(1.0 + 3.0 * i + 7.0 * j) + (i == j ? 0.5 : 0.0);
    }
  }
  std::vector<double> lu = a;
  std::vector<int> piv;
  if (run_dense_lu(n, lu, piv) != Status::Ok) return "dense LU self-test did not run";
  // Rebuild P*A from L*U and compare with the original rows.
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < n; ++j) {
      double sum = 0.0;
      for (std::size_t k = 0; k <= std::min(i, j); ++k) {
        const double l = k == i ? 1.0 : lu[k * n + i];
        sum += l * lu[j * n + k];
      }
      if (std::abs(sum - a[j * n + static_cast<std::size_t>(piv[i])]) > 1e-9) {
        return "dense LU self-test gave a wrong factorization";
      }
    }
  }
  // A = [[2, 0, 0.5], [0, 4, 3]] in CSC, a 2 x 3 matrix.
  const std::vector<int> ptr = {0, 1, 2, 4};
  const std::vector<int> idx = {0, 1, 0, 1};
  const std::vector<double> val = {2.0, 4.0, 0.5, 3.0};
  const std::vector<double> x = {1.0, 2.0, 3.0};
  const std::vector<double> z = {1.0, -1.0};
  std::vector<double> y, w;
  if (!run_spmv(2, 3, ptr, idx, val, x, y, false) || !run_spmv(2, 3, ptr, idx, val, z, w, true)) {
    return "sparse product self-test did not run";
  }
  const double expected_y[] = {3.5, 17.0};
  const double expected_w[] = {2.0, -4.0, -2.5};
  for (std::size_t i = 0; i < 2; ++i) {
    if (std::abs(y[i] - expected_y[i]) > 1e-12) return "sparse product self-test gave a wrong result";
  }
  for (std::size_t i = 0; i < 3; ++i) {
    if (std::abs(w[i] - expected_w[i]) > 1e-12) return "sparse product self-test gave a wrong result";
  }
  return "";
}

DeviceInfo probe() {
  Backend& b = backend();
  DeviceInfo info;
  const char* disabled = std::getenv("SOVEREIGN_DISABLE_CUDA");
  if (disabled && std::string(disabled) == "1") {
    info.reason = "Disabled by SOVEREIGN_DISABLE_CUDA=1.";
    return info;
  }
  void* lib = load_driver();
  if (!lib) {
    info.reason = "No NVIDIA driver found (the CUDA driver library is not installed).";
    return info;
  }
  Driver& f = b.f;
  std::string missing;
  if (!bind(lib, "cuInit", f.cuInit, missing) ||
      !bind(lib, "cuDriverGetVersion", f.cuDriverGetVersion, missing) ||
      !bind(lib, "cuDeviceGetCount", f.cuDeviceGetCount, missing) ||
      !bind(lib, "cuDeviceGet", f.cuDeviceGet, missing) ||
      !bind(lib, "cuDeviceGetName", f.cuDeviceGetName, missing) ||
      !bind(lib, "cuDeviceGetAttribute", f.cuDeviceGetAttribute, missing) ||
      !bind(lib, "cuDeviceTotalMem_v2", f.cuDeviceTotalMem, missing) ||
      !bind(lib, "cuDevicePrimaryCtxRetain", f.cuDevicePrimaryCtxRetain, missing) ||
      !bind(lib, "cuCtxSetCurrent", f.cuCtxSetCurrent, missing) ||
      !bind(lib, "cuCtxSynchronize", f.cuCtxSynchronize, missing) ||
      !bind(lib, "cuModuleLoadDataEx", f.cuModuleLoadDataEx, missing) ||
      !bind(lib, "cuModuleGetFunction", f.cuModuleGetFunction, missing) ||
      !bind(lib, "cuMemAlloc_v2", f.cuMemAlloc, missing) ||
      !bind(lib, "cuMemFree_v2", f.cuMemFree, missing) ||
      !bind(lib, "cuMemcpyHtoD_v2", f.cuMemcpyHtoD, missing) ||
      !bind(lib, "cuMemcpyDtoH_v2", f.cuMemcpyDtoH, missing) ||
      !bind(lib, "cuLaunchKernel", f.cuLaunchKernel, missing) ||
      !bind(lib, "cuGetErrorName", f.cuGetErrorName, missing)) {
    info.reason = "The NVIDIA driver is too old (missing " + missing + "). Update the driver.";
    return info;
  }
  CUresult rc = f.cuInit(0);
  if (rc != CUDA_SUCCESS) {
    info.reason = "The NVIDIA driver did not start CUDA (" + error_text(f, rc) + ").";
    return info;
  }
  f.cuDriverGetVersion(&info.driver_version);
  int count = 0;
  if (f.cuDeviceGetCount(&count) != CUDA_SUCCESS || count <= 0) {
    info.reason = "No CUDA-capable GPU is visible to the driver.";
    return info;
  }
  CUdevice dev = 0;
  char name[256] = {0};
  if (f.cuDeviceGet(&dev, 0) != CUDA_SUCCESS) {
    info.reason = "Could not open GPU 0.";
    return info;
  }
  f.cuDeviceGetName(name, static_cast<int>(sizeof(name) - 1), dev);
  info.name = name;
  f.cuDeviceGetAttribute(&info.compute_major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
  f.cuDeviceGetAttribute(&info.compute_minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
  f.cuDeviceTotalMem(&info.memory_bytes, dev);
  rc = f.cuDevicePrimaryCtxRetain(&b.ctx, dev);
  if (rc != CUDA_SUCCESS || f.cuCtxSetCurrent(b.ctx) != CUDA_SUCCESS) {
    info.reason = "Could not create a CUDA context (" + error_text(f, rc) + ").";
    return info;
  }
  const std::string ptx = ptx_source();
  char log[4096] = {0};
  int options[] = {CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void* values[] = {log, reinterpret_cast<void*>(static_cast<std::uintptr_t>(sizeof(log) - 1))};
  CUmodule module = nullptr;
  rc = f.cuModuleLoadDataEx(&module, ptx.c_str(), 2, options, values);
  if (rc != CUDA_SUCCESS) {
    info.reason = "The driver could not compile Sovereign's GPU kernels (" + error_text(f, rc) +
                  (log[0] ? ": " + std::string(log) : std::string()) + ").";
    return info;
  }
  if (f.cuModuleGetFunction(&b.row_dot, module, "row_dot") != CUDA_SUCCESS ||
      f.cuModuleGetFunction(&b.lu_pivot, module, "lu_pivot") != CUDA_SUCCESS ||
      f.cuModuleGetFunction(&b.lu_swap, module, "lu_swap") != CUDA_SUCCESS ||
      f.cuModuleGetFunction(&b.lu_scale, module, "lu_scale") != CUDA_SUCCESS ||
      f.cuModuleGetFunction(&b.lu_update, module, "lu_update") != CUDA_SUCCESS) {
    info.reason = "Sovereign's GPU kernels are missing from the compiled module.";
    return info;
  }
  const std::string failure = self_test();
  if (!failure.empty()) {
    info.reason = "GPU " + failure + ".";
    return info;
  }
  info.available = true;
  return info;
}

}  // namespace

const DeviceInfo& device_info() {
  static const bool probed = (backend().info = probe(), true);
  (void)probed;
  return backend().info;
}

Status dense_lu(std::size_t n, std::vector<double>& a, std::vector<int>& piv) {
  if (!device_info().available) return Status::Error;
  return run_dense_lu(n, a, piv);
}

bool spmv_csc(std::size_t nrows, std::size_t ncols, const std::vector<int>& col_ptr,
              const std::vector<int>& row_idx, const std::vector<double>& values,
              const std::vector<double>& x, std::vector<double>& y, bool transpose) {
  if (!device_info().available) return false;
  return run_spmv(nrows, ncols, col_ptr, row_idx, values, x, y, transpose);
}

}  // namespace cuda
}  // namespace sovereign
