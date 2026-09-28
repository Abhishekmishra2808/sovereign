cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
project(sovereign_optimizer VERSION 0.1.0 LANGUAGES CXX)

# Prefer C++17; fall back gracefully on older toolchains.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 7)
  set(CMAKE_CXX_STANDARD 17)
elseif(MSVC OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# The engine is copied to worker machines on its own, without the compiler's
# runtime DLLs (libstdc++, libgcc), so MinGW builds link them statically.
if(MINGW)
  add_link_options(-static)
endif()

option(SOVEREIGN_BUILD_TESTS "Build unit tests" ON)

include(FetchContent)

set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

add_subdirectory(solver)

if(SOVEREIGN_BUILD_TESTS)
  enable_testing()
  option(SOVEREIGN_USE_GOOGLETEST
    "Use vendored GoogleTest (requires MSVC or MinGW-w64; not MinGW.org)"
    OFF)
  if(SOVEREIGN_USE_GOOGLETEST)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(third_party/googletest EXCLUDE_FROM_ALL)
  endif()
  add_subdirectory(tests)
endif()

option(SOVEREIGN_BUILD_GPU_BENCHMARK "Build the GPU SpMV micro-benchmark" OFF)
if(SOVEREIGN_BUILD_GPU_BENCHMARK)
  add_executable(gpu_benchmark benchmarks/gpu_benchmark.cpp)
  target_link_libraries(gpu_benchmark PRIVATE sovereign_core)
endif()

install(TARGETS sovereign RUNTIME DESTINATION bin)
