cmake_minimum_required(VERSION 3.16)
project(sovereign_mono C)

# CUDA support option
option(SOVEREIGN_USE_CUDA "Enable CUDA GPU acceleration" OFF)

if(SOVEREIGN_USE_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  add_definitions(-DSOVEREIGN_USE_CUDA)
endif()

# ---------------------------------------------------------------------------
# Monocypher: Ed25519 verification + SHA-256.
#
# Used only by the offline-authorization path (verifying server-signed offline
# credentials). Sovereign never signs. Built as a separate static library so it
# can be linked by tests without dragging in the whole engine, and so the C99
# primitive is compiled with its own conservative flags.
# ---------------------------------------------------------------------------
add_library(monocypher STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/monocypher/monocypher.c
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/monocypher/monocypher-ed25519.c
)
target_include_directories(monocypher PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/monocypher
)
set_target_properties(monocypher PROPERTIES
  C_STANDARD 99
  POSITION_INDEPENDENT_CODE ON
)
if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
  # -O2 only. Deliberately NOT -ffast-math: unacceptable in a crypto primitive.
  target_compile_options(monocypher PRIVATE -O2)
endif()

add_library(sovereign_core
  core/src/types.cpp
  core/src/json_io.cpp
  core/src/mps_io.cpp
  core/src/model_validator.cpp
  core/src/verifier.cpp
  core/src/engine.cpp
  sparse/src/sparse_matrix.cpp
  numerical/src/dense_lu.cpp
  numerical/src/sparse_lu.cpp
  numerical/src/ed25519_verify.cpp
  lp/src/lp_solver.cpp
  lp/src/revised_simplex.cpp
  lp/src/dual_simplex.cpp
  lp/src/interior_point.cpp
  qp/src/qp_solver.cpp
  qp/src/convex_qp.cpp
  qp/src/qp_interior_point.cpp
  milp/src/milp_solver.cpp
  milp/src/branch_and_bound.cpp
  cuts/src/cuts.cpp
  heuristics/src/heuristics.cpp
  presolve/src/presolve.cpp
  gpu/src/gpu_spmv.cpp
)

# Add CUDA source if enabled
if(SOVEREIGN_USE_CUDA)
  add_library(sovereign_cuda
    gpu/src/cuda_spmv.cu
  )
  set_target_properties(sovereign_cuda PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
  )
  target_include_directories(sovereign_cuda
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/include
  )
  target_link_libraries(sovereign_core PUBLIC sovereign_cuda)
  target_link_libraries(sovereign_core PUBLIC CUDA::cudart)
endif()

target_include_directories(sovereign_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/sparse/include
    ${CMAKE_CURRENT_SOURCE_DIR}/numerical/include
    ${CMAKE_CURRENT_SOURCE_DIR}/lp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/qp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/milp/include
    ${CMAKE_CURRENT_SOURCE_DIR}/presolve/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cuts/include
    ${CMAKE_CURRENT_SOURCE_DIR}/heuristics/include
    ${CMAKE_CURRENT_SOURCE_DIR}/gpu/include
)

find_package(Threads REQUIRED)
target_link_libraries(sovereign_core PUBLIC nlohmann_json::nlohmann_json)
target_link_libraries(sovereign_core PUBLIC Threads::Threads)
target_link_libraries(sovereign_core PUBLIC monocypher)

add_executable(sovereign cli/main.cpp)
target_link_libraries(sovereign PRIVATE sovereign_core)

if(WIN32)
  set_target_properties(sovereign PROPERTIES WIN32_EXECUTABLE FALSE)
endif()

# ---------------------------------------------------------------------------
# Local HTTP server (sovereign-server)
#
# The transport layer only. No optimization logic, and no dependency on
# sovereign_core's internals beyond the public engine API.
# ---------------------------------------------------------------------------
add_library(sovereign_server STATIC
  server/src/job_manager.cpp
  server/src/http_server.cpp
  server/src/security.cpp
)
target_include_directories(sovereign_server
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/server/include
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cpp-httplib
)
target_link_libraries(sovereign_server PUBLIC sovereign_core)
# httplib needs pthreads; on MinGW it also wants the Winsock/WS2_32 libraries.
target_link_libraries(sovereign_server PUBLIC Threads::Threads)
if(WIN32)
  target_link_libraries(sovereign_server PUBLIC ws2_32)
endif()

add_executable(sovereign-server server/src/server_main.cpp)
target_link_libraries(sovereign-server PRIVATE sovereign_server)
# ---------------------------------------------------------------------------
# Launcher (sovereign-launcher)
#
# Process lifecycle only: reuse a running instance, otherwise pick a free
# loopback port, start the server, and open the default browser. No solver or
# HTTP logic here.
# ---------------------------------------------------------------------------
if(WIN32)
add_executable(sovereign-launcher launcher/src/launcher_main.cpp)
target_link_libraries(sovereign-launcher PRIVATE
  ws2_32
  shell32
)
set_target_properties(sovereign-launcher PROPERTIES
  WIN32_EXECUTABLE FALSE
)
endif()
