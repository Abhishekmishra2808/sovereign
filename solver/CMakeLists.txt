add_library(sovereign_core
  core/src/types.cpp
  core/src/json_io.cpp
  core/src/model_validator.cpp
  core/src/verifier.cpp
  core/src/engine.cpp
  sparse/src/sparse_matrix.cpp
  numerical/src/dense_lu.cpp
  lp/src/lp_solver.cpp
  lp/src/revised_simplex.cpp
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

target_link_libraries(sovereign_core PUBLIC nlohmann_json::nlohmann_json)

find_package(Threads REQUIRED)
target_link_libraries(sovereign_core PUBLIC Threads::Threads)

add_executable(sovereign cli/main.cpp)
target_link_libraries(sovereign PRIVATE sovereign_core)

if(WIN32)
  set_target_properties(sovereign PROPERTIES WIN32_EXECUTABLE FALSE)
endif()
