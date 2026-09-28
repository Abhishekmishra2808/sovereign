option(SOVEREIGN_USE_GOOGLETEST
  "Use vendored GoogleTest (requires MSVC or MinGW-w64; not MinGW.org)"
  OFF)

set(SOVEREIGN_TEST_SOURCES
  unit/test_mps_io.cpp
  unit/test_schemas.cpp
  unit/test_validator.cpp
  unit/test_engine_stubs.cpp
  unit/test_verifier.cpp
  unit/test_sparse.cpp
  unit/test_dense_lu.cpp
  unit/test_sparse_lu.cpp
  unit/test_sparse_symmetric.cpp
  unit/test_amd.cpp
  unit/test_sparse_ldlt.cpp
  unit/test_gpu_backend.cpp
  unit/test_presolve.cpp
  unit/test_cuts_heuristics.cpp
  unit/test_correctness_regressions.cpp
  unit/test_ed25519_verify.cpp
  lp/test_lp_solver.cpp
  lp/test_dual_simplex.cpp
  milp/test_milp_solver.cpp
  qp/test_qp_solver.cpp
)

if(SOVEREIGN_USE_GOOGLETEST)
  add_executable(sovereign_tests ${SOVEREIGN_TEST_SOURCES})
  target_link_libraries(sovereign_tests PRIVATE sovereign_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(sovereign_tests)
else()
  add_executable(sovereign_tests
    unit/main.cpp
    ${SOVEREIGN_TEST_SOURCES}
  )
  target_include_directories(sovereign_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  target_compile_definitions(sovereign_tests PRIVATE SOVEREIGN_MINI_TEST=1)
  target_link_libraries(sovereign_tests PRIVATE sovereign_core)
  add_test(NAME sovereign_tests COMMAND sovereign_tests)
endif()

# Tests read data files (e.g. Ed25519 vectors). `ctest` and a direct run use
# different working directories, so a bare relative path resolves under ctest and
# fails while passing when launched by hand. Hand the absolute source path down
# as a compile definition instead of guessing at runtime.
target_compile_definitions(sovereign_tests PRIVATE
  SOVEREIGN_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

# Run from the repository root so relative fixture paths in any other test behave
# the same under ctest as they do interactively.
set_tests_properties(sovereign_tests PROPERTIES
  WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
)
