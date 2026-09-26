option(SOVEREIGN_USE_GOOGLETEST
  "Use vendored GoogleTest (requires MSVC or MinGW-w64; not MinGW.org)"
  OFF)

set(SOVEREIGN_TEST_SOURCES
  unit/test_schemas.cpp
  unit/test_validator.cpp
  unit/test_engine_stubs.cpp
  unit/test_verifier.cpp
  unit/test_sparse.cpp
  unit/test_dense_lu.cpp
  unit/test_presolve.cpp
  unit/test_cuts_heuristics.cpp
  lp/test_lp_solver.cpp
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
