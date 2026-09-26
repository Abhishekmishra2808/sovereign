#pragma once

#include <cmath>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// Lightweight Phase 0 test harness. Prefer GoogleTest when a modern toolchain
// (MSVC / MinGW-w64 / Clang) is available via -DSOVEREIGN_USE_GOOGLETEST=ON.

namespace sovereign_test {

struct Registry {
  using Fn = void (*)();
  static Registry& instance() {
    static Registry r;
    return r;
  }
  void add(const char* name, Fn fn) { tests.push_back({name, fn}); }
  std::vector<std::pair<const char*, Fn>> tests;
  int failures = 0;
};

struct Registrar {
  Registrar(const char* name, Registry::Fn fn) { Registry::instance().add(name, fn); }
};

inline int run_all() {
  auto& reg = Registry::instance();
  int ran = 0;
  int failed_tests = 0;
  for (const auto& t : reg.tests) {
    const int before = reg.failures;
    std::cout << "[ RUN      ] " << t.first << "\n";
    t.second();
    if (reg.failures > before) {
      std::cout << "[  FAILED  ] " << t.first << "\n";
      ++failed_tests;
    } else {
      std::cout << "[       OK ] " << t.first << "\n";
    }
    ++ran;
  }
  std::cout << "[==========] " << ran << " tests ran.\n";
  if (failed_tests == 0) {
    std::cout << "[  PASSED  ] " << ran << " tests.\n";
    return 0;
  }
  std::cout << "[  FAILED  ] " << failed_tests << " tests failed.\n";
  return 1;
}

}  // namespace sovereign_test

#define SOVEREIGN_TEST(suite, name)                                        \
  static void suite##_##name##_fn();                                       \
  static ::sovereign_test::Registrar suite##_##name##_reg(                 \
      #suite "." #name, &suite##_##name##_fn);                             \
  static void suite##_##name##_fn()

#define SOVEREIGN_EXPECT_TRUE(cond)                                        \
  do {                                                                     \
    if (!(cond)) {                                                         \
      std::cerr << __FILE__ << ":" << __LINE__ << " EXPECT_TRUE failed: "  \
                << #cond << "\n";                                          \
      ::sovereign_test::Registry::instance().failures++;                   \
    }                                                                      \
  } while (0)

#define SOVEREIGN_EXPECT_FALSE(cond) SOVEREIGN_EXPECT_TRUE(!(cond))

#define SOVEREIGN_EXPECT_EQ(a, b)                                          \
  do {                                                                     \
    const auto _sa_a = (a);                                                \
    const auto _sa_b = (b);                                                \
    if (!(_sa_a == _sa_b)) {                                               \
      std::cerr << __FILE__ << ":" << __LINE__ << " EXPECT_EQ failed: "    \
                << #a << " vs " << #b << "\n";                             \
      ::sovereign_test::Registry::instance().failures++;                   \
    }                                                                      \
  } while (0)

#define SOVEREIGN_EXPECT_NE(a, b)                                          \
  do {                                                                     \
    const auto _sa_a = (a);                                                \
    const auto _sa_b = (b);                                                \
    if (_sa_a == _sa_b) {                                                  \
      std::cerr << __FILE__ << ":" << __LINE__ << " EXPECT_NE failed: "    \
                << #a << " vs " << #b << "\n";                             \
      ::sovereign_test::Registry::instance().failures++;                   \
    }                                                                      \
  } while (0)

#define SOVEREIGN_ASSERT_EQ(a, b) SOVEREIGN_EXPECT_EQ(a, b)

#define SOVEREIGN_EXPECT_NEAR(a, b, tol)                                   \
  do {                                                                     \
    const double _sa_a = static_cast<double>(a);                           \
    const double _sa_b = static_cast<double>(b);                           \
    const double _sa_t = static_cast<double>(tol);                         \
    if (std::abs(_sa_a - _sa_b) > _sa_t) {                                 \
      std::cerr << __FILE__ << ":" << __LINE__ << " EXPECT_NEAR failed\n"; \
      ::sovereign_test::Registry::instance().failures++;                   \
    }                                                                      \
  } while (0)

// Compatibility aliases used by test sources.
#define TEST(suite, name) SOVEREIGN_TEST(suite, name)
#define EXPECT_TRUE SOVEREIGN_EXPECT_TRUE
#define EXPECT_FALSE SOVEREIGN_EXPECT_FALSE
#define EXPECT_EQ SOVEREIGN_EXPECT_EQ
#define EXPECT_NE SOVEREIGN_EXPECT_NE
#define ASSERT_EQ SOVEREIGN_ASSERT_EQ
#define EXPECT_NEAR SOVEREIGN_EXPECT_NEAR
