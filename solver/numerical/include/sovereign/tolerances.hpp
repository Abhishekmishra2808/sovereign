#pragma once

namespace sovereign {

struct Tolerances {
  double feasibility = 1e-8;
  double optimality = 1e-8;
  double pivot = 1e-10;
  double zero = 1e-12;
};

inline bool is_zero(double x, double tol) {
  return x <= tol && x >= -tol;
}

}  // namespace sovereign
