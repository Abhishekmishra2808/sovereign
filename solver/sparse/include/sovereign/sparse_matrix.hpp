#pragma once

#include <cstddef>
#include <vector>

namespace sovereign {

// Compressed Sparse Column matrix.
struct SparseMatrixCSC {
  std::size_t nrows = 0;
  std::size_t ncols = 0;
  std::vector<int> col_ptr;   // size ncols + 1
  std::vector<int> row_idx;   // size nnz
  std::vector<double> values; // size nnz

  void clear();
  void resize(std::size_t rows, std::size_t cols);
  void reserve_nnz(std::size_t nnz);
  void finish_column(std::size_t col);  // call after appending entries for col
  void push_back(int row, double value);

  std::size_t nnz() const { return values.size(); }

  // y = A * x
  void multiply(const std::vector<double>& x, std::vector<double>& y) const;

  // y = A^T * x
  void multiply_transpose(const std::vector<double>& x, std::vector<double>& y) const;

  // Extract dense column into densified vector of length nrows (zeros filled).
  void extract_column(std::size_t col, std::vector<double>& dense) const;

  // Extract submatrix columns basis[j] into dense m x m column-major matrix.
  void extract_dense_basis(const std::vector<int>& basis,
                           std::vector<double>& dense_col_major) const;
};

// Compressed Sparse Row (built from CSC when needed).
struct SparseMatrixCSR {
  std::size_t nrows = 0;
  std::size_t ncols = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> values;

  static SparseMatrixCSR from_csc(const SparseMatrixCSC& a);
};

}  // namespace sovereign
