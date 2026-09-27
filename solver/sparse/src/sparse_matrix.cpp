#include "sovereign/sparse_matrix.hpp"
#include "sovereign/gpu_spmv.hpp"

#include <algorithm>
#include <stdexcept>

namespace sovereign {

void SparseMatrixCSC::clear() {
  nrows = 0;
  ncols = 0;
  col_ptr.clear();
  row_idx.clear();
  values.clear();
}

void SparseMatrixCSC::resize(std::size_t rows, std::size_t cols) {
  nrows = rows;
  ncols = cols;
  col_ptr.assign(cols + 1, 0);
  row_idx.clear();
  values.clear();
}

void SparseMatrixCSC::reserve_nnz(std::size_t nnz) {
  row_idx.reserve(nnz);
  values.reserve(nnz);
}

void SparseMatrixCSC::push_back(int row, double value) {
  row_idx.push_back(row);
  values.push_back(value);
}

void SparseMatrixCSC::finish_column(std::size_t col) {
  if (col + 1 >= col_ptr.size()) {
    throw std::runtime_error("SparseMatrixCSC::finish_column out of range");
  }
  col_ptr[col + 1] = static_cast<int>(values.size());
}

void SparseMatrixCSC::multiply(const std::vector<double>& x, std::vector<double>& y) const {
  if (x.size() != ncols) {
    throw std::invalid_argument("SparseMatrixCSC::multiply dimension mismatch");
  }
  spmv_csc_auto(nrows, ncols, col_ptr, row_idx, values, x, y);
}

void SparseMatrixCSC::multiply_transpose(const std::vector<double>& x,
                                         std::vector<double>& y) const {
  if (x.size() != nrows) {
    throw std::invalid_argument("SparseMatrixCSC::multiply_transpose dimension mismatch");
  }
  y.assign(ncols, 0.0);
  for (std::size_t j = 0; j < ncols; ++j) {
    double sum = 0.0;
    for (int p = col_ptr[j]; p < col_ptr[j + 1]; ++p) {
      sum += values[static_cast<std::size_t>(p)] *
             x[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(p)])];
    }
    y[j] = sum;
  }
}

void SparseMatrixCSC::extract_column(std::size_t col, std::vector<double>& dense) const {
  dense.assign(nrows, 0.0);
  if (col >= ncols) return;
  for (int p = col_ptr[col]; p < col_ptr[col + 1]; ++p) {
    dense[static_cast<std::size_t>(row_idx[static_cast<std::size_t>(p)])] =
        values[static_cast<std::size_t>(p)];
  }
}

void SparseMatrixCSC::extract_dense_basis(const std::vector<int>& basis,
                                          std::vector<double>& dense_col_major) const {
  const std::size_t m = basis.size();
  if (m != nrows) {
    throw std::invalid_argument("Basis size must equal number of rows");
  }
  dense_col_major.assign(m * m, 0.0);
  for (std::size_t j = 0; j < m; ++j) {
    const int col = basis[j];
    if (col < 0 || static_cast<std::size_t>(col) >= ncols) {
      throw std::invalid_argument("Invalid basis column index");
    }
    for (int p = col_ptr[static_cast<std::size_t>(col)];
         p < col_ptr[static_cast<std::size_t>(col) + 1]; ++p) {
      const std::size_t row = static_cast<std::size_t>(row_idx[static_cast<std::size_t>(p)]);
      dense_col_major[j * m + row] = values[static_cast<std::size_t>(p)];
    }
  }
}

SparseMatrixCSR SparseMatrixCSR::from_csc(const SparseMatrixCSC& a) {
  SparseMatrixCSR r;
  r.nrows = a.nrows;
  r.ncols = a.ncols;
  r.row_ptr.assign(a.nrows + 1, 0);
  for (std::size_t p = 0; p < a.row_idx.size(); ++p) {
    r.row_ptr[static_cast<std::size_t>(a.row_idx[p]) + 1]++;
  }
  for (std::size_t i = 0; i < a.nrows; ++i) {
    r.row_ptr[i + 1] += r.row_ptr[i];
  }
  const int nnz = r.row_ptr.back();
  r.col_idx.assign(static_cast<std::size_t>(nnz), 0);
  r.values.assign(static_cast<std::size_t>(nnz), 0.0);
  std::vector<int> next = r.row_ptr;
  for (std::size_t j = 0; j < a.ncols; ++j) {
    for (int p = a.col_ptr[j]; p < a.col_ptr[j + 1]; ++p) {
      const int row = a.row_idx[static_cast<std::size_t>(p)];
      const int dest = next[static_cast<std::size_t>(row)]++;
      r.col_idx[static_cast<std::size_t>(dest)] = static_cast<int>(j);
      r.values[static_cast<std::size_t>(dest)] = a.values[static_cast<std::size_t>(p)];
    }
  }
  return r;
}

}  // namespace sovereign
