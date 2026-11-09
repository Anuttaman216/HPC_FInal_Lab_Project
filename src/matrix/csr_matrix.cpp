// csr_matrix.cpp - CSR utilities and SpMV kernels.
#include "csr_matrix.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef HPCG_HAVE_OPENMP
#include <omp.h>
#endif

namespace hpcg {

void CSRMatrix::validate(global_t max_col) const {
    if (local_rows < 0) throw std::runtime_error("CSR: negative row count");
    if (row_ptr.size() != static_cast<std::size_t>(local_rows) + 1) {
        throw std::runtime_error("CSR: row_ptr has wrong size");
    }
    if (row_ptr.front() != 0) throw std::runtime_error("CSR: row_ptr[0] != 0");
    if (row_ptr.back() != static_cast<offset_t>(values.size()) ||
        col_idx.size() != values.size()) {
        throw std::runtime_error("CSR: nnz mismatch between arrays");
    }
    for (index_t i = 0; i < local_rows; ++i) {
        if (row_ptr[i + 1] < row_ptr[i]) {
            throw std::runtime_error("CSR: row_ptr not monotone at row " +
                                     std::to_string(i));
        }
        for (offset_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            if (col_idx[k] < 0 || col_idx[k] >= max_col) {
                throw std::runtime_error("CSR: column index out of range");
            }
            if (k > row_ptr[i] && col_idx[k] <= col_idx[k - 1]) {
                throw std::runtime_error("CSR: columns not strictly increasing");
            }
        }
    }
}

void CSRMatrix::shift_columns(global_t offset) {
    for (auto& c : col_idx) c = static_cast<index_t>(c - offset);
    col_base += offset;
}

std::vector<double> CSRMatrix::to_dense() const {
    const std::size_t cols = static_cast<std::size_t>(global_cols);
    std::vector<double> dense(static_cast<std::size_t>(local_rows) * cols, 0.0);
    for (index_t i = 0; i < local_rows; ++i) {
        for (offset_t k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
            const global_t col = col_idx[k] + col_base;
            dense[static_cast<std::size_t>(i) * cols + col] = values[k];
        }
    }
    return dense;
}

// Sequential CSR SpMV: y_i = sum_k values[k] * x[col_idx[k]].
// Each row is independent; the inner loop streams through values/col_idx
// contiguously and gathers x (mostly cache friendly for a 7-point stencil).
void spmv(const CSRMatrix& A, const double* x, double* y) {
    const offset_t* rp = A.row_ptr.data();
    const index_t* ci = A.col_idx.data();
    const double* v = A.values.data();
    for (index_t i = 0; i < A.local_rows; ++i) {
        double sum = 0.0;
        for (offset_t k = rp[i]; k < rp[i + 1]; ++k) sum += v[k] * x[ci[k]];
        y[i] = sum;
    }
}

void spmv_openmp(const CSRMatrix& A, const double* x, double* y) {
    const offset_t* rp = A.row_ptr.data();
    const index_t* ci = A.col_idx.data();
    const double* v = A.values.data();
    const index_t n = A.local_rows;
    // Rows are written by exactly one thread -> no race. Static schedule
    // matches the first-touch placement of the vectors.
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (offset_t k = rp[i]; k < rp[i + 1]; ++k) sum += v[k] * x[ci[k]];
        y[i] = sum;
    }
}

bool is_symmetric(const CSRMatrix& A, double tol) {
    if (A.local_rows != A.global_rows || A.global_rows != A.global_cols ||
        A.col_base != 0) {
        throw std::invalid_argument("is_symmetric needs a full square matrix");
    }
    for (index_t i = 0; i < A.local_rows; ++i) {
        for (offset_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
            const index_t j = A.col_idx[k];
            // Look up A(j,i) with a binary search in row j.
            const auto begin = A.col_idx.begin() + A.row_ptr[j];
            const auto end = A.col_idx.begin() + A.row_ptr[j + 1];
            const auto it = std::lower_bound(begin, end, i);
            if (it == end || *it != i) return false;
            const double aji = A.values[static_cast<std::size_t>(it - A.col_idx.begin())];
            if (std::fabs(aji - A.values[k]) > tol) return false;
        }
    }
    return true;
}

}  // namespace hpcg
