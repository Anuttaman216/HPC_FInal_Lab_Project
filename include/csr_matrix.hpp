// csr_matrix.hpp - Compressed Sparse Row (CSR) matrix storage.
//
// A CSRMatrix can hold either the full matrix (first_row == 0 and
// local_rows == global_rows) or a contiguous block of rows owned by one MPI
// process. Column indices are interpreted relative to `col_base`: a stored
// column c refers to global column (c + col_base). For a full matrix
// col_base == 0; the distributed solvers shift columns so that they index
// directly into the local "extended" vector window (see distributed.hpp).
#pragma once

#include <cstddef>
#include <vector>

#include "common.hpp"

namespace hpcg {

struct CSRMatrix {
    global_t global_rows = 0;  // rows of the complete matrix
    global_t global_cols = 0;  // columns of the complete matrix
    global_t first_row = 0;    // global index of local row 0
    index_t local_rows = 0;    // rows stored in this object
    global_t col_base = 0;     // stored column c == global column c + col_base

    std::vector<offset_t> row_ptr;  // size local_rows + 1
    std::vector<index_t> col_idx;   // size nnz, sorted within each row
    std::vector<double> values;     // size nnz

    global_t nnz() const { return static_cast<global_t>(values.size()); }

    // Bytes used by the CSR arrays themselves.
    std::size_t storage_bytes() const {
        return row_ptr.size() * sizeof(offset_t) +
               col_idx.size() * sizeof(index_t) +
               values.size() * sizeof(double);
    }

    // Throws std::runtime_error if the CSR invariants are violated
    // (monotone row_ptr, column indices in range and strictly increasing).
    // `max_col` is the exclusive upper bound for stored column indices.
    void validate(global_t max_col) const;

    // Shift stored column indices by -offset and record the new base.
    void shift_columns(global_t offset);

    // Dense copy (only sensible for tiny test matrices).
    std::vector<double> to_dense() const;
};

// y = A * x for the locally stored rows. `x` must be indexable by every
// stored column index; `y` must have local_rows entries. Sequential kernel.
void spmv(const CSRMatrix& A, const double* x, double* y);

// OpenMP-parallel SpMV (static row partitioning). Falls back to the
// sequential kernel when the project is built without OpenMP.
void spmv_openmp(const CSRMatrix& A, const double* x, double* y);

// Returns true if a full (non-distributed) matrix is numerically symmetric
// to within `tol` (uses binary search in the sorted rows).
bool is_symmetric(const CSRMatrix& A, double tol = 0.0);

}  // namespace hpcg
