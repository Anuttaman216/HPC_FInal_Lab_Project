// matrix_generator.cpp - 3D 7-point Poisson matrix in CSR format.
#include "matrix_generator.hpp"

#include <limits>
#include <stdexcept>

namespace hpcg {

namespace {

// Number of in-grid entries (including the diagonal) of the row belonging
// to grid point (i, j, k).
inline int row_length(const GridDims& g, int i, int j, int k) {
    int len = 1;
    len += (i > 0) + (i < g.nx - 1);
    len += (j > 0) + (j < g.ny - 1);
    len += (k > 0) + (k < g.nz - 1);
    return len;
}

}  // namespace

global_t poisson_nnz(const GridDims& g) {
    g.validate();
    const global_t nx = g.nx, ny = g.ny, nz = g.nz;
    // diagonal + 2 entries for every pair of face-adjacent grid points
    return g.rows() + 2 * ((nx - 1) * ny * nz + nx * (ny - 1) * nz +
                           nx * ny * (nz - 1));
}

CSRMatrix generate_poisson_3d(const GridDims& g) {
    return generate_poisson_3d_rows(g, 0, g.rows());
}

CSRMatrix generate_poisson_3d_rows(const GridDims& g, global_t row_begin,
                                   global_t row_end) {
    g.validate();
    const global_t n = g.rows();
    if (n > std::numeric_limits<index_t>::max()) {
        throw std::invalid_argument(
            "grid too large for 32-bit column indices (" + g.to_string() + ")");
    }
    if (row_begin < 0 || row_end > n || row_begin > row_end) {
        throw std::invalid_argument("invalid row range for matrix generation");
    }

    CSRMatrix A;
    A.global_rows = n;
    A.global_cols = n;
    A.first_row = row_begin;
    A.local_rows = static_cast<index_t>(row_end - row_begin);
    A.col_base = 0;

    const global_t plane = g.plane();

    // Pass 1: row lengths -> row_ptr (exclusive prefix sum).
    A.row_ptr.assign(static_cast<std::size_t>(A.local_rows) + 1, 0);
    for (index_t r = 0; r < A.local_rows; ++r) {
        const global_t row = row_begin + r;
        const int i = static_cast<int>(row % g.nx);
        const int j = static_cast<int>((row / g.nx) % g.ny);
        const int k = static_cast<int>(row / plane);
        A.row_ptr[r + 1] = A.row_ptr[r] + row_length(g, i, j, k);
    }
    const std::size_t nnz = static_cast<std::size_t>(A.row_ptr.back());
    A.col_idx.resize(nnz);
    A.values.resize(nnz);

    // Pass 2: fill. Rows are independent, so this loop is parallelised with
    // OpenMP (first-touch placement of the matrix pages by the thread that
    // will later process those rows with a static schedule).
    const index_t local_rows = A.local_rows;
#pragma omp parallel for schedule(static)
    for (index_t r = 0; r < local_rows; ++r) {
        const global_t row = row_begin + r;
        const int i = static_cast<int>(row % g.nx);
        const int j = static_cast<int>((row / g.nx) % g.ny);
        const int k = static_cast<int>(row / plane);
        offset_t pos = A.row_ptr[r];
        auto put = [&](global_t col, double val) {
            A.col_idx[pos] = static_cast<index_t>(col);
            A.values[pos] = val;
            ++pos;
        };
        // Neighbours are emitted in increasing column order.
        if (k > 0) put(row - plane, -1.0);
        if (j > 0) put(row - g.nx, -1.0);
        if (i > 0) put(row - 1, -1.0);
        put(row, 6.0);
        if (i < g.nx - 1) put(row + 1, -1.0);
        if (j < g.ny - 1) put(row + g.nx, -1.0);
        if (k < g.nz - 1) put(row + plane, -1.0);
    }
    return A;
}

std::vector<double> generate_rhs(const CSRMatrix& A) {
    std::vector<double> b(static_cast<std::size_t>(A.local_rows));
    for (index_t i = 0; i < A.local_rows; ++i) {
        double sum = 0.0;
        for (offset_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) sum += A.values[k];
        b[i] = sum;  // (A * ones)_i
    }
    return b;
}

std::pair<global_t, global_t> block_range(global_t n, int parts, int part) {
    if (parts <= 0 || part < 0 || part >= parts || n < 0) {
        throw std::invalid_argument("block_range: invalid arguments");
    }
    const global_t base = n / parts;
    const global_t extra = n % parts;
    const global_t begin = part * base + (part < extra ? part : extra);
    const global_t size = base + (part < extra ? 1 : 0);
    return {begin, begin + size};
}

MatrixInfo describe_poisson(const GridDims& g) {
    MatrixInfo info;
    info.grid = g;
    info.rows = g.rows();
    info.nnz = poisson_nnz(g);
    const double n = static_cast<double>(info.rows);
    const double total = n * n;
    info.density_percent = 100.0 * static_cast<double>(info.nnz) / total;
    info.sparsity_percent = 100.0 - info.density_percent;
    info.avg_nnz_per_row = static_cast<double>(info.nnz) / n;
    info.matrix_bytes = (n + 1.0) * sizeof(offset_t) +
                        static_cast<double>(info.nnz) * (sizeof(index_t) + sizeof(double));
    info.vector_bytes = 5.0 * n * sizeof(double);  // x, b, r, p, Ap
    return info;
}

}  // namespace hpcg
