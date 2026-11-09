// matrix_generator.hpp - deterministic generation of the 3D Poisson problem.
//
// The matrix is the 7-point finite-difference discretisation of the negative
// Laplacian on an nx x ny x nz grid with homogeneous Dirichlet boundaries:
//
//     A(i,i) = 6,  A(i,j) = -1 if grid points i and j are face neighbours.
//
// Boundary handling: couplings to points outside the grid are simply
// dropped (they belong to the Dirichlet boundary, whose value is zero). The
// resulting matrix is symmetric (the neighbour relation is symmetric), weakly
// diagonally dominant with strict dominance on boundary rows, and
// irreducible (the grid graph is connected). By the Taussky theorem such a
// matrix is non-singular, and a symmetric matrix with positive diagonal that
// is irreducibly diagonally dominant is positive definite.
//
// Row ordering is lexicographic: row = i + nx * (j + ny * k).
#pragma once

#include <utility>
#include <vector>

#include "common.hpp"
#include "csr_matrix.hpp"

namespace hpcg {

// Exact number of non-zeros of the 7-point Poisson matrix on grid g.
global_t poisson_nnz(const GridDims& g);

// Generate the complete matrix (all rows).
CSRMatrix generate_poisson_3d(const GridDims& g);

// Generate only global rows [row_begin, row_end) - used by each MPI rank to
// build its own block without ever materialising the full matrix.
CSRMatrix generate_poisson_3d_rows(const GridDims& g, global_t row_begin,
                                   global_t row_end);

// Deterministic right-hand side b = A * 1, so the exact solution is the
// vector of ones (the same construction HPCG uses). Computed for the locally
// stored rows only: b_i = sum_j A(i,j).
std::vector<double> generate_rhs(const CSRMatrix& A);

// Contiguous block distribution of n rows over `parts` owners. The first
// (n mod parts) owners get one extra row, so block sizes differ by at most 1.
// Returns [begin, end) for owner `part`.
std::pair<global_t, global_t> block_range(global_t n, int parts, int part);

struct MatrixInfo {
    GridDims grid;
    global_t rows = 0;
    global_t nnz = 0;
    double sparsity_percent = 0.0;  // percentage of ZERO entries
    double density_percent = 0.0;   // percentage of non-zero entries
    double avg_nnz_per_row = 0.0;
    double matrix_bytes = 0.0;      // CSR storage
    double vector_bytes = 0.0;      // CG work vectors (x, b, r, p, Ap)
    double total_bytes() const { return matrix_bytes + vector_bytes; }
};

// Analytic metadata for the full problem (no allocation required), so memory
// can be checked before generating anything.
MatrixInfo describe_poisson(const GridDims& g);

}  // namespace hpcg
