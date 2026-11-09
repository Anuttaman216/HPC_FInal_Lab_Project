// common.hpp - shared types, timers and helpers for the HPCG-inspired
// Conjugate Gradient benchmark.
//
// NOTE: This project is an *HPCG-inspired educational implementation*. It is
// not the official HPCG benchmark and its numbers must not be reported as
// official HPCG / TOP500 results.
#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace hpcg {

// Column indices are stored as 32-bit integers to halve the index traffic of
// SpMV (the kernel is memory-bandwidth bound). Row pointers and global sizes
// use 64-bit integers so that nnz may exceed 2^31.
using index_t = std::int32_t;
using offset_t = std::int64_t;
using global_t = std::int64_t;

// Dimensions of the structured 3D grid that defines the Poisson problem.
struct GridDims {
    int nx = 0;
    int ny = 0;
    int nz = 0;

    global_t rows() const {
        return static_cast<global_t>(nx) * static_cast<global_t>(ny) *
               static_cast<global_t>(nz);
    }
    global_t plane() const {
        return static_cast<global_t>(nx) * static_cast<global_t>(ny);
    }
    std::string to_string() const {
        return std::to_string(nx) + "x" + std::to_string(ny) + "x" +
               std::to_string(nz);
    }
    void validate() const {
        if (nx <= 0 || ny <= 0 || nz <= 0) {
            throw std::invalid_argument("grid dimensions must be positive");
        }
    }
};

// Monotonic wall-clock timer based on std::chrono::steady_clock.
class WallTimer {
public:
    WallTimer() : start_(clock::now()) {}
    void reset() { start_ = clock::now(); }
    double seconds() const {
        return std::chrono::duration<double>(clock::now() - start_).count();
    }

private:
    using clock = std::chrono::steady_clock;
    clock::time_point start_;
};

// ---------------------------------------------------------------------------
// Floating-point operation model (documented in report/methodology.md).
//
//   setup      : r = b - A x      -> 2*nnz (SpMV) + n (subtract)
//                (r,r), (b,b)     -> 2n + 2n
//   iteration  : Ap = A p         -> 2*nnz
//                (p,Ap)           -> 2n
//                x += alpha p     -> 2n
//                r -= alpha Ap    -> 2n
//                (r,r)            -> 2n
//                p = r + beta p   -> 2n
//
// Scalar operations (alpha, beta, sqrt) are ignored, as in HPCG. In a
// distributed run the same *global* counts are used, so redundant work and
// communication are not credited as useful flops.
// ---------------------------------------------------------------------------
inline double cg_setup_flops(global_t n, global_t nnz) {
    return 2.0 * static_cast<double>(nnz) + 5.0 * static_cast<double>(n);
}
inline double cg_iteration_flops(global_t n, global_t nnz) {
    return 2.0 * static_cast<double>(nnz) + 10.0 * static_cast<double>(n);
}
inline double cg_total_flops(global_t n, global_t nnz, int iterations) {
    return cg_setup_flops(n, nnz) +
           static_cast<double>(iterations) * cg_iteration_flops(n, nnz);
}

}  // namespace hpcg
