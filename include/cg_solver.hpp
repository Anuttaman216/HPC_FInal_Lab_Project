// cg_solver.hpp - Conjugate Gradient solver interfaces (shared-memory).
//
// Unpreconditioned CG for A x = b with A symmetric positive definite:
//
//   r0 = b - A x0,  p0 = r0
//   loop:  alpha = (r,r) / (p,Ap)
//          x     = x + alpha p
//          r_new = r - alpha Ap
//          beta  = (r_new,r_new) / (r,r)
//          p     = r_new + beta p
//   until ||r|| / ||b|| <= tol  or  max_iterations reached.
//
// The distributed (MPI and hybrid) solvers are declared in distributed.hpp.
#pragma once

#include <string>
#include <vector>

#include "common.hpp"
#include "csr_matrix.hpp"

namespace hpcg {

struct CGOptions {
    int max_iterations = 1000;
    double tolerance = 1e-8;      // on ||r_k||_2 / ||b||_2 ; 0 => fixed iterations
    bool record_history = false;  // store ||r_k||/||b|| for every iteration
};

enum class CGStatus {
    Converged,      // relative residual <= tolerance
    MaxIterations,  // iteration limit reached first
    Breakdown,      // (p,Ap) <= 0 or zero: matrix not SPD / exact breakdown
    NonFinite,      // NaN or Inf detected in a reduction
    ZeroRhs         // ||b|| == 0 -> x = 0 is the exact solution
};

std::string to_string(CGStatus status);

struct CGResult {
    CGStatus status = CGStatus::MaxIterations;
    int iterations = 0;                // number of completed CG updates
    double rhs_norm = 0.0;             // ||b||
    double initial_residual_norm = 0.0;
    double final_residual_norm = 0.0;  // recursively updated ||r||
    double relative_residual = 0.0;    // final_residual_norm / rhs_norm

    // Timing (seconds). solve_seconds covers the whole solve from the initial
    // residual to the final iteration; matrix generation and allocation are
    // excluded for every implementation.
    double solve_seconds = 0.0;
    double comm_exchange_seconds = 0.0;   // halo / allgather of p (MPI only)
    double comm_reduction_seconds = 0.0;  // MPI_Allreduce time (MPI only)

    double flops = 0.0;  // modelled floating-point operations (see common.hpp)

    std::vector<double> residual_history;

    bool converged() const {
        return status == CGStatus::Converged || status == CGStatus::ZeroRhs;
    }
    double comm_seconds() const {
        return comm_exchange_seconds + comm_reduction_seconds;
    }
    double gflops() const {
        return solve_seconds > 0.0 ? flops / (solve_seconds * 1e9) : 0.0;
    }
};

// Sequential baseline. x holds the initial guess on entry and the solution
// on exit; it must have A.local_rows entries. A must be a full matrix.
CGResult cg_solve_sequential(const CSRMatrix& A, const std::vector<double>& b,
                             std::vector<double>& x, const CGOptions& opts);

// OpenMP shared-memory solver; uses omp_get_max_threads() threads.
CGResult cg_solve_openmp(const CSRMatrix& A, const std::vector<double>& b,
                         std::vector<double>& x, const CGOptions& opts);

// ||b - A x|| / ||b|| computed from scratch (verification, not timed).
double true_relative_residual(const CSRMatrix& A, const std::vector<double>& b,
                              const std::vector<double>& x);

// True when the project was compiled with OpenMP support.
bool openmp_available();

}  // namespace hpcg
