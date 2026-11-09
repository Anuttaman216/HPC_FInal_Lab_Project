// cg_openmp.cpp - OpenMP shared-memory Conjugate Gradient solver.
//
// Parallelisation strategy
//   * Every vector kernel and the SpMV are `omp parallel for` loops with a
//     static schedule, so thread t always touches the same index range of
//     every vector (good cache / first-touch locality, no false sharing
//     except at chunk boundaries).
//   * Dot products use `reduction(+:...)`, which gives each thread a private
//     partial sum that is combined once at the end of the loop -> no data
//     race and no atomics in the inner loop.
//   * The implicit barrier at the end of each worksharing loop is the only
//     synchronisation; it is required because the next kernel reads values
//     written by other threads (e.g. SpMV reads all of p).
//   * The scalar recurrences (alpha, beta, convergence test) are computed by
//     the master thread between parallel loops, so every thread sees the
//     same values and no extra synchronisation is needed.
//
// The kernels are fused exactly like the sequential baseline. Results differ
// from the sequential solver only through the summation order of the
// reductions (validated with a tolerance in tests/test_openmp.cpp).
#include <cmath>
#include <stdexcept>

#include "cg_solver.hpp"

#ifdef HPCG_HAVE_OPENMP
#include <omp.h>
#endif

namespace hpcg {

bool openmp_available() {
#ifdef HPCG_HAVE_OPENMP
    return true;
#else
    return false;
#endif
}

#ifdef HPCG_HAVE_OPENMP

CGResult cg_solve_openmp(const CSRMatrix& A, const std::vector<double>& b,
                         std::vector<double>& x, const CGOptions& opts) {
    if (A.local_rows != A.global_rows || A.global_rows != A.global_cols) {
        throw std::invalid_argument("OpenMP CG needs the full square matrix");
    }
    if (b.size() != static_cast<std::size_t>(A.local_rows) ||
        x.size() != static_cast<std::size_t>(A.local_rows)) {
        throw std::invalid_argument("vector sizes do not match the matrix");
    }
    if (opts.max_iterations < 0 || !(opts.tolerance >= 0.0)) {
        throw std::invalid_argument("invalid CG options");
    }

    const index_t n = A.local_rows;
    const offset_t* rp = A.row_ptr.data();
    const index_t* ci = A.col_idx.data();
    const double* av = A.values.data();
    const double* bp = b.data();
    double* xp = x.data();

    // Allocate work vectors outside the timed region and initialise them in
    // parallel with the same static schedule used by the kernels
    // (first-touch page placement on NUMA systems).
    std::vector<double> r_vec(n), p_vec(n), ap_vec(n);
    double* r = r_vec.data();
    double* p = p_vec.data();
    double* Ap = ap_vec.data();
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < n; ++i) { r[i] = 0.0; p[i] = 0.0; Ap[i] = 0.0; }

    CGResult res;
    if (opts.record_history) res.residual_history.reserve(opts.max_iterations + 1);

    WallTimer timer;  // ---- timed region starts --------------------------

    double rr = 0.0, bb = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr, bb)
    for (index_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (offset_t k = rp[i]; k < rp[i + 1]; ++k) ax += av[k] * xp[ci[k]];
        const double ri = bp[i] - ax;
        r[i] = ri;
        p[i] = ri;
        rr += ri * ri;
        bb += bp[i] * bp[i];
    }
    res.rhs_norm = std::sqrt(bb);
    res.initial_residual_norm = std::sqrt(rr);
    res.flops = cg_setup_flops(n, A.nnz());

    if (!std::isfinite(rr) || !std::isfinite(bb)) {
        res.status = CGStatus::NonFinite;
    } else if (bb == 0.0) {
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < n; ++i) xp[i] = 0.0;
        res.status = CGStatus::ZeroRhs;
        rr = 0.0;
    } else {
        const double stop = opts.tolerance * res.rhs_norm;
        res.status = CGStatus::MaxIterations;
        if (opts.record_history) res.residual_history.push_back(std::sqrt(rr) / res.rhs_norm);

        int iter = 0;
        while (true) {
            if (std::sqrt(rr) <= stop) { res.status = CGStatus::Converged; break; }
            if (iter >= opts.max_iterations) break;

            // (1) Ap = A p fused with pAp = (p, Ap)
            double pAp = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : pAp)
            for (index_t i = 0; i < n; ++i) {
                double sum = 0.0;
                for (offset_t k = rp[i]; k < rp[i + 1]; ++k) sum += av[k] * p[ci[k]];
                Ap[i] = sum;
                pAp += p[i] * sum;
            }
            if (!std::isfinite(pAp)) { res.status = CGStatus::NonFinite; break; }
            if (pAp <= 0.0) { res.status = CGStatus::Breakdown; break; }

            const double alpha = rr / pAp;

            // (2) x += alpha p ; r -= alpha Ap ; rr_new = (r, r)
            double rr_new = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr_new)
            for (index_t i = 0; i < n; ++i) {
                xp[i] += alpha * p[i];
                const double ri = r[i] - alpha * Ap[i];
                r[i] = ri;
                rr_new += ri * ri;
            }
            ++iter;
            if (!std::isfinite(rr_new)) { rr = rr_new; res.status = CGStatus::NonFinite; break; }

            const double beta = rr_new / rr;
            rr = rr_new;

            // (3) p = r + beta p
#pragma omp parallel for schedule(static)
            for (index_t i = 0; i < n; ++i) p[i] = r[i] + beta * p[i];

            if (opts.record_history) res.residual_history.push_back(std::sqrt(rr) / res.rhs_norm);
        }
        res.iterations = iter;
        res.flops += static_cast<double>(iter) * cg_iteration_flops(n, A.nnz());
    }

    res.solve_seconds = timer.seconds();  // ---- timed region ends --------

    res.final_residual_norm = std::sqrt(rr);
    res.relative_residual =
        res.rhs_norm > 0.0 ? res.final_residual_norm / res.rhs_norm : 0.0;
    return res;
}

#else  // !HPCG_HAVE_OPENMP

CGResult cg_solve_openmp(const CSRMatrix&, const std::vector<double>&,
                         std::vector<double>&, const CGOptions&) {
    // Refuse instead of silently running sequential code labelled "OpenMP".
    throw std::runtime_error("this build was compiled without OpenMP support");
}

#endif

}  // namespace hpcg
