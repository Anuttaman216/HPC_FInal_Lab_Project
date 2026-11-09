// cg_sequential.cpp - sequential (baseline) Conjugate Gradient solver.
//
// All kernels are written as plain loops over contiguous arrays. Two pairs
// of operations are fused to save a pass over memory, and the OpenMP / MPI
// versions use exactly the same fusion so that every implementation performs
// the same arithmetic and the same number of memory sweeps:
//
//   (1) Ap = A p   fused with   pAp = (p, Ap)
//   (2) x += alpha p, r -= alpha Ap   fused with   rr_new = (r, r)
#include <cmath>
#include <stdexcept>

#include "cg_solver.hpp"

namespace hpcg {

std::string to_string(CGStatus status) {
    switch (status) {
        case CGStatus::Converged: return "converged";
        case CGStatus::MaxIterations: return "max_iterations";
        case CGStatus::Breakdown: return "breakdown";
        case CGStatus::NonFinite: return "non_finite";
        case CGStatus::ZeroRhs: return "zero_rhs";
    }
    return "unknown";
}

namespace {

void check_inputs(const CSRMatrix& A, const std::vector<double>& b,
                  const std::vector<double>& x, const CGOptions& opts) {
    if (A.local_rows != A.global_rows || A.global_rows != A.global_cols) {
        throw std::invalid_argument("shared-memory CG needs the full square matrix");
    }
    if (b.size() != static_cast<std::size_t>(A.local_rows) ||
        x.size() != static_cast<std::size_t>(A.local_rows)) {
        throw std::invalid_argument("vector sizes do not match the matrix");
    }
    if (opts.max_iterations < 0 || !(opts.tolerance >= 0.0)) {
        throw std::invalid_argument("invalid CG options");
    }
}

}  // namespace

CGResult cg_solve_sequential(const CSRMatrix& A, const std::vector<double>& b,
                             std::vector<double>& x, const CGOptions& opts) {
    check_inputs(A, b, x, opts);
    const index_t n = A.local_rows;
    const offset_t* rp = A.row_ptr.data();
    const index_t* ci = A.col_idx.data();
    const double* av = A.values.data();

    // Work vectors are allocated once, outside the timed region and never
    // inside the iteration loop.
    std::vector<double> r(n), p(n), Ap(n);
    CGResult res;
    if (opts.record_history) res.residual_history.reserve(opts.max_iterations + 1);

    WallTimer timer;  // ---- timed region starts --------------------------

    // r0 = b - A x0 ; rr = (r,r) ; bb = (b,b) ; p0 = r0
    double rr = 0.0, bb = 0.0;
    for (index_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (offset_t k = rp[i]; k < rp[i + 1]; ++k) ax += av[k] * x[ci[k]];
        const double ri = b[i] - ax;
        r[i] = ri;
        p[i] = ri;
        rr += ri * ri;
        bb += b[i] * b[i];
    }
    res.rhs_norm = std::sqrt(bb);
    res.initial_residual_norm = std::sqrt(rr);
    res.flops = cg_setup_flops(n, A.nnz());

    if (!std::isfinite(rr) || !std::isfinite(bb)) {
        res.status = CGStatus::NonFinite;
    } else if (bb == 0.0) {
        // b == 0: the exact solution of an SPD system is x = 0.
        for (index_t i = 0; i < n; ++i) x[i] = 0.0;
        res.status = CGStatus::ZeroRhs;
        rr = 0.0;
    } else {
        const double stop = opts.tolerance * res.rhs_norm;  // ||r|| <= tol*||b||
        res.status = CGStatus::MaxIterations;
        if (opts.record_history) res.residual_history.push_back(std::sqrt(rr) / res.rhs_norm);

        int iter = 0;
        while (true) {
            if (std::sqrt(rr) <= stop) { res.status = CGStatus::Converged; break; }
            if (iter >= opts.max_iterations) break;

            // (1) Ap = A p and pAp = (p, Ap)
            double pAp = 0.0;
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
            for (index_t i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                const double ri = r[i] - alpha * Ap[i];
                r[i] = ri;
                rr_new += ri * ri;
            }
            ++iter;
            if (!std::isfinite(rr_new)) { rr = rr_new; res.status = CGStatus::NonFinite; break; }

            const double beta = rr_new / rr;
            rr = rr_new;
            // (3) p = r + beta p
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

double true_relative_residual(const CSRMatrix& A, const std::vector<double>& b,
                              const std::vector<double>& x) {
    std::vector<double> ax(static_cast<std::size_t>(A.local_rows));
    spmv(A, x.data(), ax.data());
    double rr = 0.0, bb = 0.0;
    for (index_t i = 0; i < A.local_rows; ++i) {
        const double d = b[i] - ax[i];
        rr += d * d;
        bb += b[i] * b[i];
    }
    return bb > 0.0 ? std::sqrt(rr / bb) : std::sqrt(rr);
}

}  // namespace hpcg
