// cg_mpi.cpp - distributed-memory Conjugate Gradient (pure MPI).
//
// Per iteration every rank performs:
//   1. exchange(p)          : refresh the extended window of p
//                             (MPI_Allgatherv or halo Isend/Irecv)
//   2. Ap = A_local p_ext   : local SpMV on the owned rows, fused with the
//                             local part of (p, Ap)
//   3. MPI_Allreduce(pAp)   : global dot product
//   4. x, r updates         : local, fused with local part of (r, r)
//   5. MPI_Allreduce(rr)    : global dot product
//   6. p = r + beta p       : local
//
// alpha, beta and the convergence test are computed from the *globally
// reduced* values, so every rank takes the same branch and the sequence of
// collective calls is identical on all ranks (no deadlock, no divergence).
// No OpenMP is used here: one MPI process == one core.
#ifdef HPCG_HAVE_MPI

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "distributed.hpp"

namespace hpcg {

CGResult cg_solve_mpi(const DistributedProblem& prob, std::vector<double>& x_local,
                      const CGOptions& opts, MPI_Comm comm) {
    const CSRMatrix& A = prob.A;
    const VectorExchange& ex = prob.exchange;
    const index_t nl = A.local_rows;
    if (x_local.size() != static_cast<std::size_t>(nl)) {
        throw std::invalid_argument("x_local has wrong size");
    }
    int size = 1;
    MPI_Comm_size(comm, &size);

    const offset_t* rp = A.row_ptr.data();
    const index_t* ci = A.col_idx.data();
    const double* av = A.values.data();
    const double* b = prob.b.data();
    double* x = x_local.data();

    // Work vectors (allocated once, outside the timed region).
    std::vector<double> p_ext(static_cast<std::size_t>(ex.ext_size()), 0.0);
    std::vector<double> r_vec(nl, 0.0), ap_vec(nl, 0.0);
    double* p = p_ext.data() + ex.own_offset();  // owned part of the window
    double* r = r_vec.data();
    double* Ap = ap_vec.data();

    CGResult res;
    if (opts.record_history) res.residual_history.reserve(opts.max_iterations + 1);
    double t_exchange = 0.0, t_reduce = 0.0;

    auto exchange_p = [&]() {
        const double t = MPI_Wtime();
        ex.exchange(p_ext);
        t_exchange += MPI_Wtime() - t;
    };
    auto allreduce_sum = [&](double* vals, int count) {
        const double t = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, vals, count, MPI_DOUBLE, MPI_SUM, comm);
        t_reduce += MPI_Wtime() - t;
    };

    MPI_Barrier(comm);  // common start line for all ranks
    const double t_start = MPI_Wtime();  // ---- timed region starts -------

    // r0 = b - A x0. The window of p temporarily carries x0 so that the
    // off-process entries of x0 can be exchanged with the same plan.
    std::copy(x, x + nl, p);
    exchange_p();
    double sums[2] = {0.0, 0.0};  // (r,r), (b,b)
    for (index_t i = 0; i < nl; ++i) {
        double ax = 0.0;
        for (offset_t k = rp[i]; k < rp[i + 1]; ++k) ax += av[k] * p_ext[ci[k]];
        const double ri = b[i] - ax;
        r[i] = ri;
        sums[0] += ri * ri;
        sums[1] += b[i] * b[i];
    }
    // p0 = r0 (separate pass: the SpMV above still read the window).
    for (index_t i = 0; i < nl; ++i) p[i] = r[i];
    allreduce_sum(sums, 2);
    double rr = sums[0];
    const double bb = sums[1];
    res.rhs_norm = std::sqrt(bb);
    res.initial_residual_norm = std::sqrt(rr);
    res.flops = cg_setup_flops(prob.global_rows, prob.global_nnz);

    if (!std::isfinite(rr) || !std::isfinite(bb)) {
        res.status = CGStatus::NonFinite;
    } else if (bb == 0.0) {
        for (index_t i = 0; i < nl; ++i) x[i] = 0.0;
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

            exchange_p();
            double pAp = 0.0;
            for (index_t i = 0; i < nl; ++i) {
                double sum = 0.0;
                for (offset_t k = rp[i]; k < rp[i + 1]; ++k) sum += av[k] * p_ext[ci[k]];
                Ap[i] = sum;
                pAp += p[i] * sum;
            }
            allreduce_sum(&pAp, 1);
            if (!std::isfinite(pAp)) { res.status = CGStatus::NonFinite; break; }
            if (pAp <= 0.0) { res.status = CGStatus::Breakdown; break; }

            const double alpha = rr / pAp;
            double rr_new = 0.0;
            for (index_t i = 0; i < nl; ++i) {
                x[i] += alpha * p[i];
                const double ri = r[i] - alpha * Ap[i];
                r[i] = ri;
                rr_new += ri * ri;
            }
            allreduce_sum(&rr_new, 1);
            ++iter;
            if (!std::isfinite(rr_new)) { rr = rr_new; res.status = CGStatus::NonFinite; break; }

            const double beta = rr_new / rr;
            rr = rr_new;
            for (index_t i = 0; i < nl; ++i) p[i] = r[i] + beta * p[i];

            if (opts.record_history) res.residual_history.push_back(std::sqrt(rr) / res.rhs_norm);
        }
        res.iterations = iter;
        res.flops += static_cast<double>(iter) *
                     cg_iteration_flops(prob.global_rows, prob.global_nnz);
    }

    const double t_local = MPI_Wtime() - t_start;  // ---- timed region ends

    // Report the slowest rank's solve time (the time the job actually took)
    // and the rank-averaged communication times.
    double t_max = 0.0;
    MPI_Allreduce(&t_local, &t_max, 1, MPI_DOUBLE, MPI_MAX, comm);
    double comm_times[2] = {t_exchange, t_reduce};
    MPI_Allreduce(MPI_IN_PLACE, comm_times, 2, MPI_DOUBLE, MPI_SUM, comm);

    res.solve_seconds = t_max;
    res.comm_exchange_seconds = comm_times[0] / size;
    res.comm_reduction_seconds = comm_times[1] / size;
    res.final_residual_norm = std::sqrt(rr);
    res.relative_residual =
        res.rhs_norm > 0.0 ? res.final_residual_norm / res.rhs_norm : 0.0;
    return res;
}

}  // namespace hpcg

#endif  // HPCG_HAVE_MPI
