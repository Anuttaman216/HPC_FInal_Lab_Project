// cg_hybrid.cpp - hybrid MPI + OpenMP Conjugate Gradient.
//
// Same algorithm and data distribution as cg_mpi.cpp, but every local loop
// is an OpenMP worksharing loop (static schedule, reductions for the local
// dot products). Threading model: MPI_THREAD_FUNNELED - all MPI calls are
// issued by the master thread *between* parallel regions, never inside one,
// so there are no concurrent MPI calls. The function verifies the provided
// thread level and that it is called from the MPI main thread.
#if defined(HPCG_HAVE_MPI) && defined(HPCG_HAVE_OPENMP)

#include <omp.h>

#include <cmath>
#include <stdexcept>

#include "distributed.hpp"

namespace hpcg {

CGResult cg_solve_hybrid(const DistributedProblem& prob, std::vector<double>& x_local,
                         const CGOptions& opts, MPI_Comm comm) {
    int provided = MPI_THREAD_SINGLE, is_main = 0;
    MPI_Query_thread(&provided);
    MPI_Is_thread_main(&is_main);
    if (provided < MPI_THREAD_FUNNELED) {
        throw std::runtime_error("hybrid CG needs MPI_THREAD_FUNNELED support");
    }
    if (!is_main) {
        throw std::runtime_error("hybrid CG must be called from the MPI main thread");
    }

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

    std::vector<double> p_ext(static_cast<std::size_t>(ex.ext_size()), 0.0);
    std::vector<double> r_vec(nl, 0.0), ap_vec(nl, 0.0);
    const double* pe = p_ext.data();
    double* p = p_ext.data() + ex.own_offset();
    double* r = r_vec.data();
    double* Ap = ap_vec.data();

    CGResult res;
    if (opts.record_history) res.residual_history.reserve(opts.max_iterations + 1);
    double t_exchange = 0.0, t_reduce = 0.0;

    auto exchange_p = [&]() {  // master thread, outside parallel regions
        const double t = MPI_Wtime();
        ex.exchange(p_ext);
        t_exchange += MPI_Wtime() - t;
    };
    auto allreduce_sum = [&](double* vals, int count) {
        const double t = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, vals, count, MPI_DOUBLE, MPI_SUM, comm);
        t_reduce += MPI_Wtime() - t;
    };

    MPI_Barrier(comm);
    const double t_start = MPI_Wtime();  // ---- timed region starts -------

#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < nl; ++i) p[i] = x[i];
    exchange_p();
    double rr = 0.0, bb = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr, bb)
    for (index_t i = 0; i < nl; ++i) {
        double ax = 0.0;
        for (offset_t k = rp[i]; k < rp[i + 1]; ++k) ax += av[k] * pe[ci[k]];
        const double ri = b[i] - ax;
        r[i] = ri;
        rr += ri * ri;
        bb += b[i] * b[i];
    }
#pragma omp parallel for schedule(static)
    for (index_t i = 0; i < nl; ++i) p[i] = r[i];
    double sums[2] = {rr, bb};
    allreduce_sum(sums, 2);
    rr = sums[0];
    bb = sums[1];
    res.rhs_norm = std::sqrt(bb);
    res.initial_residual_norm = std::sqrt(rr);
    res.flops = cg_setup_flops(prob.global_rows, prob.global_nnz);

    if (!std::isfinite(rr) || !std::isfinite(bb)) {
        res.status = CGStatus::NonFinite;
    } else if (bb == 0.0) {
#pragma omp parallel for schedule(static)
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
#pragma omp parallel for schedule(static) reduction(+ : pAp)
            for (index_t i = 0; i < nl; ++i) {
                double sum = 0.0;
                for (offset_t k = rp[i]; k < rp[i + 1]; ++k) sum += av[k] * pe[ci[k]];
                Ap[i] = sum;
                pAp += p[i] * sum;
            }
            allreduce_sum(&pAp, 1);
            if (!std::isfinite(pAp)) { res.status = CGStatus::NonFinite; break; }
            if (pAp <= 0.0) { res.status = CGStatus::Breakdown; break; }

            const double alpha = rr / pAp;
            double rr_new = 0.0;
#pragma omp parallel for schedule(static) reduction(+ : rr_new)
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
#pragma omp parallel for schedule(static)
            for (index_t i = 0; i < nl; ++i) p[i] = r[i] + beta * p[i];

            if (opts.record_history) res.residual_history.push_back(std::sqrt(rr) / res.rhs_norm);
        }
        res.iterations = iter;
        res.flops += static_cast<double>(iter) *
                     cg_iteration_flops(prob.global_rows, prob.global_nnz);
    }

    const double t_local = MPI_Wtime() - t_start;  // ---- timed region ends

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

#elif defined(HPCG_HAVE_MPI)

#include <stdexcept>

#include "distributed.hpp"

namespace hpcg {
CGResult cg_solve_hybrid(const DistributedProblem&, std::vector<double>&,
                         const CGOptions&, MPI_Comm) {
    throw std::runtime_error("hybrid mode needs a build with OpenMP enabled");
}
}  // namespace hpcg

#endif
