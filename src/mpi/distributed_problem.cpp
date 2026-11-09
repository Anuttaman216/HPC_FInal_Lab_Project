// distributed_problem.cpp - row distribution, local matrix generation and the
// communication plan (Allgatherv or point-to-point halo exchange) used by the
// MPI and hybrid CG solvers.
#ifdef HPCG_HAVE_MPI

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "distributed.hpp"
#include "matrix_generator.hpp"

namespace hpcg {

std::string to_string(CommScheme scheme) {
    return scheme == CommScheme::Allgather ? "allgather" : "halo";
}

CommScheme parse_comm_scheme(const std::string& name) {
    if (name == "allgather" || name == "replicated") return CommScheme::Allgather;
    if (name == "halo" || name == "p2p") return CommScheme::Halo;
    throw std::invalid_argument("unknown communication scheme '" + name +
                                "' (expected allgather or halo)");
}

namespace {

void check_mpi(int code, const char* what) {
    if (code != MPI_SUCCESS) {
        throw std::runtime_error(std::string("MPI call failed: ") + what);
    }
}

}  // namespace

VectorExchange::VectorExchange(MPI_Comm comm, CommScheme scheme,
                               global_t global_n, global_t own_begin,
                               global_t own_end, global_t need_lo,
                               global_t need_hi)
    : comm_(comm), scheme_(scheme), global_n_(global_n),
      own_begin_(own_begin), own_end_(own_end) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    if (scheme == CommScheme::Allgather) {
        // Replicated vector: every rank holds all n entries.
        ext_lo_ = 0;
        ext_hi_ = global_n;
        int my_count = static_cast<int>(own_end - own_begin);
        counts_.assign(size, 0);
        displs_.assign(size, 0);
        check_mpi(MPI_Allgather(&my_count, 1, MPI_INT, counts_.data(), 1, MPI_INT, comm),
                  "Allgather(counts)");
        for (int q = 1; q < size; ++q) displs_[q] = displs_[q - 1] + counts_[q - 1];
        if (displs_[rank] != own_begin) {
            throw std::runtime_error("Allgatherv layout does not match row distribution");
        }
        return;
    }

    // Halo: the window is the union of the owned block and the needed range.
    ext_lo_ = std::min(own_begin, need_lo);
    ext_hi_ = std::max(own_end, need_hi);

    // Share every rank's owned block and window so that both the sender and
    // the receiver can compute identical overlap segments.
    global_t mine[4] = {own_begin, own_end, ext_lo_, ext_hi_};
    std::vector<global_t> all(4 * static_cast<std::size_t>(size));
    check_mpi(MPI_Allgather(mine, 4, MPI_INT64_T, all.data(), 4, MPI_INT64_T, comm),
              "Allgather(ranges)");

    for (int q = 0; q < size; ++q) {
        if (q == rank) continue;
        const global_t q_own_b = all[4 * q + 0], q_own_e = all[4 * q + 1];
        const global_t q_ext_lo = all[4 * q + 2], q_ext_hi = all[4 * q + 3];
        // What I need that q owns.
        const global_t rb = std::max(ext_lo_, q_own_b), re = std::min(ext_hi_, q_own_e);
        if (rb < re) recv_peers_.push_back({q, rb, re});
        // What q needs that I own.
        const global_t sb = std::max(q_ext_lo, own_begin), se = std::min(q_ext_hi, own_end);
        if (sb < se) send_peers_.push_back({q, sb, se});
    }
    requests_.resize(recv_peers_.size() + send_peers_.size());
}

global_t VectorExchange::recv_volume() const {
    if (scheme_ == CommScheme::Allgather) return global_n_ - own_size();
    global_t v = 0;
    for (const auto& s : recv_peers_) v += s.end - s.begin;
    return v;
}

void VectorExchange::exchange(std::vector<double>& ext) const {
    if (static_cast<global_t>(ext.size()) != ext_size()) {
        throw std::invalid_argument("exchange: window has wrong size");
    }
    if (scheme_ == CommScheme::Allgather) {
        // In place: this rank's block already sits at displs_[rank].
        check_mpi(MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ext.data(),
                                 counts_.data(), displs_.data(), MPI_DOUBLE, comm_),
                  "Allgatherv");
        return;
    }
    // Post receives first, then sends; the send and receive regions of the
    // window are disjoint (I only send what I own, I only receive what others
    // own), so the buffer can be shared.
    std::size_t nreq = 0;
    for (const auto& s : recv_peers_) {
        check_mpi(MPI_Irecv(ext.data() + (s.begin - ext_lo_), static_cast<int>(s.end - s.begin),
                            MPI_DOUBLE, s.peer, 0, comm_, &requests_[nreq++]),
                  "Irecv");
    }
    for (const auto& s : send_peers_) {
        check_mpi(MPI_Isend(ext.data() + (s.begin - ext_lo_), static_cast<int>(s.end - s.begin),
                            MPI_DOUBLE, s.peer, 0, comm_, &requests_[nreq++]),
                  "Isend");
    }
    if (nreq > 0) {
        check_mpi(MPI_Waitall(static_cast<int>(nreq), requests_.data(), MPI_STATUSES_IGNORE),
                  "Waitall");
    }
}

DistributedProblem build_distributed_problem(const GridDims& grid, MPI_Comm comm,
                                             CommScheme scheme) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DistributedProblem prob;
    prob.grid = grid;
    prob.global_rows = grid.rows();
    const auto range = block_range(prob.global_rows, size, rank);
    prob.first_row = range.first;

    // Each rank generates only its own rows (no global matrix, no scatter).
    prob.A = generate_poisson_3d_rows(grid, range.first, range.second);
    prob.b = generate_rhs(prob.A);

    // Contiguous range of global columns referenced by the local rows.
    global_t need_lo = range.first, need_hi = range.first;
    if (!prob.A.col_idx.empty()) {
        need_lo = *std::min_element(prob.A.col_idx.begin(), prob.A.col_idx.end());
        need_hi = *std::max_element(prob.A.col_idx.begin(), prob.A.col_idx.end()) + 1;
    }

    prob.exchange = VectorExchange(comm, scheme, prob.global_rows, range.first,
                                   range.second, need_lo, need_hi);
    // Re-index columns so they point directly into the extended window.
    prob.A.shift_columns(prob.exchange.ext_lo());
    prob.A.validate(prob.exchange.ext_size());

    global_t local_nnz = prob.A.nnz();
    MPI_Allreduce(&local_nnz, &prob.global_nnz, 1, MPI_INT64_T, MPI_SUM, comm);
    return prob;
}

double distributed_true_relative_residual(const DistributedProblem& prob,
                                          const std::vector<double>& x_local,
                                          MPI_Comm comm) {
    const VectorExchange& ex = prob.exchange;
    std::vector<double> x_ext(static_cast<std::size_t>(ex.ext_size()), 0.0);
    std::copy(x_local.begin(), x_local.end(), x_ext.begin() + ex.own_offset());
    ex.exchange(x_ext);
    std::vector<double> ax(x_local.size());
    spmv(prob.A, x_ext.data(), ax.data());
    double sums[2] = {0.0, 0.0};
    for (std::size_t i = 0; i < ax.size(); ++i) {
        const double d = prob.b[i] - ax[i];
        sums[0] += d * d;
        sums[1] += prob.b[i] * prob.b[i];
    }
    MPI_Allreduce(MPI_IN_PLACE, sums, 2, MPI_DOUBLE, MPI_SUM, comm);
    return sums[1] > 0.0 ? std::sqrt(sums[0] / sums[1]) : std::sqrt(sums[0]);
}

std::vector<double> gather_to_root(const std::vector<double>& local,
                                   global_t global_n, MPI_Comm comm, int root) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    int my_count = static_cast<int>(local.size());
    std::vector<int> counts(size), displs(size, 0);
    MPI_Gather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, root, comm);
    std::vector<double> global;
    if (rank == root) {
        for (int q = 1; q < size; ++q) displs[q] = displs[q - 1] + counts[q - 1];
        global.resize(static_cast<std::size_t>(global_n));
    }
    MPI_Gatherv(local.data(), my_count, MPI_DOUBLE, global.data(), counts.data(),
                displs.data(), MPI_DOUBLE, root, comm);
    return global;
}

}  // namespace hpcg

#endif  // HPCG_HAVE_MPI
