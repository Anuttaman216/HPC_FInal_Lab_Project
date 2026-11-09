// distributed.hpp - distributed-memory (MPI) and hybrid (MPI + OpenMP) CG.
//
// Data distribution: the rows of A (and the entries of x, b, r, Ap) are split
// into contiguous blocks with block_range(), which handles n not divisible by
// the number of ranks. Each rank generates only its own rows.
//
// SpMV needs the entries of p for every column referenced by the local rows.
// Those columns form one contiguous global range [ext_lo, ext_hi) (for the
// lexicographically ordered 7-point stencil it is the local block extended by
// one grid plane on each side). Each rank stores p in an "extended" window
// covering that range and refreshes it before every SpMV with one of two
// communication schemes:
//
//   Allgather : MPI_Allgatherv of the whole vector into every rank
//               (replicated vector, simple, O(n) traffic per rank per
//               iteration, does not scale).
//   Halo      : point-to-point MPI_Isend/MPI_Irecv of only the overlapping
//               parts of neighbouring blocks (O(nx*ny) traffic per rank).
//
// Global dot products always use MPI_Allreduce, so every rank sees the same
// alpha, beta and convergence decision and all ranks execute the same
// sequence of collective calls.
#pragma once

#ifdef HPCG_HAVE_MPI

#include <mpi.h>

#include <string>
#include <vector>

#include "cg_solver.hpp"
#include "common.hpp"
#include "csr_matrix.hpp"

namespace hpcg {

enum class CommScheme { Allgather, Halo };

std::string to_string(CommScheme scheme);
CommScheme parse_comm_scheme(const std::string& name);

// Communication plan for refreshing the extended window of a distributed
// vector. Built collectively; all ranks must call the constructor.
class VectorExchange {
public:
    VectorExchange() = default;
    VectorExchange(MPI_Comm comm, CommScheme scheme, global_t global_n,
                   global_t own_begin, global_t own_end, global_t need_lo,
                   global_t need_hi);

    // Fill the non-owned part of `ext` (size ext_size()) from the owners.
    // The owned part, ext[own_offset() .. own_offset()+own_size()), must
    // already hold this rank's current values. Collective.
    void exchange(std::vector<double>& ext) const;

    global_t ext_lo() const { return ext_lo_; }
    global_t ext_size() const { return ext_hi_ - ext_lo_; }
    global_t own_offset() const { return own_begin_ - ext_lo_; }
    global_t own_size() const { return own_end_ - own_begin_; }
    CommScheme scheme() const { return scheme_; }
    // Number of doubles this rank receives per exchange.
    global_t recv_volume() const;
    int neighbour_count() const { return static_cast<int>(recv_peers_.size()); }

private:
    struct Segment {
        int peer = 0;
        global_t begin = 0;  // global index
        global_t end = 0;
    };
    MPI_Comm comm_ = MPI_COMM_NULL;
    CommScheme scheme_ = CommScheme::Allgather;
    global_t global_n_ = 0;
    global_t own_begin_ = 0, own_end_ = 0;
    global_t ext_lo_ = 0, ext_hi_ = 0;
    std::vector<int> counts_, displs_;  // Allgatherv layout
    std::vector<Segment> recv_peers_;   // Halo: what I receive from whom
    std::vector<Segment> send_peers_;   // Halo: what I send to whom
    // Request handles reused by every exchange (no allocation per iteration).
    mutable std::vector<MPI_Request> requests_;
};

// Everything one rank needs to run distributed CG.
struct DistributedProblem {
    GridDims grid;
    global_t global_rows = 0;
    global_t global_nnz = 0;
    global_t first_row = 0;
    CSRMatrix A;              // local rows, columns shifted into the window
    std::vector<double> b;    // local part of the right-hand side
    VectorExchange exchange;  // plan for the extended window of p
};

// Collective: each rank generates its rows, builds the communication plan
// and the local right-hand side.
DistributedProblem build_distributed_problem(const GridDims& grid,
                                             MPI_Comm comm, CommScheme scheme);

// Pure MPI solver (no OpenMP inside). x_local: local initial guess/solution.
CGResult cg_solve_mpi(const DistributedProblem& prob,
                      std::vector<double>& x_local, const CGOptions& opts,
                      MPI_Comm comm);

// Hybrid solver: MPI between ranks, OpenMP threads inside each rank.
// Requires MPI to provide at least MPI_THREAD_FUNNELED; all MPI calls are
// made by the master thread outside OpenMP parallel regions.
CGResult cg_solve_hybrid(const DistributedProblem& prob,
                         std::vector<double>& x_local, const CGOptions& opts,
                         MPI_Comm comm);

// ||b - A x|| / ||b|| for a distributed x (collective, not timed).
double distributed_true_relative_residual(const DistributedProblem& prob,
                                          const std::vector<double>& x_local,
                                          MPI_Comm comm);

// Gather a distributed vector onto `root` (returns empty on other ranks).
std::vector<double> gather_to_root(const std::vector<double>& local,
                                   global_t global_n, MPI_Comm comm, int root);

}  // namespace hpcg

#endif  // HPCG_HAVE_MPI
