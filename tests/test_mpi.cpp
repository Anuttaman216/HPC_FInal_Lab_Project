// test_mpi.cpp - distributed CG tests. Run with 1, 2, 3 and 4 ranks (CTest).
//
// To avoid deadlocks, every test first performs all collective operations
// and only then evaluates local conditions; the local verdicts are combined
// with MPI_Allreduce(LAND) so that all ranks agree on PASS/FAIL.
#include <mpi.h>

#include <algorithm>
#include <cmath>

#include "cg_solver.hpp"
#include "distributed.hpp"
#include "matrix_generator.hpp"
#include "test_utils.hpp"

using namespace hpcg;
using testing_util::run_test;

namespace {

int g_rank = 0, g_size = 1;

bool all_ranks(bool local) {
    int in = local ? 1 : 0, out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    return out == 1;
}

// Sequential reference solution computed independently on every rank.
std::vector<double> reference_solution(const GridDims& g, const CGOptions& o) {
    const CSRMatrix A = generate_poisson_3d(g);
    const std::vector<double> b = generate_rhs(A);
    std::vector<double> x(A.local_rows, 0.0);
    cg_solve_sequential(A, b, x, o);
    return x;
}

}  // namespace

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    const bool root = g_rank == 0;
    if (root) std::cout << "test_mpi running on " << g_size << " rank(s)\n";

    // Grids chosen so that most rank counts give an uneven distribution and
    // 4x4x2 gives blocks smaller than one grid plane (halo spans >1 rank).
    const std::vector<GridDims> grids{{7, 5, 9}, {11, 7, 5}, {4, 4, 2}, {16, 16, 16}};

    run_test("uneven row distribution covers every row exactly once", [&] {
        for (const auto& g : grids) {
            const DistributedProblem prob = build_distributed_problem(g, MPI_COMM_WORLD, CommScheme::Halo);
            const auto range = block_range(g.rows(), g_size, g_rank);
            global_t local = prob.A.local_rows, total = 0;
            MPI_Allreduce(&local, &total, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
            global_t mx = 0, mn = 0;
            MPI_Allreduce(&local, &mx, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
            MPI_Allreduce(&local, &mn, 1, MPI_INT64_T, MPI_MIN, MPI_COMM_WORLD);
            const bool ok = total == g.rows() && prob.first_row == range.first &&
                            local == range.second - range.first && mx - mn <= 1 &&
                            prob.global_nnz == poisson_nnz(g);
            CHECK(all_ranks(ok));
        }
    }, root);

    for (CommScheme scheme : {CommScheme::Allgather, CommScheme::Halo}) {
        run_test("vector exchange delivers correct values (" + to_string(scheme) + ")", [&] {
            for (const auto& g : grids) {
                const DistributedProblem prob = build_distributed_problem(g, MPI_COMM_WORLD, scheme);
                const VectorExchange& ex = prob.exchange;
                std::vector<double> ext(static_cast<std::size_t>(ex.ext_size()), -1.0);
                for (global_t i = 0; i < ex.own_size(); ++i) {
                    ext[ex.own_offset() + i] = static_cast<double>(prob.first_row + i);
                }
                ex.exchange(ext);
                bool ok = true;
                // Every stored column must now see the value "global index".
                for (std::size_t k = 0; k < prob.A.col_idx.size(); ++k) {
                    const index_t c = prob.A.col_idx[k];
                    if (ext[c] != static_cast<double>(c + ex.ext_lo())) ok = false;
                }
                if (scheme == CommScheme::Allgather) {
                    for (global_t i = 0; i < ex.ext_size(); ++i)
                        if (ext[i] != static_cast<double>(i)) ok = false;
                }
                CHECK(all_ranks(ok));
            }
        }, root);
    }

    run_test("halo exchange moves less data than allgather", [&] {
        const GridDims g{16, 16, 16};
        const DistributedProblem h = build_distributed_problem(g, MPI_COMM_WORLD, CommScheme::Halo);
        const DistributedProblem a = build_distributed_problem(g, MPI_COMM_WORLD, CommScheme::Allgather);
        const bool ok = h.exchange.recv_volume() <= a.exchange.recv_volume() &&
                        h.exchange.recv_volume() <= 2 * g.plane();
        CHECK(all_ranks(ok));
    }, root);

    CGOptions o;
    o.tolerance = 1e-10;
    o.max_iterations = 2000;

    for (CommScheme scheme : {CommScheme::Allgather, CommScheme::Halo}) {
        for (bool hybrid : {false, true}) {
            const std::string name = std::string(hybrid ? "hybrid" : "mpi") + "/" + to_string(scheme);
            run_test(name + " CG matches the sequential solution", [&] {
                for (const auto& g : grids) {
                    const DistributedProblem prob = build_distributed_problem(g, MPI_COMM_WORLD, scheme);
                    std::vector<double> x(prob.A.local_rows, 0.0);
                    const CGResult r = hybrid ? cg_solve_hybrid(prob, x, o, MPI_COMM_WORLD)
                                              : cg_solve_mpi(prob, x, o, MPI_COMM_WORLD);
                    const double true_res = distributed_true_relative_residual(prob, x, MPI_COMM_WORLD);
                    const std::vector<double> x_all = gather_to_root(x, prob.global_rows, MPI_COMM_WORLD, 0);
                    int it_min = 0, it_max = 0;
                    MPI_Allreduce(&r.iterations, &it_min, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                    MPI_Allreduce(&r.iterations, &it_max, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

                    // --- local verdicts (no more collectives below) ---------
                    const std::vector<double> ref = reference_solution(g, o);
                    bool ok = r.converged() && true_res < 1e-9 && it_min == it_max;
                    // compare the local slice against the sequential solution
                    for (index_t i = 0; i < prob.A.local_rows; ++i) {
                        if (std::fabs(x[i] - ref[prob.first_row + i]) > 1e-8) ok = false;
                    }
                    if (root) {
                        ok = ok && x_all.size() == ref.size() && testing_util::max_abs_diff(x_all, ref) < 1e-8;
                    }
                    CHECK(all_ranks(ok));
                }
            }, root);
        }
    }

    run_test("distributed fixed-iteration run: same flops and iterations as sequential", [&] {
        const GridDims g{11, 7, 5};
        CGOptions f; f.tolerance = 0.0; f.max_iterations = 20;
        const DistributedProblem prob = build_distributed_problem(g, MPI_COMM_WORLD, CommScheme::Halo);
        std::vector<double> x(prob.A.local_rows, 0.0);
        const CGResult r = cg_solve_mpi(prob, x, f, MPI_COMM_WORLD);
        const CSRMatrix A = generate_poisson_3d(g);
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> xs(A.local_rows, 0.0);
        const CGResult s = cg_solve_sequential(A, b, xs, f);
        bool ok = r.iterations == 20 && r.flops == s.flops;
        for (index_t i = 0; i < prob.A.local_rows; ++i)
            if (std::fabs(x[i] - xs[prob.first_row + i]) > 1e-10) ok = false;
        CHECK(all_ranks(ok));
    }, root);

    run_test("distributed zero RHS and breakdown are handled on all ranks", [&] {
        DistributedProblem prob = build_distributed_problem({7, 5, 9}, MPI_COMM_WORLD, CommScheme::Halo);
        std::fill(prob.b.begin(), prob.b.end(), 0.0);
        std::vector<double> x(prob.A.local_rows, 2.0);
        const CGResult r0 = cg_solve_mpi(prob, x, o, MPI_COMM_WORLD);
        bool ok = r0.status == CGStatus::ZeroRhs;
        for (double v : x) if (v != 0.0) ok = false;

        DistributedProblem neg = build_distributed_problem({7, 5, 9}, MPI_COMM_WORLD, CommScheme::Allgather);
        for (auto& v : neg.A.values) v = -v;
        std::fill(neg.b.begin(), neg.b.end(), 1.0);
        std::vector<double> xn(neg.A.local_rows, 0.0);
        const CGResult r1 = cg_solve_hybrid(neg, xn, o, MPI_COMM_WORLD);
        ok = ok && r1.status == CGStatus::Breakdown;
        CHECK(all_ranks(ok));
    }, root);

    run_test("MPI thread level supports the hybrid solver", [&] {
        CHECK(all_ranks(provided >= MPI_THREAD_FUNNELED));
    }, root);

    int local_fail = testing_util::failure_count(), total_fail = 0;
    MPI_Allreduce(&local_fail, &total_fail, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (root) {
        std::cout << "test_mpi (" << g_size << " ranks): "
                  << (total_fail == 0 ? "ALL TESTS PASSED" : "FAILURES") << "\n";
    }
    MPI_Finalize();
    return total_fail == 0 ? 0 : 1;
}
