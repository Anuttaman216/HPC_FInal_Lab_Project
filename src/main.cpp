// main.cpp - entry point of the HPCG-inspired CG benchmark.
//
// In MPI builds MPI is always initialised with MPI_Init_thread requesting
// MPI_THREAD_FUNNELED (needed by the hybrid solver; harmless otherwise).
// Sequential and OpenMP modes may be launched directly (MPI singleton).
#include <exception>
#include <iostream>

#include "benchmark.hpp"

#ifdef HPCG_HAVE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
    int rank = 0, nprocs = 1;
#ifdef HPCG_HAVE_MPI
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) {
        std::cerr << "error: MPI_Init_thread failed\n";
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
#endif

    int code = 0;
    try {
        const hpcg::BenchmarkConfig cfg = hpcg::parse_arguments(argc, argv);
        if (cfg.help) {
            if (rank == 0) std::cout << hpcg::usage_text(argv[0]);
        } else {
#ifdef HPCG_HAVE_MPI
            if ((cfg.mode == hpcg::SolverMode::Hybrid) && provided < MPI_THREAD_FUNNELED) {
                if (rank == 0) {
                    std::cerr << "error: MPI library provides thread level " << provided
                              << " < MPI_THREAD_FUNNELED; hybrid mode unavailable\n";
                }
                code = 2;
            } else
#endif
                code = hpcg::run_benchmark(cfg);
        }
    } catch (const std::invalid_argument& e) {
        if (rank == 0) std::cerr << "error: " << e.what() << "\n";
        code = 1;
    } catch (const std::exception& e) {
        std::cerr << "[rank " << rank << "] fatal: " << e.what() << "\n";
#ifdef HPCG_HAVE_MPI
        // Another rank may be blocked in a collective: abort the whole job.
        if (nprocs > 1) MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        code = 1;
    }

#ifdef HPCG_HAVE_MPI
    MPI_Finalize();
#endif
    return code;
}
