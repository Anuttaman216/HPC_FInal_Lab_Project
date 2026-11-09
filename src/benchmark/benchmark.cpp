// benchmark.cpp - argument parsing, problem setup, repeated timed solves,
// verification and CSV reporting for all four solver modes.
#include "benchmark.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "cg_solver.hpp"
#include "matrix_generator.hpp"
#include "metrics.hpp"

#ifdef HPCG_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef HPCG_HAVE_MPI
#include <mpi.h>

#include "distributed.hpp"
#endif

namespace hpcg {

std::string to_string(SolverMode mode) {
    switch (mode) {
        case SolverMode::Sequential: return "sequential";
        case SolverMode::OpenMP: return "openmp";
        case SolverMode::MPI: return "mpi";
        case SolverMode::Hybrid: return "hybrid";
    }
    return "unknown";
}

namespace {

SolverMode parse_mode(const std::string& s) {
    if (s == "sequential" || s == "seq") return SolverMode::Sequential;
    if (s == "openmp" || s == "omp") return SolverMode::OpenMP;
    if (s == "mpi") return SolverMode::MPI;
    if (s == "hybrid") return SolverMode::Hybrid;
    throw std::invalid_argument("unknown mode '" + s +
                                "' (expected sequential|openmp|mpi|hybrid)");
}

int parse_int(const std::string& opt, const std::string& v, int min_value) {
    std::size_t used = 0;
    long value = 0;
    try {
        value = std::stol(v, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(opt + " expects an integer, got '" + v + "'");
    }
    if (used != v.size() || value < min_value || value > 1000000000L) {
        throw std::invalid_argument(opt + " expects an integer >= " +
                                    std::to_string(min_value) + ", got '" + v + "'");
    }
    return static_cast<int>(value);
}

double parse_double(const std::string& opt, const std::string& v) {
    std::size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(v, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(opt + " expects a number, got '" + v + "'");
    }
    if (used != v.size() || !std::isfinite(value)) {
        throw std::invalid_argument(opt + " expects a finite number, got '" + v + "'");
    }
    return value;
}

std::string iso_timestamp() {
    const std::time_t now = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
    return buf;
}

std::string csv_escape(const std::string& s) {
    if (s.find_first_of(",\"\n") == std::string::npos) return s;
    std::string out = "\"";
    for (char c : s) out += (c == '"') ? std::string("\"\"") : std::string(1, c);
    return out + "\"";
}

const char* kCsvHeader =
    "timestamp,label,mode,comm,nx,ny,nz,rows,nnz,mpi_procs,omp_threads,"
    "processing_units,repetitions,iterations,tolerance,max_iterations,status,"
    "converged,relative_residual,true_relative_residual,max_abs_error,"
    "time_median_s,time_min_s,time_max_s,time_mean_s,time_stddev_s,"
    "time_per_iter_ms,comm_exchange_median_s,comm_reduction_median_s,"
    "comm_fraction,gflops,setup_s,memory_mb,hostname";

bool file_is_empty(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    return !in.good() || in.tellg() <= 0;
}

// Collected per-configuration results (identical on all ranks).
struct RunSummary {
    std::vector<CGResult> runs;
    double setup_seconds = 0.0;
    double true_rel_residual = 0.0;
    double max_abs_error = 0.0;  // vs. exact solution x* = 1
    global_t rows = 0, nnz = 0;
    double memory_bytes = 0.0;
    std::vector<double> history;
};

}  // namespace

std::string usage_text(const char* program) {
    std::ostringstream o;
    o << "HPCG-inspired Conjugate Gradient benchmark (educational, NOT official HPCG)\n\n"
      << "Usage: " << program << " [options]\n\n"
      << "Solver selection:\n"
      << "  --mode M            sequential | openmp | mpi | hybrid (default sequential)\n"
      << "  --comm C            MPI vector exchange: allgather | halo (default allgather)\n"
      << "  --threads T         OpenMP threads (default: OMP_NUM_THREADS or all)\n\n"
      << "Problem (3D 7-point Poisson, exact solution x = 1):\n"
      << "  --grid N            N x N x N grid (default 32)\n"
      << "  --nx X --ny Y --nz Z  anisotropic grid (overrides --grid per axis)\n\n"
      << "Stopping criteria:\n"
      << "  --tol EPS           relative residual ||r||/||b|| (default 1e-8;\n"
      << "                      0 => run exactly --max-iters iterations)\n"
      << "  --max-iters K       iteration limit (default 1000)\n\n"
      << "Measurement:\n"
      << "  --reps R            timed repetitions, median reported (default 3)\n"
      << "  --warmup W          untimed warm-up solves (default 1)\n"
      << "  --csv FILE          append one summary row to FILE\n"
      << "  --label TEXT        experiment tag stored in the CSV\n"
      << "  --history FILE      write the residual history of the last run\n\n"
      << "Safety / misc:\n"
      << "  --mem-fraction F    refuse problems needing > F * available RAM (0.6)\n"
      << "  --force             skip the memory check\n"
      << "  --quiet             only print the final summary\n"
      << "  --info              print build/hardware information and exit\n"
      << "  --help              this text\n\n"
      << "Examples:\n"
      << "  " << program << " --mode sequential --grid 30\n"
      << "  OMP_NUM_THREADS=4 " << program << " --mode openmp --grid 30\n"
      << "  mpirun -np 4 " << program << " --mode mpi --grid 30 --comm halo\n"
      << "  OMP_NUM_THREADS=2 mpirun -np 2 " << program << " --mode hybrid --grid 30\n";
    return o.str();
}

BenchmarkConfig parse_arguments(int argc, char** argv) {
    BenchmarkConfig c;
    int grid = -1, nx = -1, ny = -1, nz = -1;
    std::string history_path;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::invalid_argument(a + " needs a value");
            return argv[++i];
        };
        if (a == "--help" || a == "-h") c.help = true;
        else if (a == "--info") c.info_only = true;
        else if (a == "--quiet" || a == "-q") c.quiet = true;
        else if (a == "--force") c.force = true;
        else if (a == "--mode") c.mode = parse_mode(next());
        else if (a == "--comm") c.comm = next();
        else if (a == "--grid") grid = parse_int(a, next(), 1);
        else if (a == "--nx") nx = parse_int(a, next(), 1);
        else if (a == "--ny") ny = parse_int(a, next(), 1);
        else if (a == "--nz") nz = parse_int(a, next(), 1);
        else if (a == "--max-iters") c.cg.max_iterations = parse_int(a, next(), 0);
        else if (a == "--tol") c.cg.tolerance = parse_double(a, next());
        else if (a == "--threads") c.threads = parse_int(a, next(), 1);
        else if (a == "--reps") c.repetitions = parse_int(a, next(), 1);
        else if (a == "--warmup") c.warmup = parse_int(a, next(), 0);
        else if (a == "--csv") c.csv_path = next();
        else if (a == "--label") c.label = next();
        else if (a == "--history") history_path = next();
        else if (a == "--mem-fraction") c.memory_fraction = parse_double(a, next());
        else throw std::invalid_argument("unknown option '" + a + "' (see --help)");
    }
    if (grid > 0) c.grid = GridDims{grid, grid, grid};
    if (nx > 0) c.grid.nx = nx;
    if (ny > 0) c.grid.ny = ny;
    if (nz > 0) c.grid.nz = nz;
    if (c.cg.tolerance < 0.0) throw std::invalid_argument("--tol must be >= 0");
    if (c.memory_fraction <= 0.0 || c.memory_fraction > 1.0) {
        throw std::invalid_argument("--mem-fraction must be in (0, 1]");
    }
    if (c.comm != "allgather" && c.comm != "halo") {
        throw std::invalid_argument("--comm must be allgather or halo");
    }
    c.cg.record_history = !history_path.empty();
    c.history_path = history_path;
    return c;
}

int run_benchmark(const BenchmarkConfig& cfg) {
    int rank = 0, nprocs = 1;
#ifdef HPCG_HAVE_MPI
    MPI_Comm comm = MPI_COMM_WORLD;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
#endif
    const bool root = (rank == 0);
    const EnvironmentInfo env = detect_environment();

    if (cfg.info_only) {
        if (root) {
            std::cout << "hostname          : " << env.hostname << "\n"
                      << "compiler          : " << env.compiler << "\n"
                      << "build type        : " << env.build_type << "\n"
                      << "build flags       : " << env.build_flags << "\n"
                      << "OpenMP            : " << env.openmp_version << "\n"
                      << "MPI library       : " << env.mpi_version << "\n"
                      << "logical CPUs      : " << env.logical_cpus << "\n";
#ifdef HPCG_HAVE_OPENMP
            std::cout << "omp max threads   : " << omp_get_max_threads() << "\n";
#endif
            if (env.total_memory_bytes > 0) {
                std::cout << std::fixed << std::setprecision(2)
                          << "memory total      : " << env.total_memory_bytes / 1073741824.0 << " GiB\n"
                          << "memory available  : " << env.available_memory_bytes / 1073741824.0 << " GiB\n";
            }
        }
        return 0;
    }

    // ---- validate the mode against the build and the launch --------------
    auto fail = [&](const std::string& msg) {
        if (root) std::cerr << "error: " << msg << "\n";
        return 2;
    };
    const bool shared_mode = cfg.mode == SolverMode::Sequential || cfg.mode == SolverMode::OpenMP;
    if (shared_mode && nprocs != 1) {
        return fail(to_string(cfg.mode) + " mode must be run with a single process");
    }
    if ((cfg.mode == SolverMode::OpenMP || cfg.mode == SolverMode::Hybrid) && !openmp_available()) {
        return fail("this build has no OpenMP support; " + to_string(cfg.mode) + " mode unavailable");
    }
#ifndef HPCG_HAVE_MPI
    if (!shared_mode) return fail("this build has no MPI support; rebuild with -DHPCG_ENABLE_MPI=ON");
#endif

    int threads = 1;
#ifdef HPCG_HAVE_OPENMP
    if (cfg.threads > 0) omp_set_num_threads(cfg.threads);
    if (cfg.mode == SolverMode::OpenMP || cfg.mode == SolverMode::Hybrid) {
        threads = omp_get_max_threads();
    }
#endif
    const int processing_units = nprocs * threads;
    const std::string comm_name = shared_mode ? "none" : cfg.comm;

    // ---- memory safety check (all ranks of the job assumed on this node) --
    const MatrixInfo info = describe_poisson(cfg.grid);
    double mem_needed = info.total_bytes();
    if (!shared_mode && cfg.comm == "allgather") {
        // every rank holds a full replica of p
        mem_needed += static_cast<double>(nprocs) * static_cast<double>(info.rows) * sizeof(double);
    }
    // Rank 0 decides and broadcasts so that all ranks take the same branch.
    int too_big = (!cfg.force && env.available_memory_bytes > 0 &&
                   mem_needed > cfg.memory_fraction * env.available_memory_bytes) ? 1 : 0;
#ifdef HPCG_HAVE_MPI
    MPI_Bcast(&too_big, 1, MPI_INT, 0, comm);
#endif
    if (too_big) {
        std::ostringstream m;
        m << std::fixed << std::setprecision(1) << "problem " << cfg.grid.to_string()
          << " needs ~" << mem_needed / 1048576.0 << " MiB but only "
          << env.available_memory_bytes / 1048576.0 << " MiB are available (limit "
          << cfg.memory_fraction * 100 << "%). Use a smaller --grid or --force.";
        return fail(m.str());
    }

    if (root && !cfg.quiet) {
        std::cout << "=== HPCG-inspired CG benchmark (educational; not official HPCG) ===\n"
                  << std::fixed << std::setprecision(4)
                  << "mode=" << to_string(cfg.mode) << " comm=" << comm_name
                  << " grid=" << cfg.grid.to_string() << " rows=" << info.rows
                  << " nnz=" << info.nnz << " sparsity=" << info.sparsity_percent << "%"
                  << " mpi_procs=" << nprocs << " omp_threads=" << threads
                  << " est_mem=" << std::setprecision(1) << mem_needed / 1048576.0 << "MiB\n";
    }

    RunSummary sum;
    sum.memory_bytes = mem_needed;
    sum.rows = info.rows;
    sum.nnz = info.nnz;
    const int total_runs = cfg.warmup + cfg.repetitions;

    if (shared_mode) {
        WallTimer setup;
        const CSRMatrix A = generate_poisson_3d(cfg.grid);
        const std::vector<double> b = generate_rhs(A);
        sum.setup_seconds = setup.seconds();
        std::vector<double> x(static_cast<std::size_t>(A.local_rows), 0.0);
        for (int run = 0; run < total_runs; ++run) {
            std::fill(x.begin(), x.end(), 0.0);
            CGResult r = cfg.mode == SolverMode::Sequential ? cg_solve_sequential(A, b, x, cfg.cg)
                                                            : cg_solve_openmp(A, b, x, cfg.cg);
            if (run >= cfg.warmup) {
                if (!cfg.quiet) {
                    std::cout << "  rep " << (run - cfg.warmup + 1) << ": iters=" << r.iterations
                              << std::scientific << std::setprecision(3)
                              << " relres=" << r.relative_residual << std::fixed
                              << std::setprecision(6) << " time=" << r.solve_seconds << "s"
                              << std::setprecision(3) << " GFLOPS=" << r.gflops() << "\n";
                }
                sum.runs.push_back(std::move(r));
            }
        }
        sum.true_rel_residual = true_relative_residual(A, b, x);
        for (double v : x) sum.max_abs_error = std::max(sum.max_abs_error, std::fabs(v - 1.0));
        sum.history = sum.runs.back().residual_history;
    }
#ifdef HPCG_HAVE_MPI
    else {
        const CommScheme scheme = parse_comm_scheme(cfg.comm);
        MPI_Barrier(comm);
        const double t0 = MPI_Wtime();
        const DistributedProblem prob = build_distributed_problem(cfg.grid, comm, scheme);
        double setup_local = MPI_Wtime() - t0;
        MPI_Allreduce(&setup_local, &sum.setup_seconds, 1, MPI_DOUBLE, MPI_MAX, comm);
        std::vector<double> x(static_cast<std::size_t>(prob.A.local_rows), 0.0);
        for (int run = 0; run < total_runs; ++run) {
            std::fill(x.begin(), x.end(), 0.0);
            CGResult r = cfg.mode == SolverMode::MPI ? cg_solve_mpi(prob, x, cfg.cg, comm)
                                                     : cg_solve_hybrid(prob, x, cfg.cg, comm);
            if (run >= cfg.warmup) {
                if (root && !cfg.quiet) {
                    std::cout << "  rep " << (run - cfg.warmup + 1) << ": iters=" << r.iterations
                              << std::scientific << std::setprecision(3)
                              << " relres=" << r.relative_residual << std::fixed
                              << std::setprecision(6) << " time=" << r.solve_seconds << "s"
                              << " comm=" << r.comm_seconds() << "s" << std::setprecision(3)
                              << " GFLOPS=" << r.gflops() << "\n";
                }
                sum.runs.push_back(std::move(r));
            }
        }
        sum.true_rel_residual = distributed_true_relative_residual(prob, x, comm);
        double local_err = 0.0;
        for (double v : x) local_err = std::max(local_err, std::fabs(v - 1.0));
        MPI_Allreduce(&local_err, &sum.max_abs_error, 1, MPI_DOUBLE, MPI_MAX, comm);
        sum.history = sum.runs.back().residual_history;
        if (root && !cfg.quiet) {
            std::cout << "  rank 0 receives " << prob.exchange.recv_volume()
                      << " doubles per exchange ";
            if (scheme == CommScheme::Halo) {
                std::cout << "from " << prob.exchange.neighbour_count() << " halo neighbour(s)\n";
            } else {
                std::cout << "via MPI_Allgatherv from all other ranks\n";
            }
        }
    }
#endif

    // ---- aggregate ---------------------------------------------------------
    std::vector<double> times, exch, red;
    bool iterations_consistent = true;
    for (const auto& r : sum.runs) {
        times.push_back(r.solve_seconds);
        exch.push_back(r.comm_exchange_seconds);
        red.push_back(r.comm_reduction_seconds);
        if (r.iterations != sum.runs.front().iterations) iterations_consistent = false;
    }
    const SampleStats ts = compute_stats(times);
    const SampleStats es = compute_stats(exch);
    const SampleStats rs = compute_stats(red);
    const CGResult& last = sum.runs.back();
    const int iters = last.iterations;
    const double flops = cg_total_flops(sum.rows, sum.nnz, iters);
    const double gf = gflops(flops, ts.median);
    const double comm_fraction = ts.median > 0 ? (es.median + rs.median) / ts.median : 0.0;
    const bool fixed_iteration_mode = cfg.cg.tolerance == 0.0;
    const bool ok_status = last.converged() ||
                           (fixed_iteration_mode && last.status == CGStatus::MaxIterations);

    if (root) {
        std::cout << std::fixed << std::setprecision(6)
                  << "RESULT mode=" << to_string(cfg.mode) << " comm=" << comm_name
                  << " grid=" << cfg.grid.to_string() << " rows=" << sum.rows
                  << " nnz=" << sum.nnz << " procs=" << nprocs << " threads=" << threads
                  << " iters=" << iters << " status=" << to_string(last.status)
                  << std::scientific << std::setprecision(3)
                  << " relres=" << last.relative_residual
                  << " true_relres=" << sum.true_rel_residual
                  << " max_err=" << sum.max_abs_error << std::fixed << std::setprecision(6)
                  << " t_median=" << ts.median << "s"
                  << " comm_frac=" << std::setprecision(3) << comm_fraction
                  << " GFLOPS=" << gf << " mem=" << std::setprecision(1)
                  << sum.memory_bytes / 1048576.0 << "MiB\n";
        if (!iterations_consistent) {
            std::cout << "warning: iteration count differed between repetitions\n";
        }
        if (!ok_status) {
            std::cerr << "warning: solver did not converge (status "
                      << to_string(last.status) << ")\n";
        }

        if (!cfg.csv_path.empty()) {
            const bool header = file_is_empty(cfg.csv_path);
            std::ofstream out(cfg.csv_path, std::ios::app);
            if (!out) {
                std::cerr << "error: cannot open " << cfg.csv_path << " for writing\n";
            } else {
                if (header) out << kCsvHeader << "\n";
                out << std::setprecision(9);
                out << iso_timestamp() << ',' << csv_escape(cfg.label) << ','
                    << to_string(cfg.mode) << ',' << comm_name << ',' << cfg.grid.nx << ','
                    << cfg.grid.ny << ',' << cfg.grid.nz << ',' << sum.rows << ',' << sum.nnz
                    << ',' << nprocs << ',' << threads << ',' << processing_units << ','
                    << ts.count << ',' << iters << ',' << cfg.cg.tolerance << ','
                    << cfg.cg.max_iterations << ',' << to_string(last.status) << ','
                    << (ok_status ? 1 : 0) << ',' << last.relative_residual << ','
                    << sum.true_rel_residual << ',' << sum.max_abs_error << ','
                    << ts.median << ',' << ts.min << ',' << ts.max << ',' << ts.mean << ','
                    << ts.stddev << ',' << (iters > 0 ? 1000.0 * ts.median / iters : 0.0)
                    << ',' << es.median << ',' << rs.median << ',' << comm_fraction << ','
                    << gf << ',' << sum.setup_seconds << ','
                    << sum.memory_bytes / 1048576.0 << ',' << csv_escape(env.hostname) << "\n";
            }
        }
        if (!cfg.history_path.empty()) {
            std::ofstream h(cfg.history_path);
            h << "iteration,relative_residual\n" << std::setprecision(12);
            for (std::size_t k = 0; k < sum.history.size(); ++k) {
                h << k << ',' << sum.history[k] << "\n";
            }
        }
    }
    if (last.status == CGStatus::Breakdown || last.status == CGStatus::NonFinite) return 4;
    return ok_status ? 0 : 3;
}

}  // namespace hpcg
