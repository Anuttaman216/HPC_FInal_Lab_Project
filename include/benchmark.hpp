// benchmark.hpp - command-line driven benchmark runner.
#pragma once

#include <string>
#include <vector>

#include "cg_solver.hpp"
#include "common.hpp"

namespace hpcg {

enum class SolverMode { Sequential, OpenMP, MPI, Hybrid };

std::string to_string(SolverMode mode);

struct BenchmarkConfig {
    SolverMode mode = SolverMode::Sequential;
    GridDims grid{32, 32, 32};
    CGOptions cg;
    int repetitions = 3;
    int warmup = 1;          // untimed warm-up solves before measuring
    int threads = 0;         // 0 => OMP_NUM_THREADS / runtime default
    std::string comm = "allgather";
    std::string csv_path;    // append a summary row when non-empty
    std::string label;       // free-form experiment tag stored in the CSV
    std::string history_path;  // residual history of the last run (CSV)
    double memory_fraction = 0.6;  // refuse problems above this share of RAM
    bool force = false;      // skip the memory safety check
    bool quiet = false;
    bool info_only = false;  // print environment and exit
    bool help = false;
};

// Parses argv; throws std::invalid_argument with a helpful message on error.
BenchmarkConfig parse_arguments(int argc, char** argv);
std::string usage_text(const char* program);

// Runs the configured benchmark. Returns the process exit code.
// In MPI builds this must be called after MPI_Init_thread by every rank.
int run_benchmark(const BenchmarkConfig& config);

}  // namespace hpcg
