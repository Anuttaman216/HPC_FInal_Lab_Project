// metrics.hpp - performance metrics and simple statistics.
#pragma once

#include <string>
#include <vector>

namespace hpcg {

struct SampleStats {
    int count = 0;
    double median = 0.0;
    double mean = 0.0;
    double min = 0.0;
    double max = 0.0;
    double stddev = 0.0;  // sample standard deviation (n-1)
};

SampleStats compute_stats(std::vector<double> samples);

// S(p) = T(1) / T(p)
double speedup(double t_baseline, double t_parallel);
// E(p) = S(p) / p * 100   (percent)
double efficiency_percent(double speedup_value, int processing_units);
// GFLOPS = flops / (seconds * 1e9)
double gflops(double flops, double seconds);

// Environment description used in logs and CSV files.
struct EnvironmentInfo {
    std::string hostname;
    std::string compiler;
    std::string build_type;
    std::string build_flags;
    std::string openmp_version;  // "disabled" if not compiled with OpenMP
    std::string mpi_version;     // "disabled" if not compiled with MPI
    int logical_cpus = 0;
    double available_memory_bytes = -1.0;  // -1 when unknown
    double total_memory_bytes = -1.0;
};

EnvironmentInfo detect_environment();

}  // namespace hpcg
