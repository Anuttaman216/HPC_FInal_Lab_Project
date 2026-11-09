// metrics.cpp - statistics, derived metrics and environment detection.
#include "metrics.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>
#include <sstream>
#include <thread>

#ifdef HPCG_HAVE_OPENMP
#include <omp.h>
#endif
#ifdef HPCG_HAVE_MPI
#include <mpi.h>
#endif
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

#ifndef HPCG_BUILD_TYPE
#define HPCG_BUILD_TYPE "unknown"
#endif
#ifndef HPCG_BUILD_FLAGS
#define HPCG_BUILD_FLAGS "unknown"
#endif

namespace hpcg {

SampleStats compute_stats(std::vector<double> samples) {
    SampleStats s;
    s.count = static_cast<int>(samples.size());
    if (samples.empty()) return s;
    std::sort(samples.begin(), samples.end());
    const std::size_t n = samples.size();
    s.min = samples.front();
    s.max = samples.back();
    s.median = (n % 2 == 1) ? samples[n / 2]
                            : 0.5 * (samples[n / 2 - 1] + samples[n / 2]);
    s.mean = std::accumulate(samples.begin(), samples.end(), 0.0) / static_cast<double>(n);
    if (n > 1) {
        double acc = 0.0;
        for (double v : samples) acc += (v - s.mean) * (v - s.mean);
        s.stddev = std::sqrt(acc / static_cast<double>(n - 1));
    }
    return s;
}

double speedup(double t_baseline, double t_parallel) {
    return t_parallel > 0.0 ? t_baseline / t_parallel : 0.0;
}

double efficiency_percent(double speedup_value, int processing_units) {
    return processing_units > 0 ? 100.0 * speedup_value / processing_units : 0.0;
}

double gflops(double flops, double seconds) {
    return seconds > 0.0 ? flops / (seconds * 1e9) : 0.0;
}

namespace {

std::string openmp_version_string() {
#ifdef _OPENMP
    switch (_OPENMP) {
        case 200505: return "2.5 (200505)";
        case 200805: return "3.0 (200805)";
        case 201107: return "3.1 (201107)";
        case 201307: return "4.0 (201307)";
        case 201511: return "4.5 (201511)";
        case 201811: return "5.0 (201811)";
        case 202011: return "5.1 (202011)";
        case 202111: return "5.2 (202111)";
        default: return "_OPENMP=" + std::to_string(_OPENMP);
    }
#else
    return "disabled";
#endif
}

std::string compiler_string() {
#if defined(__clang__)
    return std::string("clang ") + __clang_version__;
#elif defined(__GNUC__)
    return std::string("g++ ") + __VERSION__;
#elif defined(_MSC_VER)
    return "msvc " + std::to_string(_MSC_VER);
#else
    return "unknown";
#endif
}

// Reads a "Key:  value kB" line from /proc/meminfo (Linux only).
double meminfo_bytes(const std::string& key) {
    std::ifstream in("/proc/meminfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.compare(0, key.size() + 1, key + ":") == 0) {
            std::istringstream ss(line.substr(key.size() + 1));
            double kb = 0.0;
            ss >> kb;
            return kb * 1024.0;
        }
    }
    return -1.0;
}

}  // namespace

EnvironmentInfo detect_environment() {
    EnvironmentInfo env;
#if defined(__unix__) || defined(__APPLE__)
    char host[256] = {0};
    if (gethostname(host, sizeof(host) - 1) == 0) env.hostname = host;
#endif
    if (env.hostname.empty()) env.hostname = "unknown";
    env.compiler = compiler_string();
    env.build_type = HPCG_BUILD_TYPE;
    env.build_flags = HPCG_BUILD_FLAGS;
    env.openmp_version = openmp_version_string();
#ifdef HPCG_HAVE_MPI
    char lib[MPI_MAX_LIBRARY_VERSION_STRING] = {0};
    int len = 0;
    MPI_Get_library_version(lib, &len);
    std::string v(lib, static_cast<std::size_t>(len));
    const auto cut = v.find_first_of(",\n");
    env.mpi_version = cut == std::string::npos ? v : v.substr(0, cut);
#else
    env.mpi_version = "disabled";
#endif
    env.logical_cpus = static_cast<int>(std::thread::hardware_concurrency());
    env.available_memory_bytes = meminfo_bytes("MemAvailable");
    env.total_memory_bytes = meminfo_bytes("MemTotal");
    return env;
}

}  // namespace hpcg
