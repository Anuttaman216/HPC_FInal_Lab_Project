// stream_triad.cpp - STREAM-style triad (a = b + s*c) memory bandwidth probe.
//
// Used only to place the CG results on a roofline: CG is memory-bound, so
// its attainable performance is ~ arithmetic intensity x sustained bandwidth.
// This is a simplified re-implementation of the idea of McCalpin's STREAM
// benchmark, not the official STREAM code; byte counting follows the STREAM
// convention (24 bytes per element, write-allocate traffic not counted).
//
// Usage: stream_triad [--mb MB_PER_ARRAY] [--reps R] [--threads t1,t2,...]
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef HPCG_HAVE_OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    std::size_t mb = 128;
    int reps = 10;
    std::vector<int> threads;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--mb" && i + 1 < argc) mb = std::stoul(argv[++i]);
        else if (a == "--reps" && i + 1 < argc) reps = std::stoi(argv[++i]);
        else if (a == "--threads" && i + 1 < argc) {
            std::stringstream ss(argv[++i]);
            std::string t;
            while (std::getline(ss, t, ',')) threads.push_back(std::stoi(t));
        } else {
            std::cerr << "usage: stream_triad [--mb N] [--reps R] [--threads 1,2,4]\n";
            return 1;
        }
    }
#ifdef HPCG_HAVE_OPENMP
    if (threads.empty()) threads.push_back(omp_get_max_threads());
#else
    threads.assign(1, 1);
#endif
    const std::size_t n = mb * 1024 * 1024 / sizeof(double);
    std::vector<double> a(n), b(n), c(n);
    const double s = 3.0;

    std::cout << "threads,array_mb,best_GBps,median_GBps\n";
    for (int t : threads) {
#ifdef HPCG_HAVE_OPENMP
        omp_set_num_threads(t);
#endif
        // first touch with the same static schedule as the kernel
#pragma omp parallel for schedule(static)
        for (std::size_t i = 0; i < n; ++i) { a[i] = 0.0; b[i] = 1.0; c[i] = 2.0; }
        std::vector<double> rates;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
            for (std::size_t i = 0; i < n; ++i) a[i] = b[i] + s * c[i];
            const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            rates.push_back(24.0 * static_cast<double>(n) / sec / 1e9);
        }
        // check the result so the loop cannot be optimised away
        if (a[n / 2] != 7.0) { std::cerr << "triad verification failed\n"; return 2; }
        std::sort(rates.begin(), rates.end());
        std::cout << t << ',' << mb << ',' << rates.back() << ',' << rates[rates.size() / 2] << "\n";
    }
    return 0;
}
