// test_openmp.cpp - the OpenMP solver really runs in parallel and produces a
// solution numerically equivalent to the sequential baseline.
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <thread>

#include "cg_solver.hpp"
#include "matrix_generator.hpp"
#include "test_utils.hpp"

using namespace hpcg;
using testing_util::max_abs_diff;
using testing_util::run_test;

int main() {
    const int hw = std::max(1u, std::thread::hardware_concurrency());
    std::vector<int> thread_counts{1, 2, 4};
    if (hw > 4) thread_counts.push_back(hw);

    run_test("OpenMP support compiled in", [] { CHECK(openmp_available()); });

    run_test("requested thread team is actually created", [&] {
        for (int t : thread_counts) {
            omp_set_num_threads(t);
            int team = 0;
#pragma omp parallel
            {
#pragma omp single
                team = omp_get_num_threads();
            }
            CHECK(team == t);
        }
    });

    const CSRMatrix A = generate_poisson_3d({24, 20, 22});
    const std::vector<double> b = generate_rhs(A);
    CGOptions o; o.tolerance = 1e-10; o.max_iterations = 2000;
    std::vector<double> x_seq(A.local_rows, 0.0);
    const CGResult ref = cg_solve_sequential(A, b, x_seq, o);

    for (int t : thread_counts) {
        run_test("OpenMP(" + std::to_string(t) + " threads) matches sequential", [&] {
            omp_set_num_threads(t);
            std::vector<double> x(A.local_rows, 0.0);
            const CGResult r = cg_solve_openmp(A, b, x, o);
            CHECK(r.converged());
            // Reductions sum in a different order, so allow +-1 iteration and
            // compare the solutions with a tolerance, never exact equality.
            CHECK(std::abs(r.iterations - ref.iterations) <= 1);
            CHECK(max_abs_diff(x, x_seq) < 1e-8);
            CHECK(true_relative_residual(A, b, x) < 1e-9);
            CHECK_NEAR(r.rhs_norm, ref.rhs_norm, 1e-10 * ref.rhs_norm);
        });
    }

    run_test("OpenMP solver is reproducible run-to-run (to rounding)", [&] {
        // The OpenMP specification does not fix the order in which the
        // per-thread partial sums of a reduction are combined (libgomp
        // combines them in arrival order), so repeated runs may differ in the
        // last bits. They must agree to rounding and take the same iterations.
        omp_set_num_threads(std::min(4, hw));
        std::vector<double> x1(A.local_rows, 0.0), x2(A.local_rows, 0.0);
        const CGResult r1 = cg_solve_openmp(A, b, x1, o);
        const CGResult r2 = cg_solve_openmp(A, b, x2, o);
        CHECK(std::abs(r1.iterations - r2.iterations) <= 1);
        CHECK(max_abs_diff(x1, x2) < 1e-10);
    });

    run_test("OpenMP handles zero RHS, breakdown and NaN", [&] {
        omp_set_num_threads(2);
        std::vector<double> zero(A.local_rows, 0.0), x(A.local_rows, 5.0);
        CHECK(cg_solve_openmp(A, zero, x, o).status == CGStatus::ZeroRhs);
        for (double v : x) CHECK(v == 0.0);

        CSRMatrix neg = generate_poisson_3d({4, 4, 4});
        for (auto& v : neg.values) v = -v;  // negative definite
        std::vector<double> bn(neg.local_rows, 1.0), xn(neg.local_rows, 0.0);
        CHECK(cg_solve_openmp(neg, bn, xn, o).status == CGStatus::Breakdown);

        std::vector<double> bnan = b;
        bnan[17] = std::nan("");
        std::fill(x.begin(), x.end(), 0.0);
        CHECK(cg_solve_openmp(A, bnan, x, o).status == CGStatus::NonFinite);
    });

    run_test("OpenMP fixed-iteration mode matches sequential flop count", [&] {
        omp_set_num_threads(2);
        CGOptions f; f.tolerance = 0.0; f.max_iterations = 30;
        std::vector<double> x(A.local_rows, 0.0), xs(A.local_rows, 0.0);
        const CGResult r = cg_solve_openmp(A, b, x, f);
        const CGResult s = cg_solve_sequential(A, b, xs, f);
        CHECK(r.iterations == 30 && s.iterations == 30);
        CHECK(r.flops == s.flops);
        CHECK(max_abs_diff(x, xs) < 1e-10);
    });

    return testing_util::finish("test_openmp");
}
