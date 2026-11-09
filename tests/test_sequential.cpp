// test_sequential.cpp - correctness and robustness of the sequential CG.
#include <cmath>
#include <limits>

#include "cg_solver.hpp"
#include "matrix_generator.hpp"
#include "test_utils.hpp"

using namespace hpcg;
using testing_util::run_test;

namespace {

// Build a full CSR matrix from a small dense matrix (tests only).
CSRMatrix from_dense(const std::vector<double>& d, index_t n) {
    CSRMatrix A;
    A.global_rows = A.global_cols = n;
    A.local_rows = n;
    A.row_ptr.push_back(0);
    for (index_t i = 0; i < n; ++i) {
        for (index_t j = 0; j < n; ++j) {
            const double v = d[static_cast<std::size_t>(i) * n + j];
            if (v != 0.0) { A.col_idx.push_back(j); A.values.push_back(v); }
        }
        A.row_ptr.push_back(static_cast<offset_t>(A.values.size()));
    }
    return A;
}

}  // namespace

int main() {
    run_test("3x3 hand-made SPD system with known solution", [] {
        // A = [4 1 0; 1 3 1; 0 1 2], x* = (1,2,3) -> b = (6,10,8)
        const CSRMatrix A = from_dense({4, 1, 0, 1, 3, 1, 0, 1, 2}, 3);
        const std::vector<double> b{6, 10, 8};
        std::vector<double> x(3, 0.0);
        CGOptions o; o.tolerance = 1e-13; o.max_iterations = 50;
        const CGResult r = cg_solve_sequential(A, b, x, o);
        CHECK(r.status == CGStatus::Converged);
        CHECK(r.iterations <= 4);  // <= n in exact arithmetic (+1 for rounding)
        CHECK_NEAR(x[0], 1.0, 1e-11);
        CHECK_NEAR(x[1], 2.0, 1e-11);
        CHECK_NEAR(x[2], 3.0, 1e-11);
    });

    run_test("Poisson 10^3 converges to the exact solution x = 1", [] {
        const CSRMatrix A = generate_poisson_3d({10, 10, 10});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> x(A.local_rows, 0.0);
        CGOptions o; o.tolerance = 1e-10; o.max_iterations = 500;
        const CGResult r = cg_solve_sequential(A, b, x, o);
        CHECK(r.converged());
        CHECK(r.relative_residual <= 1e-10);
        double err = 0.0;
        for (double v : x) err = std::max(err, std::fabs(v - 1.0));
        CHECK(err < 1e-8);
        // recursively updated residual agrees with the true residual
        CHECK(true_relative_residual(A, b, x) < 1e-9);
    });

    run_test("finite termination on a tiny Poisson grid (2x2x2)", [] {
        const CSRMatrix A = generate_poisson_3d({2, 2, 2});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> x(A.local_rows, 0.0);
        CGOptions o; o.tolerance = 1e-12;
        const CGResult r = cg_solve_sequential(A, b, x, o);
        CHECK(r.converged());
        CHECK(r.iterations <= 8);
    });

    run_test("residual history: starts at 1, ends below tolerance", [] {
        const CSRMatrix A = generate_poisson_3d({8, 8, 8});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> x(A.local_rows, 0.0);
        CGOptions o; o.tolerance = 1e-9; o.record_history = true;
        const CGResult r = cg_solve_sequential(A, b, x, o);
        CHECK(r.converged());
        CHECK(r.residual_history.size() == static_cast<std::size_t>(r.iterations) + 1);
        CHECK_NEAR(r.residual_history.front(), 1.0, 1e-15);  // x0 = 0 => r0 = b
        CHECK(r.residual_history.back() <= 1e-9);
        CHECK(r.residual_history[r.residual_history.size() - 2] > 1e-9);
    });

    run_test("zero right-hand side returns x = 0 immediately", [] {
        const CSRMatrix A = generate_poisson_3d({4, 4, 4});
        const std::vector<double> b(A.local_rows, 0.0);
        std::vector<double> x(A.local_rows, 3.0);
        const CGResult r = cg_solve_sequential(A, b, x, CGOptions{});
        CHECK(r.status == CGStatus::ZeroRhs);
        CHECK(r.converged());
        for (double v : x) CHECK(v == 0.0);
    });

    run_test("breakdown detected for a non-SPD matrix", [] {
        const CSRMatrix A = from_dense({-1, 0, 0, -2}, 2);
        const std::vector<double> b{1, 1};
        std::vector<double> x(2, 0.0);
        const CGResult r = cg_solve_sequential(A, b, x, CGOptions{});
        CHECK(r.status == CGStatus::Breakdown);
        CHECK(!r.converged());
    });

    run_test("NaN in the input is reported, not propagated silently", [] {
        const CSRMatrix A = generate_poisson_3d({3, 3, 3});
        std::vector<double> b = generate_rhs(A);
        b[5] = std::numeric_limits<double>::quiet_NaN();
        std::vector<double> x(A.local_rows, 0.0);
        const CGResult r = cg_solve_sequential(A, b, x, CGOptions{});
        CHECK(r.status == CGStatus::NonFinite);
    });

    run_test("infinity in the matrix is reported", [] {
        CSRMatrix A = generate_poisson_3d({3, 3, 3});
        A.values[10] = std::numeric_limits<double>::infinity();
        const std::vector<double> b(A.local_rows, 1.0);
        std::vector<double> x(A.local_rows, 0.0);
        const CGResult r = cg_solve_sequential(A, b, x, CGOptions{});
        CHECK(r.status == CGStatus::NonFinite);
    });

    run_test("iteration limit respected (status max_iterations)", [] {
        const CSRMatrix A = generate_poisson_3d({12, 12, 12});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> x(A.local_rows, 0.0);
        CGOptions o; o.tolerance = 1e-14; o.max_iterations = 5;
        const CGResult r = cg_solve_sequential(A, b, x, o);
        CHECK(r.status == CGStatus::MaxIterations);
        CHECK(r.iterations == 5);
        CHECK(r.relative_residual < 1.0);
    });

    run_test("fixed-iteration mode (tol = 0) runs exactly max_iterations", [] {
        const CSRMatrix A = generate_poisson_3d({10, 10, 10});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> x(A.local_rows, 0.0);
        CGOptions o; o.tolerance = 0.0; o.max_iterations = 25;
        const CGResult r = cg_solve_sequential(A, b, x, o);
        CHECK(r.iterations == 25);
        CHECK_NEAR(r.flops, cg_total_flops(A.global_rows, A.nnz(), 25), 0.5);
    });

    run_test("exact initial guess converges in zero iterations", [] {
        const CSRMatrix A = generate_poisson_3d({6, 6, 6});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> x(A.local_rows, 1.0);
        const CGResult r = cg_solve_sequential(A, b, x, CGOptions{});
        CHECK(r.status == CGStatus::Converged);
        CHECK(r.iterations == 0);
    });

    run_test("numerical stability on 30^3: tight tolerance, residual gap small", [] {
        const CSRMatrix A = generate_poisson_3d({30, 30, 30});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> x(A.local_rows, 0.0);
        CGOptions o; o.tolerance = 1e-11; o.max_iterations = 2000;
        const CGResult r = cg_solve_sequential(A, b, x, o);
        CHECK(r.converged());
        const double true_rel = true_relative_residual(A, b, x);
        CHECK(true_rel < 1e-10);  // drift between recursive and true residual is tiny
        double err = 0.0;
        for (double v : x) err = std::max(err, std::fabs(v - 1.0));
        CHECK(err < 1e-7);
    });

    run_test("invalid inputs throw", [] {
        const CSRMatrix A = generate_poisson_3d({3, 3, 3});
        std::vector<double> b(5, 1.0), x(A.local_rows, 0.0);
        bool threw = false;
        try { cg_solve_sequential(A, b, x, CGOptions{}); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    });

    return testing_util::finish("test_sequential");
}
