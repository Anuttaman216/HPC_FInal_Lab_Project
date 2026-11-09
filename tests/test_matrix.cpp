// test_matrix.cpp - CSR construction, symmetry, positive definiteness, SpMV,
// right-hand side and row distribution tests.
#include <cmath>
#include <random>

#include "csr_matrix.hpp"
#include "matrix_generator.hpp"
#include "test_utils.hpp"

using namespace hpcg;
using testing_util::run_test;

namespace {

// Dense Cholesky factorisation; returns false as soon as a pivot is <= 0,
// i.e. iff the symmetric matrix is NOT positive definite.
bool dense_cholesky_succeeds(std::vector<double> a, std::size_t n) {
    for (std::size_t j = 0; j < n; ++j) {
        double d = a[j * n + j];
        for (std::size_t k = 0; k < j; ++k) d -= a[j * n + k] * a[j * n + k];
        if (!(d > 0.0)) return false;
        const double ljj = std::sqrt(d);
        a[j * n + j] = ljj;
        for (std::size_t i = j + 1; i < n; ++i) {
            double s = a[i * n + j];
            for (std::size_t k = 0; k < j; ++k) s -= a[i * n + k] * a[j * n + k];
            a[i * n + j] = s / ljj;
        }
    }
    return true;
}

// Smallest eigenvalue of the 7-point operator: the matrix is the Kronecker
// sum of three 1D [-1 2 -1] matrices, whose eigenvalues are
// 2 - 2 cos(k pi / (m + 1)), k = 1..m.
double analytic_lambda_min(const GridDims& g) {
    const double pi = std::acos(-1.0);
    auto l1 = [&](int m) { return 2.0 - 2.0 * std::cos(pi / (m + 1)); };
    return l1(g.nx) + l1(g.ny) + l1(g.nz);
}

}  // namespace

int main() {
    run_test("nnz formula matches generated matrix", [] {
        for (GridDims g : {GridDims{1, 1, 1}, GridDims{2, 1, 1}, GridDims{3, 4, 5},
                           GridDims{10, 10, 10}, GridDims{7, 1, 9}}) {
            const CSRMatrix A = generate_poisson_3d(g);
            CHECK(A.nnz() == poisson_nnz(g));
            CHECK(A.global_rows == g.rows());
        }
        CHECK(poisson_nnz({1, 1, 1}) == 1);
        CHECK(poisson_nnz({2, 1, 1}) == 4);
        CHECK(poisson_nnz({10, 10, 10}) == 6400);  // 1000 + 2*3*900
    });

    run_test("CSR structure is valid (monotone, sorted, in range)", [] {
        const GridDims g{6, 5, 4};
        const CSRMatrix A = generate_poisson_3d(g);
        A.validate(A.global_cols);  // throws on violation
    });

    run_test("CSR validate() detects corrupted structure", [] {
        CSRMatrix A = generate_poisson_3d({3, 3, 3});
        std::swap(A.col_idx[0], A.col_idx[1]);  // break ordering
        bool threw = false;
        try { A.validate(A.global_cols); } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);
    });

    run_test("7-point stencil coefficients (6 on diagonal, -1 neighbours)", [] {
        const GridDims g{5, 5, 5};
        const CSRMatrix A = generate_poisson_3d(g);
        for (index_t i = 0; i < A.local_rows; ++i) {
            int diag = 0;
            for (offset_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; ++k) {
                if (A.col_idx[k] == i) { CHECK(A.values[k] == 6.0); ++diag; }
                else CHECK(A.values[k] == -1.0);
            }
            CHECK(diag == 1);
        }
        // interior point (2,2,2) has all 7 entries, corner (0,0,0) has 4
        const index_t centre = 2 + 5 * (2 + 5 * 2);
        CHECK(A.row_ptr[centre + 1] - A.row_ptr[centre] == 7);
        CHECK(A.row_ptr[1] - A.row_ptr[0] == 4);
    });

    run_test("matrix is symmetric (CSR transpose lookup)", [] {
        CHECK(is_symmetric(generate_poisson_3d({6, 5, 4})));
        CHECK(is_symmetric(generate_poisson_3d({1, 7, 3})));
    });

    run_test("matrix is symmetric (dense comparison)", [] {
        const CSRMatrix A = generate_poisson_3d({3, 4, 2});
        const std::size_t n = static_cast<std::size_t>(A.global_rows);
        const std::vector<double> d = A.to_dense();
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) CHECK(d[i * n + j] == d[j * n + i]);
    });

    run_test("positive definite: dense Cholesky succeeds", [] {
        for (GridDims g : {GridDims{4, 3, 5}, GridDims{2, 2, 2}, GridDims{6, 1, 1}}) {
            const CSRMatrix A = generate_poisson_3d(g);
            CHECK(dense_cholesky_succeeds(A.to_dense(), static_cast<std::size_t>(A.global_rows)));
        }
    });

    run_test("positive definite: smallest eigenvalue equals analytic value > 0", [] {
        // A - s I is positive definite iff s < lambda_min. Cholesky on shifted
        // matrices brackets lambda_min tightly around the analytic value.
        const GridDims g{4, 3, 5};
        const CSRMatrix A = generate_poisson_3d(g);
        const std::size_t n = static_cast<std::size_t>(A.global_rows);
        const double lmin = analytic_lambda_min(g);
        CHECK(lmin > 0.0);
        auto shifted = [&](double s) {
            std::vector<double> d = A.to_dense();
            for (std::size_t i = 0; i < n; ++i) d[i * n + i] -= s;
            return d;
        };
        CHECK(dense_cholesky_succeeds(shifted(lmin - 1e-6), n));
        CHECK(!dense_cholesky_succeeds(shifted(lmin + 1e-6), n));
    });

    run_test("x^T A x > 0 for random vectors", [] {
        const CSRMatrix A = generate_poisson_3d({8, 7, 6});
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> U(-1.0, 1.0);
        std::vector<double> x(A.local_rows), y(A.local_rows);
        for (int trial = 0; trial < 20; ++trial) {
            for (auto& v : x) v = U(rng);
            spmv(A, x.data(), y.data());
            double q = 0.0;
            for (index_t i = 0; i < A.local_rows; ++i) q += x[i] * y[i];
            CHECK(q > 0.0);
        }
    });

    run_test("SpMV matches dense matrix-vector product", [] {
        const CSRMatrix A = generate_poisson_3d({4, 5, 3});
        const std::size_t n = static_cast<std::size_t>(A.global_rows);
        const std::vector<double> d = A.to_dense();
        std::vector<double> x(n), y(n), y_omp(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = std::sin(0.37 * static_cast<double>(i)) + 0.1;
        spmv(A, x.data(), y.data());
        spmv_openmp(A, x.data(), y_omp.data());
        for (std::size_t i = 0; i < n; ++i) {
            double ref = 0.0;
            for (std::size_t j = 0; j < n; ++j) ref += d[i * n + j] * x[j];
            CHECK_NEAR(y[i], ref, 1e-13);
            CHECK_NEAR(y_omp[i], ref, 1e-13);
        }
    });

    run_test("row-block generation equals slices of the full matrix", [] {
        const GridDims g{5, 4, 6};
        const CSRMatrix full = generate_poisson_3d(g);
        for (int parts : {1, 3, 7}) {
            for (int p = 0; p < parts; ++p) {
                const auto r = block_range(g.rows(), parts, p);
                const CSRMatrix blk = generate_poisson_3d_rows(g, r.first, r.second);
                CHECK(blk.first_row == r.first);
                CHECK(blk.local_rows == r.second - r.first);
                for (index_t i = 0; i < blk.local_rows; ++i) {
                    const global_t gi = r.first + i;
                    const offset_t len = blk.row_ptr[i + 1] - blk.row_ptr[i];
                    CHECK(len == full.row_ptr[gi + 1] - full.row_ptr[gi]);
                    for (offset_t k = 0; k < len; ++k) {
                        CHECK(blk.col_idx[blk.row_ptr[i] + k] == full.col_idx[full.row_ptr[gi] + k]);
                        CHECK(blk.values[blk.row_ptr[i] + k] == full.values[full.row_ptr[gi] + k]);
                    }
                }
            }
        }
    });

    run_test("right-hand side b = A*1 (deterministic)", [] {
        const CSRMatrix A = generate_poisson_3d({5, 5, 5});
        const std::vector<double> b = generate_rhs(A);
        std::vector<double> ones(A.local_rows, 1.0), ref(A.local_rows);
        spmv(A, ones.data(), ref.data());
        for (index_t i = 0; i < A.local_rows; ++i) CHECK_NEAR(b[i], ref[i], 0.0);
        const index_t centre = 2 + 5 * (2 + 5 * 2);
        CHECK(b[centre] == 0.0);  // interior rows sum to zero
        CHECK(b[0] == 3.0);       // corner: 6 - 3 neighbours
        CHECK(generate_rhs(A) == b);
    });

    run_test("block_range: uneven distribution covers all rows", [] {
        for (global_t n : {0, 1, 10, 11, 1000, 1001}) {
            for (int parts : {1, 2, 3, 4, 7, 16}) {
                global_t expected_begin = 0, min_sz = n + 1, max_sz = -1;
                for (int p = 0; p < parts; ++p) {
                    const auto r = block_range(n, parts, p);
                    CHECK(r.first == expected_begin);
                    CHECK(r.second >= r.first);
                    min_sz = std::min(min_sz, r.second - r.first);
                    max_sz = std::max(max_sz, r.second - r.first);
                    expected_begin = r.second;
                }
                CHECK(expected_begin == n);
                CHECK(max_sz - min_sz <= 1);
            }
        }
        CHECK((block_range(10, 3, 0) == std::pair<global_t, global_t>(0, 4)));
        CHECK((block_range(10, 3, 1) == std::pair<global_t, global_t>(4, 7)));
        CHECK((block_range(10, 3, 2) == std::pair<global_t, global_t>(7, 10)));
    });

    run_test("matrix metadata (sparsity, memory estimate)", [] {
        const GridDims g{20, 20, 20};
        const MatrixInfo info = describe_poisson(g);
        CHECK(info.rows == 8000);
        CHECK(info.nnz == poisson_nnz(g));
        CHECK(info.sparsity_percent > 99.9 && info.sparsity_percent < 100.0);
        CHECK_NEAR(info.sparsity_percent + info.density_percent, 100.0, 1e-9);
        const CSRMatrix A = generate_poisson_3d(g);
        CHECK_NEAR(info.matrix_bytes, static_cast<double>(A.storage_bytes()), 0.5);
    });

    run_test("invalid grid dimensions are rejected", [] {
        bool threw = false;
        try { generate_poisson_3d({0, 4, 4}); } catch (const std::invalid_argument&) { threw = true; }
        CHECK(threw);
    });

    return testing_util::finish("test_matrix");
}
