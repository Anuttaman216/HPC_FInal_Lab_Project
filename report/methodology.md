# Methodology

This document fixes the experimental rules used by every measurement in the
project, so that results are reproducible and comparisons are fair.

> This is an **HPCG-inspired educational implementation**. It is not the
> official HPCG benchmark, is not HPCG-compliant, and its numbers must not be
> reported as HPCG or TOP500 results.

## 1. Problem definition

| Item | Choice |
|---|---|
| Equation | 3D Poisson, −∆u = f, homogeneous Dirichlet boundary |
| Discretisation | 7-point finite-difference stencil on an `nx × ny × nz` grid |
| Matrix entries | `A(i,i) = 6`, `A(i,j) = −1` for the ≤ 6 face neighbours inside the grid |
| Boundary | couplings to points outside the grid are dropped (Dirichlet value 0) |
| Row ordering | lexicographic, `row = i + nx·(j + ny·k)` |
| Storage | CSR: `row_ptr` (int64), `col_idx` (int32), `values` (double) |
| Right-hand side | `b = A·1` (so the exact solution is `x* = 1`, as in HPCG) |
| Initial guess | `x0 = 0` |
| Non-zeros | `nnz = n + 2[(nx−1)ny·nz + nx(ny−1)nz + nx·ny(nz−1)]` (verified in tests) |

**Why the matrix is SPD.** It is symmetric because the neighbour relation is
symmetric. Every row has diagonal 6 and at most six off-diagonal entries of
magnitude 1, so it is weakly diagonally dominant, and strictly dominant on
every boundary row. The grid graph is connected, so the matrix is
irreducible. An irreducibly diagonally dominant matrix is non-singular
(Taussky), and a symmetric matrix with positive diagonal that is
irreducibly diagonally dominant is positive definite. The test suite also
checks this numerically: a dense Cholesky factorisation succeeds, and
shifted Cholesky factorisations bracket the smallest eigenvalue around its
analytic value `Σ_d (2 − 2cos(π/(n_d+1)))`.

## 2. Algorithm

Unpreconditioned Conjugate Gradient (Hestenes & Stiefel, 1952):

```
r0 = b − A x0 ;  p0 = r0
repeat
    α     = (r,r) / (p,Ap)
    x     = x + α p
    r_new = r − α Ap
    β     = (r_new,r_new) / (r,r)
    p     = r_new + β p
until ||r|| / ||b|| ≤ tol  or  k = max_iters
```

Every implementation fuses the same kernels:

1. `Ap = A·p` fused with `(p, Ap)`,
2. `x += α p`, `r −= α Ap` fused with `(r, r)`,
3. `p = r + β p`.

Breakdown handling: `(p,Ap) ≤ 0` gives `breakdown` (matrix not SPD or exact
breakdown). A NaN or Inf in any reduction gives `non_finite`. `||b|| = 0`
gives `zero_rhs` with `x = 0`. The iteration limit gives `max_iterations`.
Fixed-iteration runs (`--tol 0`) use the last status deliberately, for
weak scaling.

## 3. Implementations

| Mode | Parallelism | Communication |
|---|---|---|
| sequential | none (baseline) | none |
| openmp | `omp parallel for schedule(static)`; `reduction(+:…)` for dot products | shared memory, implicit barrier after each loop |
| mpi | contiguous row blocks per rank (`block_range`, sizes differ by ≤ 1) | `MPI_Allgatherv` (replicated `p`) **or** halo `MPI_Isend/Irecv`; `MPI_Allreduce` for every dot product |
| hybrid | MPI between ranks, OpenMP inside each rank | as MPI. `MPI_THREAD_FUNNELED`, and MPI is called only by the master thread outside parallel regions |

Each MPI rank generates only its own rows, so no global matrix is ever
built or scattered. The columns referenced by the local rows form one
contiguous global range `[lo, hi)`. `p` is stored in an *extended window*
over that range, and the stored column indices are shifted to point into
the window.

* **Allgather (replicated vector):** the window is the whole vector. Each
  rank receives `n − n_local` doubles per iteration. This is O(n) per rank,
  so it does not scale.
* **Halo:** each rank receives only the overlap of its window with each
  neighbour's block. For the 7-point stencil this is at most one grid plane
  (`nx·ny` doubles) per side. Peers and segments are computed collectively
  once, at setup.

All ranks take the same branch, because α, β and the convergence test use
`MPI_Allreduce` results. So the sequence of collective calls is identical
on every rank.

## 4. Timing rules

* The timed region starts just before the initial residual `r0 = b − A x0`
  and ends after the last iteration. Matrix generation, allocation of the
  work vectors, process start-up and verification are **excluded** for every
  mode.
* Sequential and OpenMP runs use `std::chrono::steady_clock`. MPI and hybrid
  runs use `MPI_Wtime`, start after an `MPI_Barrier`, and report the
  **maximum** time over the ranks.
* Communication time is measured around each exchange and each
  `MPI_Allreduce` and averaged over ranks. Allreduce time includes waiting
  for the slowest rank (load imbalance), so it is an upper bound on pure
  network or copy cost.
* Each configuration runs 1 untimed warm-up solve, then R timed solves
  (R = 5 for strong scaling, 3 elsewhere). The **median** is reported. Min,
  max, mean and standard deviation are kept in the CSV.

## 5. Floating-point operation model

Per CG iteration, for global size `n` and `nnz` non-zeros:

| Kernel | Flops |
|---|---|
| SpMV `A·p` | 2·nnz |
| `(p,Ap)` | 2n |
| `x += αp` | 2n |
| `r −= αAp` | 2n |
| `(r,r)` | 2n |
| `p = r + βp` | 2n |
| **Total per iteration** | **2·nnz + 10n** |
| Setup (`r0`, `(r,r)`, `(b,b)`) | 2·nnz + 5n |

`GFLOPS = (setup + iterations · per-iteration flops) / (median time · 10⁹)`.
Scalar operations are ignored. Distributed runs are credited with the same
*global* flop count as the sequential run, so communication and redundant
work never count as useful flops.

## 6. Memory-traffic model (roofline)

Assuming every array is streamed from DRAM once per kernel and the gathered
`p` hits in cache:

| Traffic | Bytes |
|---|---|
| CSR values + column indices | 12 per non-zero |
| `row_ptr`, gathered `p`, write `Ap` | 24 per row |
| update kernel (read x,p,r,Ap; write x,r) | 48 per row |
| `p` update (read r,p; write p) | 24 per row |
| **Total** | **12·nnz + 96n** |

With `nnz ≈ 6.9n` the arithmetic intensity is ≈ 0.13 flop/byte. This is far
below the machine balance of any modern CPU, so CG is **memory-bandwidth
bound**. The roofline graph uses the STREAM-style triad bandwidth measured
by `build/stream_triad` (24 bytes per element, STREAM convention).

## 7. Derived metrics

* Speedup `S(p) = T_seq / T(p)`. `T_seq` is the sequential run with the same
  experiment label, grid, tolerance and iteration cap.
* Efficiency `E(p) = S(p) / p × 100 %`, where `p = MPI processes × OpenMP
  threads` (total processing units).
* Weak-scaling efficiency `E_w(p) = T(1) / T(p) × 100 %`, using
  `nx = ny = B`, `nz = B·p` and a fixed number of iterations (`--tol 0`),
  as HPCG does, so that iteration counts do not depend on the problem size.
* Communication fraction = (exchange + Allreduce time) / solve time.

## 8. Fair-comparison rules

1. Strong-scaling configurations share one grid (128³ by default), one
   tolerance (1e−8) and one iteration cap (5000). The iteration counts are
   reported and verified to be equal.
2. The size sweep uses the same tolerance for every size. Its curves compare
   implementations at equal size, never across sizes. Larger grids need more
   iterations (condition number ~ h⁻²).
3. Weak scaling uses fixed iterations, so per-unit work is constant.
4. Configurations that need more processing units than the machine has are
   **skipped with an explanation** in `results/run_status.csv`. No
   oversubscription is used unless `ALLOW_OVERSUBSCRIBE=1`.
5. Every number in the report comes from `results/benchmark.csv`. Nothing is
   hard-coded or extrapolated.

## 9. Correctness validation

`ctest` runs four unit suites and three end-to-end runs. The MPI suite runs
with 1, 2, 3 and 4 ranks; 3 ranks gives an uneven distribution, and the
4×4×2 grid makes halos span several ranks. The checks cover:

* CSR construction and invariants, the nnz formula, and stencil values
* Symmetry (CSR transpose lookup and dense comparison)
* Positive definiteness (Cholesky and the analytic λ_min bracket)
* SpMV against a dense product
* Sequential CG on a hand-made 3×3 system with a known solution
* Convergence to `x* = 1`, and agreement between the true and recursive
  residuals
* Breakdown, NaN, Inf, zero-RHS and iteration-limit handling
* OpenMP against sequential solutions for 1, 2, 4 and all threads
  (tolerance comparison)
* MPI and hybrid with both exchange schemes against the sequential solution
* Exchange correctness, uneven distribution, and identical iteration counts
  on all ranks

Every benchmark run also reports the true residual `||b − Ax||/||b||` and
`max|x − 1|`.

## 10. Execution environment

Experiments ran in an Ubuntu 24.04 Docker container (g++ 13.3, Open MPI
4.1.6, OpenMP 4.5) on Docker Desktop's WSL2 virtual machine on a Windows 11
laptop. The exact hardware and software description is in
`results/system_info.txt`. Linux `perf` hardware counters are not available
inside this VM; this limitation is recorded in the system info.
