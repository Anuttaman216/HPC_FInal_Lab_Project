# P08 – High-Performance Conjugate-Gradient Benchmark: A New Metric for Ranking High-Performance Computing Systems

**Performance Evaluation of Shared-Memory and Distributed-Memory Parallelization Techniques for Conjugate Gradient-Based HPC Workloads**

*BE Computer Science – High Performance Computing project report*

> **Disclaimer.** The software described here is an *HPCG-inspired educational
> implementation*. It is not the official HPCG benchmark, it does not follow
> the official HPCG run rules, and none of its numbers are HPCG or TOP500
> results.

---

## 1. Abstract

Dense-linear-algebra benchmarks such as HPL rank supercomputers mainly by
floating-point throughput. Many scientific codes, however, are limited by
memory bandwidth and communication. HPCG was proposed to capture that
behaviour with a sparse, preconditioned Conjugate Gradient (CG) workload.

This project implements an HPCG-inspired CG benchmark in C++17 with four
interchangeable back-ends:

* sequential,
* OpenMP,
* MPI, with two communication schemes (replicated-vector `MPI_Allgatherv`
  and point-to-point halo exchange),
* hybrid MPI + OpenMP.

All four solve a 3D 7-point Poisson system stored in CSR format. The
implementations are verified by 10 automated test targets and benchmarked on
a 16-thread laptop CPU (Intel Core 7 240H) inside a Linux container, using
strong-scaling, problem-size and weak-scaling experiments. A
STREAM-style bandwidth probe and a roofline model explain the measurements:

* CG has an arithmetic intensity of ≈ 0.13 flop/byte and is memory-bandwidth
  bound.
* The best speedup is only 1.40× (OpenMP, 8 threads; 5.95 s → 4.24 s on
  a 2.1-million-row problem), matching the measured 1.32× growth of
  memory bandwidth from 1 to 8 threads.
* Replicated-vector MPI becomes slower than sequential at 8 processes
  (communication share 51 %), while halo exchange keeps communication at
  or below 12 % and is 1.69× faster.
* Cache-resident problem sizes inflate parallel speedups to as much as
  3.9×.

The results reproduce, at laptop scale, the motivation for HPCG: sparse
iterative solvers are limited by memory and communication, not by peak
floating-point performance.

## 2. Introduction

Ranking HPC systems requires a benchmark that reflects how the machines are
actually used. HPL measures dense LU factorisation, which is almost entirely
compute-bound [7]. Modern simulation codes instead rely on sparse matrices,
iterative solvers, stencils and global reductions. Their speed depends on
memory bandwidth, latency and network performance [1, 12]. Dongarra and
Heroux therefore proposed the High Performance Conjugate Gradient (HPCG)
benchmark as a complementary ranking metric [1, 2, 3].

This project rebuilds the computational core of HPCG at an educational
scale: a CG solver for a sparse SPD stencil matrix. It uses the core to
study experimentally how shared-memory and distributed-memory
parallelisation behave on a memory-bound workload.

## 3. Background of High-Performance Computing

HPC systems combine many processing elements so that large problems can be
solved faster (strong scaling) or larger problems can be solved in the same
time (weak scaling). Today's systems are clusters of shared-memory nodes,
each with several multicore CPUs (and often accelerators), connected by a
high-speed network. Two programming models dominate:

* **Shared memory (OpenMP)**: threads in one address space; work is divided
  among loop iterations; synchronisation uses barriers and reductions [14].
* **Distributed memory (MPI)**: processes with private memory that exchange
  explicit messages and perform collective operations [13].

They are frequently combined (**hybrid MPI + OpenMP**) [17]. Performance
analysis uses speedup, efficiency, Amdahl's and Gustafson's laws [9, 10]
and the roofline model [8].

## 4. HPCG Benchmark Overview

According to its authors [1–3] and the reference code [15], HPCG:

1. builds a sparse SPD matrix from a **27-point stencil** on a 3D grid, with
   one local subdomain per MPI process, so the global problem grows with the
   number of processes;
2. solves `Ax = b` with **CG preconditioned by a multigrid V-cycle with a
   symmetric Gauss-Seidel smoother**;
3. exercises the kernels SpMV, SymGS, DDOT (with `MPI_Allreduce`), WAXPBY
   and the multigrid restriction and prolongation, with halo exchanges
   between neighbouring subdomains;
4. validates symmetry and convergence and measures the optimised code over a
   long timed phase, reporting GFLOP/s.

Because these kernels have low arithmetic intensity, HPCG results on leading
systems are a small fraction of their HPL results [2, 3]. HPCG therefore
ranks machines by memory-system and network quality as well as by peak
compute.

## 5. Comparison Between HPL and HPCG

| Aspect | HPL | HPCG | This project |
|---|---|---|---|
| Problem | dense random system | sparse 27-pt stencil SPD | sparse **7-pt** stencil SPD |
| Algorithm | LU with partial pivoting | multigrid-preconditioned CG | **unpreconditioned** CG |
| Dominant kernel | DGEMM | SpMV, SymGS | SpMV, dot, AXPY |
| Arithmetic intensity | high (O(n) flop/word) | low (< 1 flop/byte) | ≈ 0.13 flop/byte (model, §19) |
| Bottleneck | floating-point units | memory bandwidth, latency, network | memory bandwidth, reductions |
| Communication | panel broadcasts | halo exchange + Allreduce | Allgatherv **or** halo + Allreduce |
| Fraction of peak | high | low | not measured (peak unknown) – see roofline |
| Scaling style | weak (n grows) | weak (local subdomain fixed) | strong, size sweep **and** weak |
| Official ranking | TOP500 | HPCG list | **none – educational** |

## 6. Problem Statement

Design, implement and evaluate an HPCG-inspired benchmark that:

* solves a sparse SPD system with CG using sequential, OpenMP, MPI and
  hybrid MPI + OpenMP back-ends;
* measures time, GFLOPS, speedup, efficiency and communication overhead
  under identical problem sizes and stopping criteria;
* explains the observed performance with established models (Amdahl,
  Gustafson, roofline);
* states clearly where and why it differs from the official HPCG.

## 7. Research Objectives

1. Study HPCG and its motivation (§4, literature review).
2. Understand the limitations of HPL for scientific workloads (§5).
3. Investigate sparse matrix computation in CSR format (§9, §12).
4. Analyse OpenMP performance on a memory-bound kernel (§19.2–19.3).
5. Analyse MPI communication overhead for two communication schemes (§19.3–19.4).
6. Investigate scalability and parallel efficiency, strong and weak (§19.3, §19.6).
7. Compare the experimental observations with the HPCG methodology (§20).

## 8. Literature Review

The full review is in [`literature_review.md`](literature_review.md). In
short:

* HPL's dense, compute-bound workload is not representative of sparse
  scientific codes [1, 7].
* HPCG adds sparse SpMV, multigrid and global reductions [1–3, 15] and is
  bandwidth- and latency-bound [19].
* CG convergence depends on κ(A), and for 3D Poisson κ = O(h⁻²) [4–6].
* CSR SpMV is memory-bound, and its multicore performance follows memory
  bandwidth [12].
* The roofline [8] and STREAM [11] quantify that bound.
* OpenMP reductions and barriers, and MPI halo exchanges and Allreduce, are
  the parallel overheads of CG [13, 14, 18].
* Hybrid MPI + OpenMP trades fewer messages against idle threads and
  placement sensitivity [17].

## 9. Mathematical Foundation

**Model problem.** −∆u = f on the unit cube with u = 0 on the boundary. The
standard 7-point finite-difference stencil on an `nx × ny × nz` grid gives
(after scaling by h²)

$$ (Au)_{ijk} = 6u_{ijk} - u_{i\pm1,jk} - u_{i,j\pm1,k} - u_{ij,k\pm1}. $$

Neighbours outside the grid are boundary values (zero) and are dropped.
`A` is symmetric, irreducibly diagonally dominant with a positive diagonal,
and therefore positive definite. `A` is the Kronecker sum of three 1D
`[−1 2 −1]` matrices, so its eigenvalues are

$$ \lambda_{abc} = \sum_{d\in\{x,y,z\}} \left(2 - 2\cos\frac{k_d\pi}{n_d+1}\right),\quad \kappa(A) = \frac{\lambda_{max}}{\lambda_{min}} = O(h^{-2}). $$

The test suite verifies λ_min numerically.

**CSR.** For `n` rows and `nnz` non-zeros, CSR uses `row_ptr[n+1]`,
`col_idx[nnz]` and `values[nnz]`. SpMV is `y_i = Σ_{k=row_ptr[i]}^{row_ptr[i+1]-1} values[k]·x[col_idx[k]]`.

**Conjugate Gradient.** Starting from `x₀`, with `r₀ = p₀ = b − Ax₀`:

$$ \alpha_k = \frac{r_k^T r_k}{p_k^T A p_k},\quad x_{k+1} = x_k + \alpha_k p_k,\quad r_{k+1} = r_k - \alpha_k A p_k, $$
$$ \beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k},\quad p_{k+1} = r_{k+1} + \beta_k p_k. $$

The iteration stops when `||r_k||₂ / ||b||₂ ≤ tol`. CG minimises the
A-norm of the error over the Krylov subspace, and the error decreases at
least as `2((√κ − 1)/(√κ + 1))^k` [5]. The number of iterations needed
therefore grows like √κ = O(1/h), roughly linearly in the grid dimension.
§19.5 confirms this experimentally.

**Performance metrics.**

$$ S(p) = \frac{T_{seq}}{T(p)},\qquad E(p) = \frac{S(p)}{p}\times100\%,\qquad \text{GFLOPS} = \frac{F}{T\cdot10^9}, $$

with `F = (2nnz + 5n) + k(2nnz + 10n)` for `k` iterations (methodology §5),
and `p` = MPI processes × OpenMP threads. Amdahl's law
`S ≤ 1/(s + (1−s)/p)` and the roofline bound `P ≤ min(P_peak, I·B)` frame
the interpretation.

## 10. System Architecture

```
                   +------------------------------------------+
  CLI / scripts -> |  src/main.cpp  ->  benchmark.cpp          |  argument parsing, memory guard,
                   |  (MPI_Init_thread FUNNELED in MPI builds) |  warm-up + timed repetitions,
                   +---------------------+--------------------+  verification, CSV output
                                         |
          +------------------------------+---------------------------------+
          |                              |                                 |
  matrix/ (CSR + generator)     shared-memory solvers            distributed solvers
  csr_matrix.cpp                sequential/cg_sequential.cpp     mpi/distributed_problem.cpp
  matrix_generator.cpp          openmp/cg_openmp.cpp               (row blocks, VectorExchange:
  - 7-pt Poisson, b = A*1                                           Allgatherv | halo Isend/Irecv)
  - block_range (uneven split)                                    mpi/cg_mpi.cpp
                                                                   hybrid/cg_hybrid.cpp
          |
  tests/ (CTest: matrix, sequential, openmp, mpi x {1,2,3,4} ranks, e2e)
          |
  scripts/run_benchmarks.sh -> results/benchmark.csv, bandwidth.csv, system_info.txt
          |
  python/analyze.py -> summary.csv/.md   python/visualize.py -> graphs/*.png   python/dashboard.py -> dashboard.html
```

The layers are kept separate. Numerical kernels do not know about timing or
I/O. The benchmark layer owns timing, repetitions and reporting. Analysis
is pure Python and reads only the CSV files.

## 11. Technology Stack

| Layer | Technology |
|---|---|
| Language | C++17 (g++ 13.3, `-O3`) |
| Shared memory | OpenMP 4.5 (libgomp) |
| Distributed memory | Open MPI 4.1.6 (MPI-3.1) |
| Build / test | CMake ≥ 3.16, CTest |
| Analysis | Python 3, NumPy, Pandas, Matplotlib |
| Environment | Ubuntu 24.04 Docker image (`docker/Dockerfile`) on Docker Desktop / WSL2, Windows 11 host |
| Profiling | Linux `perf` not usable in this VM (documented). Communication time is measured with `MPI_Wtime` instrumentation |

## 12. Implementation Methodology

* **One algorithm, four back-ends.** Every back-end uses the same kernel
  fusion, the same convergence test and the same flop model, so differences
  in time come from parallelisation alone (methodology §2–3).
* **No allocation inside iterations.** Work vectors are allocated before
  the timed region. MPI request arrays are allocated once in the exchange
  plan.
* **Safety.** All inputs are validated. A memory guard refuses problems
  needing more than 60 % of the available RAM (decided by rank 0 and
  broadcast). Breakdown, NaN and Inf are detected and reported, and the
  process exits with a non-zero status.
* **Honest labelling.** If OpenMP or MPI are not found, the corresponding
  modes are not compiled and the program refuses to run them; it never
  falls back to sequential code under a parallel name.
* **Development order.** Generator → sequential CG + tests → OpenMP + tests →
  MPI + tests (1–4 ranks) → hybrid → benchmark automation → analysis →
  documentation. The code was compiled and tested after each stage.

## 13. Sequential Implementation

`src/sequential/cg_sequential.cpp` is the baseline. It makes three passes
over memory per iteration (fused SpMV + dot, fused updates + dot, and the
`p` update), with plain contiguous loops that the compiler can vectorise.
Timing uses `std::chrono::steady_clock`. The result structure records the
iterations, the initial and final residual, the relative residual, the
status and the modelled flops. The true residual is recomputed outside the
timed region.

## 14. OpenMP Implementation

`src/openmp/cg_openmp.cpp` turns each fused kernel into an
`omp parallel for schedule(static)` loop. The two dot products use
`reduction(+:…)`, so no shared variable is written concurrently. The
implicit barrier at the end of each loop is the only synchronisation; it is
needed because SpMV reads all of `p`. α, β and the convergence test are
computed serially between loops, so all threads use identical scalars. Work
vectors are first-touched in parallel with the same static schedule, and
the matrix generator fills rows in parallel.

*Determinism.* The OpenMP specification does not define the order in which
the per-thread partial sums of a reduction are combined. With libgomp,
repeated runs with the same thread count gave results that differed in the
last bits. Our first unit test assumed bit-exact equality and failed. The
test now checks agreement to 1e−10 and iteration counts within ±1. This is a
genuine observation from development.

## 15. MPI Implementation

`src/mpi/distributed_problem.cpp` and `src/mpi/cg_mpi.cpp`:

1. `block_range` assigns contiguous row blocks. The first `n mod P` ranks
   get one extra row, so uneven sizes are handled.
2. Each rank generates only its rows (no scatter of a global matrix) and the
   matching part of `b`.
3. The global column range used by the local rows defines the extended
   window of `p`. Columns are re-indexed into the window.
4. `VectorExchange` refreshes the window before each SpMV, with either
   * **Allgatherv**: replicated vector, in place, `n − n_local` doubles
     received per rank per iteration; or
   * **Halo**: `MPI_Irecv` from and `MPI_Isend` to exactly the ranks whose
     blocks overlap the window, which is at most one grid plane (`nx·ny`
     values) per side for this stencil.
5. Local SpMV fused with the local `(p,Ap)`, then `MPI_Allreduce`. The
   updates are fused with the local `(r,r)`, then `MPI_Allreduce`.
6. Convergence decisions use the reduced values, so every rank leaves the
   loop at the same iteration and the collective sequence is identical.
7. The solve time is the maximum over ranks (`MPI_Wtime` after a barrier).
   Exchange and Allreduce times are accumulated separately and averaged
   over ranks.

*Limitation of Allgatherv.* Memory and traffic per rank are O(n), so this
scheme cannot scale beyond a node. It is kept as the simple baseline the
assignment asked for, and it serves as the comparison point for the halo
scheme.

## 16. Hybrid Implementation

`src/hybrid/cg_hybrid.cpp` reuses the MPI data distribution and exchange
plan and parallelises every local loop with OpenMP. MPI is initialised with
`MPI_Init_thread(MPI_THREAD_FUNNELED)`. The solver checks
`MPI_Query_thread ≥ FUNNELED` and `MPI_Is_thread_main`, and makes MPI
calls only between parallel regions. This is the "master-only" style of
Rabenseifner et al. [17]. A check in the container confirmed that Open MPI
4.1.6 returns `provided = MPI_THREAD_FUNNELED` for this request, and the
unit tests assert that level. Ranks are launched
with `--bind-to none`, because Open MPI's default core binding would pin
all of a rank's threads to one core.

## 17. Experimental Setup

The exact values recorded by the benchmark script are in
`results/system_info.txt`.

| Item | Value |
|---|---|
| CPU | Intel Core 7 240H (hybrid architecture: 6 performance + 4 efficiency cores per Intel specification; 16 logical CPUs), seen by the VM as 16 vCPUs |
| Memory visible to the container | ≈ 11.5 GiB (Docker Desktop WSL2 VM) |
| Host OS | Windows 11 Home (build 26200) |
| Guest OS | Ubuntu 24.04 container, WSL2 kernel |
| Compiler | g++ 13.3.0, `-O3 -DNDEBUG`, no `-march=native` (portable build) |
| OpenMP | 4.5 (libgomp) |
| MPI | Open MPI 4.1.6, shared-memory transport, `--bind-to none` |
| Matrix (strong scaling) | 128³ grid: 2,097,152 rows, 14,581,760 non-zeros, ≈ 263 MiB |
| Stopping criterion | `||r||/||b|| ≤ 1e−8`, max 5000 iterations |
| Repetitions | 1 warm-up + 5 timed (strong), 1 + 3 (size, weak); median reported |

**Threats to validity.** The machine is a laptop. Clock frequency changes
with temperature and power state (turbo, battery or AC). The heterogeneous
P- and E-cores make a static loop schedule imbalanced once E-cores or
hyper-threads are used. The hypervisor adds scheduling noise. Background
Windows processes were not controlled. Results are therefore indicative of
trends, not precise hardware characterisation.

## 18. Benchmarking Methodology

The full rules are in [`methodology.md`](methodology.md). In short: the
timed region excludes generation and allocation; the median of repetitions
is reported; speedup is relative to the sequential run with the same grid
and tolerance; processing units are processes × threads; configurations
that need more units than available are skipped and logged; and weak scaling
uses a fixed iteration count. The experiments run automatically through
`scripts/run_all.sh`, which performs the dependency check, clean build,
CTest, benchmarks, analysis, graphs and dashboard.

## 19. Results and Performance Analysis

All values below come from `results/benchmark.csv`: 50 configurations,
50 successful and 0 skipped, as logged in `results/run_status.csv`. The
derived tables are in `results/summary.md`, the graphs in
`results/graphs/`, and an interactive view in `results/dashboard.html`.
Times are medians of 5 timed repetitions after 1 warm-up (strong scaling),
or of 3 repetitions (size and weak sweeps).

### 19.1 Correctness of every measured run

Every strong-scaling configuration (sequential, OpenMP 1–16 threads, MPI
1–8 processes with both schemes, six hybrid layouts) needed **exactly 296
iterations**. Every one finished with true relative residual
`9.786 × 10⁻⁹` (< tol = 10⁻⁸) and `max|x − 1| ≈ 7.69 × 10⁻⁸`. The parallel
versions therefore solve the same problem to the same accuracy as the
baseline, and the time comparisons below are like-for-like. All 10 CTest
targets passed (`results/test_results.txt`).

### 19.2 Memory bandwidth (STREAM-style triad)

| Threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Best triad bandwidth (GB/s) | 21.9 | 24.1 | 27.4 | 28.9 | 27.8 |

**Observation.** One core already uses about 76 % of the best bandwidth
measured for the whole chip. The total bandwidth grows only **1.32×** from
1 to 8 threads. Any purely memory-bound kernel on this machine is
therefore limited to a speedup of roughly 1.3–1.4×, whatever the number of
cores.

### 19.3 Strong scaling (128³ grid, 2,097,152 rows, 14,581,760 non-zeros)

| Configuration | p | Median time (s) | Speedup | Efficiency | GFLOPS | Comm. share |
|---|---|---|---|---|---|---|
| Sequential | 1 | 5.953 | 1.00 | 100 % | 2.50 | – |
| OpenMP 2t | 2 | 4.540 | 1.31 | 65.6 % | 3.28 | – |
| OpenMP 4t | 4 | 4.455 | 1.34 | 33.4 % | 3.34 | – |
| **OpenMP 8t** | 8 | **4.240** | **1.40** | 17.5 % | **3.51** | – |
| OpenMP 16t | 16 | 6.634 | 0.90 | 5.6 % | 2.24 | – |
| MPI-allgather 2p | 2 | 5.148 | 1.16 | 57.8 % | 2.89 | 13.5 % |
| MPI-allgather 4p | 4 | 6.089 | 0.98 | 24.4 % | 2.44 | 34.1 % |
| MPI-allgather 8p | 8 | 7.446 | 0.80 | 10.0 % | 2.00 | 50.6 % |
| MPI-halo 2p | 2 | 4.730 | 1.26 | 62.9 % | 3.15 | 3.0 % |
| MPI-halo 4p | 4 | 4.882 | 1.22 | 30.5 % | 3.05 | 8.5 % |
| MPI-halo 8p | 8 | 4.405 | 1.35 | 16.9 % | 3.38 | 11.5 % |
| Hybrid-halo 2×2 | 4 | 4.728 | 1.26 | 31.5 % | 3.15 | 3.4 % |
| Hybrid-halo 2×4 | 8 | 4.540 | 1.31 | 16.4 % | 3.28 | 1.9 % |
| **Hybrid-halo 4×2** | 8 | **4.269** | **1.39** | 17.4 % | **3.49** | 5.1 % |
| Hybrid-halo 2×8 | 16 | 6.124 | 0.97 | 6.1 % | 2.43 | 17.6 % |
| Hybrid-halo 4×4 | 16 | 10.229 | 0.58 | 3.6 % | 1.45 | 45.9 % |
| Hybrid-halo 8×2 | 16 | 9.717 | 0.61 | 3.8 % | 1.53 | 62.3 % |

Figures: `01_time_vs_threads.png`, `02_time_vs_mpi_procs.png`,
`03_speedup.png`, `04_efficiency.png`, `05_gflops_comparison.png`,
`06_implementation_comparison.png`.

**Observations.**

1. **The speedup ceiling equals the bandwidth ceiling.** The best speedup,
   1.40× (OpenMP, 8 threads), matches the 1.32× bandwidth gain of §19.2.
   Converting GFLOPS to bandwidth with the traffic model (methodology §6),
   the sequential solver streams about **18.8 GB/s**, or 86 % of the
   1-thread triad. The best parallel run streams about **26.3 GB/s**, or
   91 % of the best triad. The solver is therefore already close to the
   memory roof (`11_roofline.png`). Adding cores cannot help, because the
   cores are waiting for DRAM, not computing. This is the behaviour HPCG
   was designed to expose. The efficiency drop is a bandwidth limit, not
   an Amdahl serial fraction.
2. **16 processing units are slower than 8** for every model (OpenMP,
   hybrid 2×8, 4×4, 8×2). With 16 units every hyper-thread of the
   16-vCPU VM is busy. The extra hyper-threads add no bandwidth, and they
   compete with spinning threads in OpenMP barriers and Open MPI
   busy-polling. They also compete with the hypervisor and Windows host,
   and with the hybrid P- and E-cores, which make a static schedule
   imbalanced. Run-to-run variance confirms this: hybrid 4×4 ranged from
   9.45 s to 12.06 s, against 4.19–4.34 s for 4×2.
3. **Replicated vectors do not scale.** With `MPI_Allgatherv`, every rank
   receives `n − n/P` doubles per iteration: 1.84 M doubles (14.7 MB) per
   rank at P = 8. The communication share rises from 13.5 % to 50.6 %, and
   the run with 8 processes is *slower than sequential* (0.80×).
4. **Halo exchange fixes this.** A rank receives at most two grid planes
   (2 × 16,384 doubles ≈ 0.26 MB), about 56× less than Allgatherv at P = 8.
   The communication share stays at or below 11.5 %, and MPI-halo 8p
   (1.35×) almost matches OpenMP. At P = 8 the halo scheme is 1.69× faster
   than Allgatherv on the same problem.
5. **MPI with 1 process costs about 9 %** over sequential (6.48 s against
   5.95 s), even though no messages are sent. The cost comes from the
   window copy of `x` in setup, the distributed bookkeeping and the
   `MPI_Allreduce` calls (two per iteration, 592 in total). The measured
   reduction time with one rank is only 0.002 s, so MPI calls themselves
   are not the cause. We did not isolate the remaining difference further
   (candidates are memory placement and run-to-run noise); treat it as the
   cost of the MPI code path.
6. **Hybrid layouts.** Hybrid 4×2 (1.39×) equals the best OpenMP result
   and has a lower communication share than pure MPI with 8 ranks (5.1 %
   against 11.5 %). This agrees with the literature: hybrid MPI + OpenMP
   reduces the number of messages and reductions per node [17]. It is not
   faster, because the bound is memory bandwidth, which is shared by all
   threads and ranks alike.

### 19.4 Communication overhead

Figure `07_comm_overhead.png` splits each MPI run into computation, vector
exchange and Allreduce. Measured communication time, averaged over ranks,
for the whole solve:

| P | Allgatherv scheme (s) | Halo scheme (s) |
|---|---|---|
| 2 | 0.695 | 0.140 |
| 4 | 2.079 | 0.414 |
| 8 | 3.771 | 0.508 |

Allreduce time also includes waiting for the slowest rank, so the split
slightly overstates pure communication.

### 19.5 Performance vs. matrix size

| Grid | Rows | Iterations | Seq. (s) | OpenMP 8t (s) | MPI-halo 4p (s) | OpenMP speedup | MPI speedup |
|---|---|---|---|---|---|---|---|
| 16³ | 4,096 | 41 | 0.0022 | 0.0006 | 0.0017 | 3.89 | 1.29 |
| 32³ | 32,768 | 81 | 0.0150 | 0.0056 | 0.0092 | 2.67 | 1.62 |
| 48³ | 110,592 | 120 | 0.1295 | 0.0378 | 0.0341 | 3.43 | 3.80 |
| 64³ | 262,144 | 158 | 0.3772 | 0.1291 | 0.1937 | 2.92 | 1.95 |
| 96³ | 884,736 | 225 | 2.866 | 1.269 | 1.399 | 2.26 | 2.05 |
| 128³ | 2,097,152 | 296 | 8.158 | 4.703 | 4.866 | 1.73 | 1.68 |
| 160³ | 4,096,000 | 368 | 17.352 | 10.247 | 11.071 | 1.69 | 1.57 |

Figures: `08_perf_vs_size.png` and `10_convergence.png`.

**Observations.**

1. **The cache effect.** Up to about 64³ (≈ 33 MB of matrix + vectors at
   64³, much less below it), most of the working set fits in the caches.
   There the parallel versions reach 2.7–3.9× speedup and an "effective
   bandwidth" of 50–85 GB/s, far above DRAM triad, because the data comes
   from cache. Beyond 96³ the data comes from DRAM, and the speedup
   collapses to the bandwidth ratio (≈ 1.6–1.7×). Small benchmark
   problems therefore overstate parallel performance badly. This is why
   HPCG requires problem sizes well beyond the cache capacity.
2. **Iteration count grows linearly with the grid dimension.** For
   N = 16 to 160, k/N stays between 2.3 and 2.6. This matches the
   CG theory: κ = O(N²) and k = O(√κ) = O(N) (§9).
3. **Laptop variability.** The sequential 128³ run took 5.95 s in the
   strong-scaling sweep but 8.16 s in the size sweep, which ran about
   15 minutes later in the same campaign. Speedups are always computed
   against the sequential run of the *same* sweep, so the comparisons
   stay valid. The difference itself (thermal or power-state throttling
   on a laptop) is a reminder that absolute times on this machine carry
   roughly ±30 % uncertainty across sessions.

### 19.6 Weak scaling (64 × 64 × 64·p rows, fixed 100 iterations)

| Units p | OpenMP time (s) | OpenMP E_w | MPI-halo time (s) | MPI-halo E_w |
|---|---|---|---|---|
| 1 | 0.327 | 100 % | 0.267 | 100 % |
| 2 | 0.300 | 109 % | 0.355 | 75 % |
| 4 | 0.689 | 48 % | 0.614 | 43 % |
| 8 | 1.337 | 24 % | 1.323 | 20 % |
| 16 | 4.190 | 8 % | – | – |

Figure: `09_weak_scaling.png`. The value above 100 % at p = 2 comes from
the slower single-thread baseline in that sweep; it is measurement noise,
not super-linear scaling.

**Observation.** Weak scaling fails on one node for the same reason
strong scaling does. The total work grows with p, but the memory bandwidth
stays fixed, so the time grows almost linearly with p. Gustafson's law
assumes resources that grow with the machine. On a cluster each node
brings its own memory bus, and weak scaling is then expected to be much
better. This is the regime HPCG targets, and we could not test it here.

### 19.7 Roofline

Model intensity: `I = (2·nnz + 10n)/(12·nnz + 96n) ≈ 0.133 flop/byte`.
With the measured roofs, attainable performance is
`0.133 × 21.9 ≈ 2.9 GFLOPS` for one thread and
`0.133 × 28.9 ≈ 3.8 GFLOPS` for the chip. The measured 2.50 GFLOPS
(sequential) and 3.51 GFLOPS (best parallel) sit just under these roofs
(`11_roofline.png`). The peak floating-point rate of the CPU, tens of
GFLOPS per core with AVX2 FMA, is irrelevant at this intensity. This is
the quantitative core of the HPCG argument.

## 20. Discussion

* **What the experiments show (our observations).** On a modern laptop CPU
  the CG solver is memory-bandwidth bound. The best implementation runs at
  1.4× the sequential speed and about 90 % of the measured STREAM-triad
  bandwidth. Parallel efficiency is therefore low in every model, and a
  low efficiency here is not a sign of a bad implementation. The memory
  bus is the bottleneck. How data is communicated matters more than how
  many processes are used: halo exchange beats replicated vectors by 1.7×
  at 8 processes, and Allgatherv becomes slower than sequential.
  Problem sizes that fit in cache inflate speedups by more than 2×.
* **Relation to HPCG (established facts and our comparison).** HPCG was
  created because HPL's compute-bound kernel ranks machines by peak flops
  [1–3]. We observed the mechanism directly: the same CPU that sustains
  only ~3.5 GFLOPS on CG has a far higher DGEMM peak, and the CG figure is
  set by the triad bandwidth. Published HPCG results likewise reach only a
  few percent of peak [2, 3], and Marjanović et al. [19] model HPCG
  performance from bandwidth and network characteristics. Our single-node
  results reproduce the *memory* side of this argument. The *network* side
  (latency of global Allreduce across thousands of nodes) is only hinted at
  by our Allreduce measurements, because all ranks share one machine.
* **Methodological differences from HPCG.**
  * No preconditioner: HPCG's SymGS smoother has sequential dependencies
    and is harder to parallelise than our kernels.
  * 7-point instead of 27-point stencil: lower nnz per row, slightly
    different intensity.
  * 1D instead of 3D decomposition.
  * Tolerance-based strong scaling: HPCG uses fixed-iteration weak
    scaling.
  * No minimum run time or optimisation validation.

  For these reasons our GFLOPS are *not* comparable with HPCG lists.
* **Practical lessons.**
  1. Measure the bandwidth roof before judging parallel efficiency.
  2. Never benchmark sparse solvers with cache-resident problems.
  3. Avoid replicated-vector communication.
  4. Check MPI process binding for hybrid runs (default core binding would
     serialise threads).
  5. Do not expect bit-identical results from OpenMP reductions.

## 21. Limitations

1. **Not HPCG.** The project uses no multigrid / symmetric Gauss–Seidel
   preconditioner, a 7-point rather than 27-point stencil, a 1D row-block
   decomposition rather than a 3D process grid, and none of HPCG's run rules
   (minimum run time, optimisation validation). Its GFLOPS are therefore not
   comparable with published HPCG numbers.
2. **Single node.** Every "distributed" run uses Open MPI's shared-memory
   transport inside one laptop. Real network latency and bandwidth were not
   measured. The MPI results describe intra-node message passing.
3. **Replicated-vector scheme.** The Allgatherv variant has O(n) memory and
   traffic per rank by design. It is included as the simple baseline and is
   not a scalable method.
4. **1D decomposition.** Halo volume per rank is one full `nx·ny` plane per
   side regardless of the process count, so the surface-to-volume ratio
   worsens as `P` grows. A 3D decomposition (as in HPCG) would scale better.
5. **Virtualised, heterogeneous laptop.** Results come from a Docker/WSL2 VM
   on a hybrid P/E-core CPU with turbo and thermal effects, with no CPU
   pinning control from the host. Run-to-run variance is visible in the
   min/max columns of `benchmark.csv`.
6. **No hardware counters.** Linux `perf` is not usable in the VM, so cache
   miss rates and achieved DRAM bandwidth are *modelled* (traffic model plus
   triad bandwidth), not measured.
7. **Traffic model.** The 12·nnz + 96n bytes/iteration model assumes perfect
   reuse of the gathered `p` vector and no write-allocate traffic. Real
   traffic is somewhat higher, so the "effective bandwidth" column is a lower
   bound on the bandwidth actually used.
8. **Allreduce time includes waiting.** Measured reduction time contains load
   imbalance, so the computation/communication split is approximate.

## 22. Future Scope

* Implement the HPCG preconditioner (symmetric Gauss–Seidel + multigrid
  V-cycle) and the 27-point stencil to approach the official workload.
* Use a 3D domain decomposition with 6- or 26-neighbour halo exchange.
* Overlap communication and computation: post the halo `Irecv/Isend`,
  compute the interior rows, then the boundary rows. Use a single
  non-blocking `MPI_Iallreduce`, or pipelined / communication-avoiding CG
  (Chronopoulos–Gear, Ghysels–Vanroose).
* Use a persistent OpenMP parallel region per solve to remove fork/join
  overhead, and thread pinning (`OMP_PROC_BIND`, `OMP_PLACES`) that respects
  P- and E-cores.
* Store the matrix-free stencil or use SELL-C-σ / blocked formats to cut
  the 12 B/nnz matrix traffic, which dominates the memory traffic.
* Run on a real multi-node cluster to measure network effects, and collect
  hardware counters with `perf` / LIKWID on bare metal.
* Use GPU back-ends (CUDA/HIP) to compare HBM bandwidth against DDR.
* Add an interactive Streamlit dashboard. The current dashboard is a static
  HTML page, chosen to avoid an extra dependency.

## 23. Conclusion

We built a complete, tested HPCG-inspired CG benchmark with sequential,
OpenMP, MPI (two communication schemes) and hybrid back-ends, plus an
automated measurement and analysis pipeline. All implementations converge
to the exact solution of the 3D Poisson problem with identical iteration
counts, and all numbers in this report come from recorded runs.

The experiments reproduce, at laptop scale, the central argument behind
HPCG. A CG workload is limited by memory bandwidth and communication, not
by arithmetic. Adding cores helps only until the memory bus is saturated,
and how vector data is communicated (replicated versus halo) matters more
than how many processes are used. This is why a sparse, CG-based metric
ranks systems differently from HPL, and why it complements HPL as a
predictor of real scientific-application performance.

## 24. References

1. J. Dongarra and M. A. Heroux, *Toward a New Metric for Ranking High Performance Computing Systems*, Sandia National Laboratories Technical Report SAND2013-4744, June 2013.
2. J. Dongarra, M. A. Heroux and P. Luszczek, "High-performance conjugate-gradient benchmark: A new metric for ranking high-performance computing systems," *International Journal of High Performance Computing Applications*, 30(1):3–10, 2016.
3. J. Dongarra, M. A. Heroux and P. Luszczek, "A new metric for ranking high-performance computing systems," *National Science Review*, 3(1):30–35, 2016.
4. M. R. Hestenes and E. Stiefel, "Methods of Conjugate Gradients for Solving Linear Systems," *Journal of Research of the National Bureau of Standards*, 49(6):409–436, 1952.
5. Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM, 2003.
6. J. R. Shewchuk, *An Introduction to the Conjugate Gradient Method Without the Agonizing Pain*, Technical Report CMU-CS-94-125, Carnegie Mellon University, 1994.
7. J. J. Dongarra, P. Luszczek and A. Petitet, "The LINPACK Benchmark: past, present and future," *Concurrency and Computation: Practice and Experience*, 15(9):803–820, 2003.
8. S. Williams, A. Waterman and D. Patterson, "Roofline: An Insightful Visual Performance Model for Multicore Architectures," *Communications of the ACM*, 52(4):65–76, 2009.
9. G. M. Amdahl, "Validity of the single processor approach to achieving large scale computing capabilities," *Proc. AFIPS Spring Joint Computer Conference*, pp. 483–485, 1967.
10. J. L. Gustafson, "Reevaluating Amdahl's Law," *Communications of the ACM*, 31(5):532–533, 1988.
11. J. D. McCalpin, "Memory Bandwidth and Machine Balance in Current High Performance Computers," *IEEE Computer Society Technical Committee on Computer Architecture (TCCA) Newsletter*, December 1995.
12. S. Williams, L. Oliker, R. Vuduc, J. Shalf, K. Yelick and J. Demmel, "Optimization of sparse matrix–vector multiplication on emerging multicore platforms," *Parallel Computing*, 35(3):178–194, 2009 (extended version of the SC'07 paper).
13. Message Passing Interface Forum, *MPI: A Message-Passing Interface Standard, Version 3.1*, 2015 (and Version 4.0, 2021). <https://www.mpi-forum.org/docs/>
14. OpenMP Architecture Review Board, *OpenMP Application Programming Interface, Version 4.5*, November 2015 (and Version 5.2, 2021). <https://www.openmp.org/specifications/>
15. HPCG Benchmark, official website and reference implementation. <https://www.hpcg-benchmark.org>, <https://github.com/hpcg-benchmark/hpcg>
16. TOP500 Supercomputer Sites. <https://www.top500.org>
17. R. Rabenseifner, G. Hager and G. Jost, "Hybrid MPI/OpenMP Parallel Programming on Clusters of Multi-Core SMP Nodes," *Proc. 17th Euromicro International Conference on Parallel, Distributed and Network-based Processing (PDP)*, pp. 427–436, 2009.
18. G. Hager and G. Wellein, *Introduction to High Performance Computing for Scientists and Engineers*, CRC Press, 2010.
19. V. Marjanović, J. Gracia and C. W. Glass, "Performance Modeling of the HPCG Benchmark," in *High Performance Computing Systems. Performance Modeling, Benchmarking, and Simulation (PMBS 2014)*, LNCS 8966, Springer, pp. 172–192, 2015.
20. W. Gropp, E. Lusk and A. Skjellum, *Using MPI: Portable Parallel Programming with the Message-Passing Interface*, 3rd ed., MIT Press, 2014.
21. B. Chapman, G. Jost and R. van der Pas, *Using OpenMP: Portable Shared Memory Parallel Programming*, MIT Press, 2007.

*All references are established publications or official documentation.
Statements attributed to them summarise their published content. All
statements about measured performance are this project's own observations
from `results/benchmark.csv`.*

