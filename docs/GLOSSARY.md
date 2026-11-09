# Glossary – definitions used in this project

Terms appear in the order you meet them when reading the code. Each entry
says what the term means and where it shows up in the project.

## Benchmarks and context

| Term | Definition | Where in this project |
|---|---|---|
| **HPC** (High-Performance Computing) | Using many processors together to solve problems faster (strong scaling) or to solve bigger problems (weak scaling). | Whole project |
| **HPL** (High-Performance LINPACK) | Benchmark that solves a *dense* linear system by LU factorisation. It ranks the TOP500 list and is compute-bound. | Compared against in report §5 |
| **HPCG** (High Performance Conjugate Gradient) | Benchmark by Dongarra, Heroux and Luszczek that solves a *sparse* system with preconditioned CG. It is memory- and network-bound and complements HPL. | This project is *inspired by* it, not an official version |
| **TOP500** | Twice-yearly ranking of supercomputers by HPL performance. | Context only; we make no ranking claims |
| **Benchmark** | A fixed, repeatable workload used to compare systems or implementations. | `build/hpcg_benchmark` |

## Linear algebra

| Term | Definition | Where |
|---|---|---|
| **Linear system** `Ax = b` | Find vector `x` given matrix `A` and right-hand side `b`. | Solved by every mode |
| **Sparse matrix** | A matrix in which almost all entries are zero (here > 99.9 %). Only the non-zeros are stored. | `CSRMatrix` |
| **nnz** | Number of non-zero entries. | Printed in every `RESULT` line |
| **Symmetric** | `A = Aᵀ`, i.e. `A(i,j) = A(j,i)`. | Tested in `test_matrix` |
| **Positive definite (SPD)** | `xᵀAx > 0` for every non-zero `x`. Equivalently, all eigenvalues are > 0. CG is guaranteed to work only for SPD matrices. | Tested by Cholesky + eigenvalue bracket |
| **Poisson equation** | `−∆u = f`, the model PDE for diffusion, electrostatics and pressure. Its discretisation gives the test matrix. | `matrix_generator.cpp` |
| **7-point stencil** | Discretisation where each grid point couples to itself (coefficient 6) and its 6 face neighbours (coefficient −1). | `generate_poisson_3d_rows` |
| **Dirichlet boundary** | The solution is fixed (here zero) on the boundary, so couplings to outside points are dropped. | Boundary rows have < 7 entries |
| **Diagonally dominant** | `|A(i,i)| ≥ Σ_{j≠i} |A(i,j)|` in every row. Together with symmetry, a positive diagonal and connectivity, this proves SPD. | `methodology.md` §1 |
| **Condition number κ(A)** | `λ_max / λ_min`. Larger κ means slower CG convergence. For this matrix κ grows like (grid size)². | Explains iteration growth |
| **Residual** `r = b − Ax` | How far the current `x` is from satisfying the system. | Stopping test |
| **Relative residual** | `‖r‖₂ / ‖b‖₂`, a scale-free accuracy measure. | `--tol` (default 1e-8) |
| **True vs. recursive residual** | CG updates `r` cheaply by recursion. The *true* residual `b − Ax` is recomputed once at the end to verify it. | `true_relres` in output |

## Algorithm and storage

| Term | Definition | Where |
|---|---|---|
| **Conjugate Gradient (CG)** | Iterative method for SPD systems. Each step makes one SpMV, two dot products and three vector updates. | `src/*/cg_*.cpp` |
| **Krylov subspace** | `span{r₀, Ar₀, A²r₀, …}`. CG finds the best solution in this growing space. | Theory, report §9 |
| **α (alpha), β (beta)** | CG step length (`rᵀr / pᵀAp`) and direction-update factor (`r_newᵀr_new / rᵀr`). | Every solver |
| **Search direction p** | The direction CG moves `x` along in each iteration. | Exchanged between MPI ranks |
| **Breakdown** | `pᵀAp ≤ 0`, which is impossible for an SPD matrix. It signals a bad matrix, and the solver stops with status `breakdown`. | `CGStatus::Breakdown` |
| **CSR** (Compressed Sparse Row) | Storage with three arrays: `values` (non-zeros row by row), `col_idx` (their columns), `row_ptr` (where each row starts). | `include/csr_matrix.hpp` |
| **SpMV** | Sparse Matrix–Vector multiplication `y = A·x`. It is the most expensive CG kernel. | `spmv()`, fused into solvers |
| **Dot product** | `Σ xᵢyᵢ`. In parallel it needs a reduction. | `(p,Ap)`, `(r,r)` |
| **AXPY** | Vector update `y = y + αx`. | `x += αp`, `r −= αAp` |
| **Kernel fusion** | Merging two loops into one so data is read from memory once instead of twice. | SpMV + dot, updates + dot |

## Parallel programming

| Term | Definition | Where |
|---|---|---|
| **Thread** | An execution stream sharing memory with others in the same process. | OpenMP mode |
| **Process / rank** | An independent program instance with private memory. MPI numbers them 0…P−1 (the *rank*). | MPI mode |
| **OpenMP** | Compiler directives (`#pragma omp …`) for shared-memory threading. | `cg_openmp.cpp` |
| **`parallel for` / static schedule** | Splits loop iterations among threads in equal contiguous chunks, the same ones every time. | All OpenMP loops |
| **Reduction** | Combines per-thread or per-rank partial results (e.g. sums) into one value without races. | `reduction(+:…)`, `MPI_Allreduce` |
| **Barrier** | Synchronisation point that every thread must reach before any continues. Implicit at the end of each `omp for`. | OpenMP loops |
| **Race condition** | Two threads write (or one writes and another reads) the same memory without ordering, giving a wrong result. Avoided by design. | — |
| **First touch** | On NUMA systems a memory page is placed near the thread that first writes it, so vectors are initialised in parallel. | `cg_openmp.cpp` |
| **MPI** | Message Passing Interface, the standard library for distributed-memory communication. | `src/mpi/`, `src/hybrid/` |
| **Collective operation** | An MPI call that every rank must make in the same order (e.g. `MPI_Allreduce`). A mismatch deadlocks. | All decisions use reduced values |
| **`MPI_Allreduce`** | Combines a value from all ranks (e.g. sum) and gives the result to every rank. | Every dot product |
| **`MPI_Allgatherv`** | Every rank contributes its block and receives everyone's blocks, so the whole vector is replicated. | `--comm allgather` |
| **Halo / ghost exchange** | Each rank receives only the neighbouring vector entries its rows reference, via `MPI_Isend`/`MPI_Irecv`. | `--comm halo` |
| **Row-block distribution** | Each rank owns a contiguous range of rows. With uneven `n/P`, the first `n mod P` ranks get one extra row. | `block_range()` |
| **Extended window** | A rank's local copy of `p` covering every column its rows use (its own block plus halos). | `VectorExchange` |
| **Hybrid MPI + OpenMP** | MPI between processes, OpenMP threads inside each process. | `cg_hybrid.cpp` |
| **`MPI_THREAD_FUNNELED`** | Thread-support level: threads may exist, but only the main thread calls MPI. | Requested and verified |
| **Oversubscription** | Running more processes or threads than there are CPUs. Avoided by default. | `run_benchmarks.sh` skips such runs |
| **Process binding** | Pinning a process to specific cores. We use `--bind-to none` because Open MPI's default would pin all of a hybrid rank's threads to one core. | `MPIRUN` in scripts |

## Performance metrics and models

| Term | Definition | Where |
|---|---|---|
| **Processing units p** | MPI processes × OpenMP threads. | `processing_units` CSV column |
| **Execution time T(p)** | Wall-clock solve time (median of repetitions), excluding matrix generation. | `time_median_s` |
| **Speedup S(p)** | `T_sequential / T(p)`. | `speedup` in `summary.csv` |
| **Parallel efficiency E(p)** | `S(p)/p × 100 %`. 100 % is ideal. | `efficiency_pct` |
| **Strong scaling** | Fixed problem size, more units. Goal: shorter time. | `label=strong` |
| **Weak scaling** | Problem size grows with units (constant work per unit). Goal: constant time. Efficiency is `T(1)/T(p)`. | `label=weak` |
| **FLOP / GFLOPS** | Floating-point operation(s). GFLOPS is billions of FLOPs per second. Here it is *estimated* from an operation-count model. | `gflops` |
| **Flop model** | Per iteration `2·nnz + 10n`; setup `2·nnz + 5n`. | `include/common.hpp` |
| **Memory bandwidth** | Bytes per second that the memory system can deliver. | `results/bandwidth.csv` |
| **STREAM triad** | The bandwidth test `a = b + s·c`, which moves 24 bytes per element. | `src/tools/stream_triad.cpp` |
| **Arithmetic intensity** | Flops per byte moved. CG ≈ 0.13 flop/byte, which is very low. | `arith_intensity` |
| **Roofline model** | Attainable performance = min(peak flops, intensity × bandwidth). | `11_roofline.png` |
| **Memory-bound** | Performance is limited by bandwidth, not arithmetic. Adding cores stops helping once the memory bus is saturated. | Main finding |
| **Amdahl's law** | Strong-scaling limit `S ≤ 1/(s + (1−s)/p)` for a serial fraction `s`. | Report §9 |
| **Gustafson's law** | Weak-scaling view: scaled speedup `s + p(1−s)`. | Report §19.6 |
| **Communication fraction** | (vector-exchange time + Allreduce time) / solve time. | `comm_fraction` |
| **Cache-resident** | The working set fits in the CPU caches, which are faster than DRAM, so small problems look unrealistically fast. | Size sweep, report §19.5 |
| **Median of repetitions** | Robust central value that ignores outlier runs. | All reported times |
| **Warm-up run** | An untimed first solve that loads caches and pages before measuring. | `--warmup` |
