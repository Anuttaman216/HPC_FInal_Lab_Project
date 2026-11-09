# Presentation Outline (12 slides)

All numbers on these slides are measured values from `results/benchmark.csv`
(see `results/summary.md`). Graph files are in `results/graphs/`.

---

## Slide 1 – Project Title and Team

**Slide content**

* P08 – High-Performance Conjugate-Gradient Benchmark: A New Metric for
  Ranking HPC Systems
* Subtitle: Performance evaluation of shared- and distributed-memory CG
* Team: _Name 1, Name 2, Name 3, Name 4_ · Guide: _…_ · Department / College / 2026
* Footnote: HPCG-inspired educational implementation, not official HPCG

**Speaker notes.** Introduce the team. Say at the outset that we rebuilt the
core of the HPCG benchmark for study, and that we do not claim official
HPCG results.

---

## Slide 2 – Introduction to HPC

**Slide content**

* HPC means many cores and nodes working on one problem.
* Two programming models: shared memory (OpenMP) and distributed memory
  (MPI). Real systems combine them (hybrid).
* Goals: solve faster (strong scaling) or solve bigger (weak scaling).
* Limits: Amdahl's law, memory bandwidth, communication.

**Speaker notes.** Keep this short. The key idea for later slides: a
processor is only as fast as the data it can get from memory.

---

## Slide 3 – What is HPCG?

**Slide content**

* High Performance Conjugate Gradient benchmark (Dongarra, Heroux,
  Luszczek, 2013–2016).
* Sparse 27-point 3D stencil, multigrid-preconditioned CG.
* Kernels: SpMV, symmetric Gauss-Seidel, dot products (Allreduce), halo
  exchange.
* Purpose: rank systems by performance that real scientific apps see.

**Speaker notes.** HPCG was published as a complement to HPL. Mention the
official site, hpcg-benchmark.org. It stresses memory and network, not
peak flops.

---

## Slide 4 – HPL vs HPCG

**Slide content**

| | HPL | HPCG |
|---|---|---|
| Kernel | dense LU (DGEMM) | sparse SpMV + SymGS |
| Intensity | high | < 1 flop/byte |
| Bottleneck | FPUs | memory bandwidth, network |
| % of peak | high | a few % |

**Speaker notes.** The same machine looks very different under the two
metrics. Our slide 10 shows why, measured on a laptop.

---

## Slide 5 – Problem Statement and Objectives

**Slide content**

* Build an HPCG-inspired CG benchmark in C++ with 4 back-ends:
  sequential, OpenMP, MPI, hybrid.
* Measure time, GFLOPS, speedup, efficiency and communication overhead
  fairly (same problem, same tolerance).
* Explain the results with roofline, Amdahl and Gustafson.
* Compare our methodology with official HPCG.

**Speaker notes.** Stress fairness. Every configuration solves the same
2.1-million-row system to the same 1e−8 tolerance, and all of them took
296 iterations.

---

## Slide 6 – Mathematical Foundation

**Slide content**

* 3D Poisson, 7-point stencil: 6 on the diagonal, −1 for neighbours, so
  the matrix is SPD (proved, and tested by Cholesky).
* CSR storage: `row_ptr`, `col_idx`, `values`.
* CG: α = rᵀr / pᵀAp; x += αp; r −= αAp; β = r'ᵀr' / rᵀr; p = r + βp.
* Flops per iteration: 2·nnz + 10n. Intensity ≈ 0.13 flop/byte.
* Iterations grow linearly with grid size (measured: 41 → 368 for
  16³ → 160³).

**Speaker notes.** Explain why the intensity is low: each non-zero needs
12 bytes of memory traffic for only 2 flops.

---

## Slide 7 – Proposed Architecture

**Slide content**

* Diagram: CLI benchmark → {matrix generator, sequential, OpenMP, MPI,
  hybrid} → CSV → Python analysis → graphs and dashboard.
* MPI: row blocks with uneven split. Vector exchange by **Allgatherv**
  (replicated) or **halo** (neighbours only).
* Hybrid: `MPI_THREAD_FUNNELED`; MPI only outside OpenMP regions.
* Tests: 10 CTest targets; MPI tested on 1, 2, 3 and 4 ranks.

**Speaker notes.** Point out the separation of layers, and that every rank
generates its own rows, so no global matrix is ever built.

---

## Slide 8 – Implementation and Technologies

**Slide content**

* C++17, g++ 13.3 `-O3`, OpenMP 4.5, Open MPI 4.1.6, CMake/CTest.
* Python: Pandas, NumPy, Matplotlib.
* Docker Ubuntu 24.04 image for reproducibility on a Windows laptop.
* Robustness: breakdown, NaN and Inf detection; memory guard; median of
  repeated runs.

**Speaker notes.** Mention the OpenMP determinism finding: reductions are
not bit-identical between runs, so tests compare with a tolerance.

---

## Slide 9 – Experimental Setup

**Slide content**

* Intel Core 7 240H laptop, 16 logical CPUs, about 11.5 GiB RAM visible
  to the WSL2 VM.
* Strong scaling: 128³ grid (2,097,152 rows, 14.6 M non-zeros), tol 1e−8,
  1 warm-up + 5 runs, median.
* Size sweep 16³–160³; weak scaling 64×64×64p with fixed 100 iterations.
* STREAM-style triad: 21.9 GB/s (1 thread), 28.9 GB/s (best).

**Speaker notes.** Mention the threats to validity: laptop thermals,
hybrid P/E cores, virtualisation.

---

## Slide 10 – Results and Performance Analysis

**Slide content** (graphs `03_speedup.png`, `07_comm_overhead.png`,
`11_roofline.png`)

* Sequential 5.95 s → best 4.24 s (OpenMP 8t): **1.40× speedup**.
* Bandwidth grows only 1.32× from 1 to 8 threads, so the speedup ceiling
  is set by memory.
* MPI Allgatherv at 8 processes: 0.80× (slower than sequential), with 51 %
  of the time in communication.
* MPI halo at 8 processes: 1.35×, with 11.5 % in communication, 1.69×
  faster than Allgatherv.
* 16 units: slower than 8 (hyper-threads, oversubscription, spin-waiting).

**Speaker notes.** Show the roofline: our points sit at about 90 % of the
measured bandwidth roof, so we are close to the hardware limit, not
poorly parallelised.

---

## Slide 11 – Research Findings and Limitations

**Slide content**

* CG is memory-bound, so efficiency is limited by bandwidth, not by
  Amdahl's serial fraction.
* Communication design matters more than process count.
* Cache-sized problems inflate speedup to as much as 3.9×: benchmark
  large problems.
* Weak scaling on one node fails, because bandwidth is shared.
* Limitations: single node, no preconditioner, 7-point stencil, 1D
  decomposition, no `perf` counters, laptop variance.

**Speaker notes.** Separate clearly what the literature says (HPCG
motivation) from what we measured (the numbers on the previous slide).

---

## Slide 12 – Conclusion and Future Scope

**Slide content**

* A working, tested, reproducible HPCG-inspired CG benchmark with 4
  back-ends and an automated analysis pipeline.
* It confirms why HPCG complements HPL: sparse workloads rank machines
  by memory and network.
* Future work:
  * multigrid + SymGS preconditioner, 27-point stencil;
  * 3D decomposition;
  * communication–computation overlap and pipelined CG;
  * multi-node and GPU runs; hardware counters.

**Speaker notes.** End with the one-sentence takeaway: *for sparse
scientific codes, bytes per second matter more than flops per second.*
Then take questions.
