# Literature Review

Research topic: *Performance Evaluation of Shared-Memory and
Distributed-Memory Parallelization Techniques for Conjugate Gradient-Based
HPC Workloads.*

This review separates **established facts from the cited literature** from
**our own observations**, which appear only in the project report's results
chapter. Bracketed numbers refer to the reference list at the end.

## 1. HPL and the TOP500

The High-Performance LINPACK (HPL) benchmark solves a dense linear system by
LU factorisation with partial pivoting. Since 1993 it has been the ranking
metric of the TOP500 list [7, 16]. Its work is dominated by dense
matrix-matrix multiplication (DGEMM). DGEMM performs O(n³) flops on O(n²)
data, so it has high arithmetic intensity and on modern systems reaches a
large fraction of theoretical peak [7].

Dongarra and Heroux argued that this very property makes HPL a poor
predictor for many scientific applications [1]. Those applications are
dominated by sparse operations, irregular memory access and global
communication. Optimising a system for HPL can push architectures toward
features that do not help these workloads.

## 2. HPCG: motivation and design

The High Performance Conjugate Gradient (HPCG) benchmark was proposed as a
complementary metric [1, 2, 3]. According to its authors and its reference
implementation [15], HPCG:

* generates a synthetic sparse SPD system from a 27-point stencil on a 3D
  grid, with a local subdomain per MPI process (a regular, weak-scaled
  problem);
* solves it with **preconditioned** CG. The preconditioner is a multigrid
  V-cycle with a symmetric Gauss-Seidel smoother. The main kernels are
  SpMV, SymGS, dot products (global reductions), vector updates (WAXPBY)
  and multigrid restriction and prolongation;
* uses MPI for distributed memory, with halo exchanges between neighbouring
  subdomains, and optionally OpenMP inside each process;
* runs a fixed number of iterations per CG set and includes validation and
  optimisation-checking phases. The result is reported in GFLOP/s.

Published HPCG results reach only a few percent of peak, far below HPL [2, 3].
This confirms that memory bandwidth and communication latency, not
floating-point throughput, limit this class of workload.

## 3. Conjugate Gradient

CG [4] is the method of choice for large sparse SPD systems. It needs only
matrix-vector products, dot products and vector updates. It minimises the
A-norm of the error over a growing Krylov subspace, and its convergence
rate depends on the condition number κ(A). For the 3D Poisson matrix
κ = O(h⁻²), so the iteration count grows roughly linearly with the grid
dimension [5, 6]. Shewchuk's tutorial [6] and Saad's monograph [5] give
accessible derivations and discuss practical issues: the residual is
updated recursively and slowly drifts from the true residual b − Ax, and
the method breaks down when pᵀAp ≤ 0.

## 4. Sparse matrix computation and CSR

Compressed Sparse Row (CSR) storage keeps the non-zeros row by row together
with their column indices and a row-pointer array [5]. CSR SpMV streams the
values and indices once and gathers entries of the input vector. It
performs 2 flops per non-zero for at least 12 bytes moved (8-byte value +
4-byte index), so it is strongly memory-bound. Williams et al. [12] showed
on multicore platforms that SpMV performance follows the sustainable memory
bandwidth rather than core count, and that the important optimisations are
reducing index storage, register and cache blocking, and NUMA-aware data
placement.

## 5. Memory bandwidth, cache and the roofline model

McCalpin's STREAM benchmark [11] measures sustainable memory bandwidth with
simple vector kernels (copy, scale, add, triad). It introduced the notion
of *machine balance* (flops per word of bandwidth). The roofline model [8]
bounds attainable performance by min(peak flops, arithmetic intensity ×
memory bandwidth), giving a visual model of whether a kernel is compute- or
memory-bound. CG's arithmetic intensity is well below 1 flop/byte, which
puts it firmly under the bandwidth roof [8, 12]. Problems that fit in the
last-level cache are bounded by the (higher) cache bandwidth instead of the
DRAM roof [18], which is why small problem sizes give misleadingly high
performance.

## 6. OpenMP shared-memory parallelisation

OpenMP [14, 21] parallelises loops with worksharing constructs. Reductions
give each thread a private partial sum that is combined at the end of the
construct. The specification does not fix the combination order, so
floating-point results can differ in the last bits between runs and thread
counts. Each worksharing loop ends with an implicit barrier, and fork/join
and barrier costs become significant when the loops are short [18]. Static
scheduling together with first-touch page placement keeps each thread on
the same data in every kernel [12, 18].

## 7. MPI distributed-memory parallelisation

MPI [13, 20] provides point-to-point and collective communication. In
distributed CG the matrix rows are partitioned. Each iteration needs (i) the
off-process vector entries referenced by the local rows (a halo exchange,
or in the simplest case an all-gather of the whole vector) and (ii) two
global reductions (`MPI_Allreduce`) for the dot products. The reductions
are global synchronisation points whose latency grows with process count
(typically O(log P)) and does not shrink with problem size. For this reason
they are widely identified as a scalability bottleneck of Krylov methods.
Halo exchanges move data proportional to the subdomain surface, whereas an
all-gather moves data proportional to the whole vector.

## 8. Hybrid MPI + OpenMP

Hybrid programming uses MPI between nodes (or sockets) and OpenMP threads
within them. Rabenseifner, Hager and Jost [17] analysed its benefits: fewer
and larger messages, less memory for halos and replicated data, and the use
of shared caches. They also analysed its pitfalls: idle threads during
communication in the master-only (`MPI_THREAD_FUNNELED`) style, and
sensitivity to thread and process placement and to pinning. The MPI
standard defines thread-support levels (SINGLE, FUNNELED, SERIALIZED,
MULTIPLE), which must be requested with `MPI_Init_thread` and checked [13].

## 9. Scalability laws

* **Amdahl's law** [9]: for a serial fraction s, the speedup on p units is
  bounded by 1 / (s + (1 − s)/p) (strong scaling).
* **Gustafson's law** [10]: when the problem grows with p (weak scaling),
  the scaled speedup is s + p(1 − s). Weak scaling is the regime HPCG is
  designed for.
* For memory-bound kernels on one shared-memory node, a third limit applies:
  once the memory bus is saturated, adding cores does not increase
  throughput. This is a bandwidth limit, not a serial fraction [8, 11, 18].

## 10. Performance modelling of HPCG

Marjanović, Gracia and Glass [19] built performance models of HPCG. They
showed that its performance can be predicted from memory bandwidth and
network characteristics, which supports the view that HPCG measures the
memory and interconnect subsystem rather than peak compute.

## 11. Gap addressed by this project

The official HPCG is a large code base with a multigrid preconditioner and
strict run rules, so it is hard to use for teaching. This project builds a
small, fully tested CG code that keeps HPCG's essential ingredients: a 3D
stencil SPD matrix, CSR SpMV, global dot products and halo exchange. With it
we compare sequential, OpenMP, MPI (two communication schemes) and hybrid
versions on one laptop, and explain the measurements with the roofline and
bandwidth arguments above. The differences from HPCG are listed in the
project report (§5 and §21).

## References

See `project_report.md`, §24 (same numbering).
