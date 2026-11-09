# Design decisions and reasoning

Each decision below gives the **choice**, the **reasoning**, and the
**alternatives** considered. Facts about HPCG come from its published
papers and official code (see report §24). Statements about performance on
our machine come from our own measurements in `results/`.

---

### D1. Build an HPCG-*inspired* code, not a port of the official HPCG

* **Choice:** A small CG benchmark (about 2,300 lines of C++) that keeps
  HPCG's essential ingredients:
  * a 3D-stencil SPD matrix,
  * CSR SpMV,
  * global dot products,
  * halo exchange.
* **Reasoning:** The official HPCG has a multigrid preconditioner, strict
  run rules and a large code base. For a course project the goal is to
  *understand* why sparse CG behaves differently from HPL, and that needs
  code a student can read and modify. We never claim official results.
* **Alternative:** Running the official HPCG binary. It gives a single
  number but teaches little about the parallel techniques.

### D2. Use the 7-point Poisson stencil with Dirichlet boundaries

* **Choice:** Diagonal 6 and −1 for each of the (up to 6) neighbours. Out-of-grid
  neighbours are dropped.
* **Reasoning:**
  * The assignment asked for it.
  * It is the classic model problem.
  * SPD can be *proved*: symmetric and irreducibly diagonally dominant
    with a positive diagonal.
  * Its eigenvalues are known in closed form, so tests can check them
    exactly.
* **Alternative:** HPCG's 27-point stencil. It has more non-zeros per row
  but the same memory-bound character. Listed as future work.

### D3. Set the right-hand side to `b = A·1`

* **Reasoning:** The exact solution is then known (all ones), so every run
  can report its true error `max|x − 1|`, not just a residual. HPCG uses
  the same idea. The construction is deterministic, so runs are
  reproducible.

### D4. Store column indices as 32-bit and row pointers as 64-bit

* **Reasoning:** SpMV is limited by memory traffic. A 4-byte index instead
  of 8 bytes cuts matrix traffic from 16 to 12 bytes per non-zero (−25 %).
  64-bit row pointers allow more than 2³¹ non-zeros. The generator refuses
  grids that would overflow 32-bit columns.

### D5. Use the same fused kernels in every implementation

* **Choice:** SpMV fused with `(p,Ap)`, and the updates fused with `(r,r)`,
  in sequential, OpenMP, MPI and hybrid alike.
* **Reasoning:** Fair comparison. If only the parallel versions were
  fused, part of their "speedup" would come from fewer memory passes
  rather than from parallelism. With identical kernels, time differences
  come from parallelisation alone.

### D6. Use a per-kernel `omp parallel for` with a static schedule

* **Reasoning:** This is the simplest correct structure. The reductions
  avoid races, and the implicit barriers give exactly the synchronisation
  CG needs (SpMV must see all of `p`). A static schedule gives every thread
  the same index range in every kernel, which helps cache reuse and
  first-touch placement.
* **Alternative:** One persistent parallel region per solve. It has less
  fork/join overhead but makes the scalar recurrences (α, β, the stop test)
  harder to keep race-free. Listed as future work. At our problem sizes
  fork/join cost is negligible compared with memory time.

### D7. Have each MPI rank generate only its own rows

* **Reasoning:** No rank ever holds the full matrix, so memory scales with
  the number of ranks and no scatter step is needed. HPCG also generates
  locally. Because the generator is deterministic, the distributed matrix
  is bit-identical to the corresponding slice of the sequential one, and a
  test checks this.

### D8. Use a contiguous row-block distribution with an uneven split

* **Reasoning:** It is simple, keeps neighbouring rows on the same rank
  (good locality for a lexicographically ordered stencil), and makes the
  column range a rank needs contiguous. Block sizes differ by at most one
  row, so the load is balanced even when `n mod P ≠ 0`. A test with 3
  ranks covers this.
* **Alternative:** A 3D domain decomposition (as in HPCG). It has a
  smaller halo surface at large `P` but is more complex. Listed as future
  work.

### D9. Offer two communication schemes: Allgatherv and halo

* **Reasoning:** The assignment allowed the replicated-vector
  `MPI_Allgatherv` for simplicity, and we kept it as the baseline. We added
  the halo exchange because the research question is about communication
  overhead. Having both on the same problem isolates the effect of the
  communication design.
* **Measured result** (our observation): at 8 processes the halo scheme is
  1.69× faster, and Allgatherv is slower than sequential.

### D10. Compute all decisions from `MPI_Allreduce` results

* **Reasoning:** If ranks decided convergence from *local* values, they
  could leave the loop at different iterations, and the next collective
  call would deadlock. Using reduced values guarantees that every rank
  executes the same sequence of collectives. The tests also check that all
  ranks report the same iteration count.

### D11. Use the FUNNELED threading level for hybrid

* **Reasoning:** This is the minimum level needed: only the main thread
  calls MPI, and only between OpenMP regions. It avoids the overhead and
  complexity of `MPI_THREAD_MULTIPLE`. The code verifies the provided level
  and refuses to run otherwise.

### D12. Define the timing boundaries

* **Choice:** Time from the initial residual to the last iteration. Exclude
  generation, allocation and verification. MPI reports the *maximum* time
  over ranks.
* **Reasoning:**
  * We measure the solver, not set-up.
  * The slowest rank determines when the job finishes.
  * The same boundaries apply to every mode, which is required for valid
    speedups.

### D13. Report the median of repetitions after a warm-up run

* **Reasoning:** Laptop timings are noisy (turbo, thermals, background
  processes). The median resists outliers, and the warm-up removes
  first-touch and cold-cache effects. Min, max, mean and standard deviation
  are kept in the CSV so the variance is visible.

### D14. Use a modelled flop count

* **Reasoning:** Hardware flop counters were unavailable (no `perf` in the
  VM). A fixed analytic count (`2·nnz + 10n` per iteration) is standard
  practice, as HPCG does the same. It also credits every implementation
  with the same useful work, so communication never counts as flops.

### D15. Add a STREAM-style bandwidth probe and a roofline graph

* **Reasoning:** Low parallel efficiency could mean bad code or a
  hardware limit. Measuring the bandwidth roof separates the two. The
  measurements put CG at about 90 % of triad bandwidth, which shows that
  the limit is hardware.

### D16. Run in a Docker Ubuntu container

* **Reasoning:** The development PC (Windows 11) had no C++ compiler or
  MPI. A container gives the exact Linux, g++ and Open MPI toolchain the
  assignment specified, with no system-wide installation, and anyone with
  Docker can reproduce it (`docker/Dockerfile`). The trade-off is that the
  container runs inside a VM, which adds timing noise and blocks `perf`.
  This is documented as a limitation.

### D17. Write tests without a test-framework dependency

* **Reasoning:** A 75-line header (`tests/test_utils.hpp`) is enough, and it
  keeps the build dependency-free. The MPI tests combine verdicts with
  `MPI_Allreduce(LAND)`, so a failure on one rank cannot deadlock the
  others.

### D18. Compare floating-point results with a tolerance

* **Reasoning:** Parallel reductions sum in a different order, and
  floating-point addition is not associative. We also *observed* that
  OpenMP reductions differ between identical runs (libgomp combines
  partial sums in arrival order). Tests therefore use tolerances such as
  1e-8 on the solution and allow ±1 iteration.

### D19. Use a static HTML dashboard instead of Streamlit

* **Reasoning:** Streamlit is not in the Ubuntu package archive and would
  add a heavy dependency. The assignment allowed static graphs as the
  fallback. A self-contained HTML file with sortable tables and embedded
  graphs works offline in any browser.

### D20. Skip, never fake, unsupported configurations

* **Reasoning:** The benchmark script skips configurations needing more
  cores than exist and records the reason in `results/run_status.csv`.
  The analysis drops runs that did not converge, with a warning. Graphs
  are drawn only from data that exists. No value is ever invented or
  interpolated.
