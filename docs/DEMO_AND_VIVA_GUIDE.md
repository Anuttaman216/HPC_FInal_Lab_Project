# Demo and Viva Guide

You don't need to have written the code to present it well. You need to
understand **the story**, be able to **run the demo**, and know **about 10
numbers and 10 ideas**. This guide gives you all three, in order.

Plan your time like this:

* **Tonight (about 90 min):** Parts 1–5. Read the story, practise the demo
  twice, memorise the cheat sheet.
* **Tomorrow before the session (10 min):** Part 2, the setup checklist.
* **During the viva:** Part 3 (the demo script) and Part 6 (questions and
  answers).

---

## Part 1 – The story of the project, in plain English

Read this three times. Everything else builds on it.

**1. Why the project exists.** Supercomputers are ranked in the TOP500
list using a benchmark called **HPL**. HPL solves a *dense* system of
equations, which is almost pure arithmetic, so it mostly measures how fast
the CPU can multiply numbers. Real scientific programs (weather, fluid
flow, structural engineering) solve *sparse* systems instead, and those
are slowed down by **moving data from memory**, not by arithmetic. So a
machine can rank highly on HPL and still be slow for real science.
In 2013 Jack Dongarra and Michael Heroux proposed **HPCG** (High
Performance Conjugate Gradient), a benchmark that measures this
"realistic" behaviour.

**2. What we built.** A smaller, educational version of HPCG. It:

* builds a big **sparse matrix**: up to 4 million equations, more than
  99.99 % zeros, coming from a 3D grid;
* solves `Ax = b` with the **Conjugate Gradient (CG)** method;
* does that in **4 ways**:
  * **Sequential**: one CPU core, the baseline;
  * **OpenMP**: several *threads* sharing the same memory (one program);
  * **MPI**: several *processes*, each with its own memory, sending
    messages;
  * **Hybrid**: MPI processes, each with OpenMP threads inside;
* measures time, speed (GFLOPS), speedup and efficiency, and draws graphs.

**3. What we found.** On the laptop, using 8 cores made the solver only
**1.4× faster**, not 8×. That is *not* a bug. We measured the memory
bandwidth separately (the STREAM triad test): it only grows from 21.9 to
28.9 GB/s, which is 1.32×, when going from 1 to 8 cores. CG needs about
7.5 bytes from memory for every arithmetic operation, so the cores mostly
**wait for memory**. The speedup ceiling equals the bandwidth ceiling.
**This is exactly the point of HPCG**: for sparse problems, memory speed
matters more than CPU speed.

**4. Second finding (MPI).** There are two ways for MPI processes to share
data:

* **Allgatherv**: every process receives the *whole* vector every
  iteration.
* **Halo exchange**: each process receives only the boundary values it
  actually needs from its neighbours.

With 8 processes, Allgatherv spent **51 %** of its time communicating and
was *slower than one core*. The halo version spent only **11.5 %** and was
**1.69× faster**. The lesson: *how* you communicate matters more than how
many processes you use.

**The one-sentence summary to say at the end:** *"For sparse scientific
codes like CG, bytes per second matter more than flops per second, which
is why HPCG complements HPL for ranking supercomputers."*

### Analogies you can use

* **CG** is like adjusting a guess again and again. Each round it gets
  closer to the true answer, and we stop when the error is below 10⁻⁸.
* **Memory-bound** is like a kitchen with 8 chefs but only one narrow
  door for ingredients. Adding chefs doesn't help once the door is the
  bottleneck.
* **OpenMP vs MPI:** OpenMP is several people working at the *same desk*
  (shared memory). MPI is people at *different desks* passing notes
  (messages).
* **Halo exchange:** each worker only asks their neighbours for the edge
  of their part, instead of photocopying the whole book for everyone
  (Allgatherv).

---

## Part 2 – Setup checklist

### Tonight (do this once to be sure it all works)

1. Open the folder `C:\Users\anutt\Downloads\HPC_Final_Lab_Project\HPCG-Performance-Analyzer`.
2. Start **Docker Desktop** from the Start menu. Wait until the bottom
   left of its window says *Engine running*.
3. Open **PowerShell** in the project folder: in File Explorer, click the
   address bar, type `powershell` and press Enter.
4. Run the demo:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
```

5. Press **ENTER** at each step. The whole demo takes about 3 minutes.
   Practise saying the "what to say" lines from Part 3 while it runs.
6. Open the results page:

```powershell
start results\dashboard.html
```

### Tomorrow, 10 minutes before

| Check | How |
|---|---|
| Laptop on charger, not on battery | Battery mode slows the CPU and makes timings worse |
| Docker Desktop running | Start menu → Docker Desktop → wait for *Engine running* (30–60 s) |
| Close heavy apps (Chrome tabs, games, Teams video) | They compete for CPU and memory and change timings |
| PowerShell open in the project folder | As in step 3 above |
| `results\dashboard.html` open in the browser | `start results\dashboard.html` |
| GitHub page open in another tab | https://github.com/Anuttaman216/HPC_FInal_Lab_Project |
| `report\project_report.md` open in VS Code or Notepad | To show the report if asked |

**If you present on a different computer:** install Docker Desktop, clone
the GitHub repository, and run the same `demo.ps1` command. The first run
builds the toolchain image (needs internet, about 3–5 minutes), and the
script compiles the code automatically.

---

## Part 3 – The live demo: what to run and what to say

Start the demo with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1
```

Each step prints a short explanation and waits for ENTER. Say the line in
quotes, then press ENTER.

| Step | What appears | What you say |
|---|---|---|
| **0. Build** | "already built – skipping" | "The code is C++17, compiled with CMake using -O3 optimisation, with OpenMP and MPI support. It runs in a Linux Docker container so that anyone can reproduce it." |
| **1. Info** | g++ 13.3, OpenMP 4.5, Open MPI 4.1.6, 16 CPUs, about 11.5 GiB RAM | "This is the environment. It is recorded with every result so the experiment is reproducible." |
| **2. Tests** | `100% tests passed … out of 10` | "Before measuring speed we prove correctness: 10 automated tests. They check the matrix is symmetric positive definite, CG reaches the exact answer, and OpenMP, MPI and hybrid give the same answer as sequential. MPI is tested with 1, 2, 3 and 4 processes; 3 gives an uneven split of rows." |
| **3. Sequential** | `iters=158 status=converged … max_err=2.9e-08 t_median≈0.5s` | "One core solves 262,144 equations. It converges in 158 iterations; the error from the exact answer is about 3×10⁻⁸. This is our baseline." |
| **4. OpenMP 4 threads** | same `iters=158`, lower time | "Same problem with 4 threads sharing memory. Same iterations and same accuracy, so it is the same computation, just faster." |
| **5a. MPI Allgatherv** | `receives 196608 doubles`, `comm_frac ≈ 0.3` | "Now 4 separate processes. With Allgatherv every process receives the whole vector every iteration: about 196,000 numbers. Around 30 % of the time is communication." |
| **5b. MPI halo** | `receives 4096 doubles`, lower `comm_frac` | "With halo exchange each process gets only the 4,096 boundary values it needs, 48 times less data, so communication drops a lot." |
| **6. Hybrid** | `procs=2 threads=2`, converged | "Hybrid: 2 MPI processes, each with 2 OpenMP threads. Only the main thread calls MPI. That is the FUNNELED threading level, and the code checks it." |
| **7. Bandwidth** | GB/s for 1, 4, 8 threads | "This explains the limited speedup. Memory bandwidth grows much less than the number of cores, and CG is limited by memory, not computation." |
| **8. Summary table** | the 128³ results | "The full experiment ran 50 configurations on 2.1 million equations. The best speedup is 1.40× with 8 threads, matching the bandwidth growth. At 8 processes Allgatherv is slower than one core, while halo is 1.69× faster than Allgatherv." |

Then switch to the browser (`dashboard.html`) and show **three graphs**:

1. **03_speedup.png.** "The dotted line is ideal speedup. Ours flattens at
   about 1.4. That is the memory wall."
2. **07_comm_overhead.png.** "Orange (Allgatherv) communication grows with
   processes; green (halo) stays small."
3. **11_roofline.png.** "The roofline: at 0.13 flops per byte the maximum
   possible performance is set by the bandwidth line, and our points sit
   just under it, at about 90 % of the hardware limit. So the code is
   efficient. The hardware is the limit."

> **Live numbers will differ a little** from the saved results (laptop
> temperature, background apps). That's normal. Say: *"Timings vary by a
> few percent run to run; that's why the benchmark reports the median of
> several repetitions."* For the official numbers, quote the saved
> results, not the live run.

### If the examiner asks you to run something specific

Open a Linux shell inside the container:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo.ps1 -Shell
```

Then type any of these commands:

```bash
./build/hpcg_benchmark --help
```

```bash
./build/hpcg_benchmark --mode sequential --grid 50
```

```bash
OMP_NUM_THREADS=8 ./build/hpcg_benchmark --mode openmp --grid 50
```

```bash
mpirun -np 2 ./build/hpcg_benchmark --mode mpi --comm halo --grid 50
```

```bash
OMP_NUM_THREADS=2 mpirun --bind-to none -np 3 ./build/hpcg_benchmark --mode hybrid --grid 50
```

```bash
cd build && ctest --output-on-failure && cd ..
```

```bash
./build/hpcg_benchmark --mode sequential --grid 2000
```

The last one is a deliberately oversized problem. It shows the memory
safety check refusing to run.

Type `exit` to leave the shell. Keep `--grid` at 128 or below during the
viva so each run finishes in seconds.

---

## Part 4 – Cheat sheet: numbers to memorise

| What | Value |
|---|---|
| Main test problem | 128 × 128 × 128 grid = **2,097,152 equations**, **14,581,760 non-zeros** |
| Sparsity | more than **99.99 %** zeros |
| Matrix entries | **6** on the diagonal, **−1** for each of up to 6 neighbours (7-point stencil) |
| Exact answer | all ones (because `b = A × 1`) |
| Stopping rule | relative residual below **10⁻⁸** |
| Iterations at 128³ | **296**, the same for every implementation |
| Sequential time | **5.95 s** |
| Best time | **4.24 s** (OpenMP 8 threads), so speedup **1.40×** |
| Memory bandwidth | **21.9 GB/s** (1 thread) → **28.9 GB/s** (best), only **1.32×** more |
| Arithmetic intensity of CG | about **0.13 flops per byte** (very low, hence memory-bound) |
| MPI Allgatherv, 8 procs | **0.80×** (slower than one core), **51 %** communication |
| MPI halo, 8 procs | **1.35×**, **11.5 %** communication, **1.69×** faster than Allgatherv |
| Best hybrid | 4 procs × 2 threads = **1.39×** |
| 16 threads | slower than 8 (hyper-threads add no bandwidth and compete) |
| Tests | **10/10 pass** |
| Experiments | **50 configurations**, 0 failed, 0 skipped |
| Laptop | Intel Core 7 240H, 16 logical CPUs; Docker on Windows 11 |
| Tools | C++17, g++ 13.3, OpenMP 4.5, Open MPI 4.1.6, CMake, Python (Pandas, Matplotlib) |

---

## Part 5 – Code tour (if asked "show me where …")

| They ask about | Open this file : line | What to point at |
|---|---|---|
| How the matrix is made | `src/matrix/matrix_generator.cpp:89` | `put(row, 6.0)` with the `−1` neighbours around it |
| Exact solution / RHS | `src/matrix/matrix_generator.cpp:97` | `generate_rhs`: b = A × 1 |
| Uneven row split for MPI | `src/matrix/matrix_generator.cpp:107` | `block_range`: first `n mod P` ranks get one extra row |
| CSR SpMV | `src/matrix/csr_matrix.cpp:61` | loop over `row_ptr`, `col_idx`, `values` |
| CG algorithm | `src/sequential/cg_sequential.cpp:105` | `alpha = rr / pAp`, then the x, r, p updates below it |
| Breakdown check | `src/sequential/cg_sequential.cpp:103` | `pAp <= 0` → stop safely |
| Timing starts | `src/sequential/cg_sequential.cpp:60` | timer starts after allocation, so setup isn't counted |
| OpenMP parallel loop | `src/openmp/cg_openmp.cpp:111` | `#pragma omp parallel for ... reduction(+ : pAp)` |
| MPI Allgatherv | `src/mpi/distributed_problem.cpp:100` | `MPI_Allgatherv(MPI_IN_PLACE ...)` |
| MPI halo exchange | `src/mpi/distributed_problem.cpp:110` | `MPI_Irecv` / `MPI_Isend` to neighbours only |
| MPI dot product | `src/mpi/cg_mpi.cpp:115` | `allreduce_sum(&pAp, 1)` (`MPI_Allreduce`) |
| MPI timing | `src/mpi/cg_mpi.cpp:147` | `MPI_MAX`: report the slowest process |
| Hybrid thread level | `src/hybrid/cg_hybrid.cpp:23` | `MPI_Query_thread` check for FUNNELED |
| MPI start-up | `src/main.cpp:19` | `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)` |
| FLOP formula | `include/common.hpp:80` | `2·nnz + 10n` per iteration |
| Tests | `tests/test_matrix.cpp`, `tests/test_mpi.cpp` | each `run_test("...")` is one check |
| Graphs | `python/visualize.py` | reads `results/benchmark.csv` only, never invents data |

---

## Part 6 – Viva questions and model answers

Answer in your own words; the key point is in **bold**.

### A. Basics and motivation

**Q1. What is your project in one line?**
An HPCG-inspired benchmark that solves a large sparse system with the
Conjugate Gradient method using sequential, OpenMP, MPI and hybrid
versions, and analyses **why sparse solvers don't scale like dense ones**.

**Q2. What is HPC?**
Using many processors together to solve problems faster or to solve
bigger problems, on clusters of multicore nodes connected by a fast
network.

**Q3. What is HPL? Why is it not enough?**
HPL solves a *dense* linear system by LU factorisation and ranks the
TOP500. It is **compute-bound**: lots of arithmetic per byte of data.
Most real scientific codes are **memory- and communication-bound**, so
HPL overstates how fast machines are for them.

**Q4. What is HPCG?**
The High Performance Conjugate Gradient benchmark (Dongarra, Heroux and
Luszczek, 2013–2016). It solves a sparse 3D-stencil system with CG
preconditioned by multigrid. It stresses **memory bandwidth and network
latency**, like real applications. Top machines reach only a few percent
of peak on HPCG.

**Q5. Is your project the official HPCG?**
**No.** It is HPCG-*inspired* and educational. Official HPCG uses a
27-point stencil, a multigrid preconditioner, a 3D domain decomposition
and strict run rules. We use a 7-point stencil, plain CG and a 1D row
split. Our numbers are not HPCG or TOP500 results.

**Q6. Why did you choose this topic / what is new?**
It shows experimentally *why* HPCG was needed. We added a direct
comparison of two MPI communication schemes and a **roofline analysis
with measured bandwidth**, which explains the low speedup.

### B. Mathematics

**Q7. What equation are you solving?**
The 3D Poisson equation, discretised on a grid with a 7-point stencil.
Each grid point gives one equation, `6·u(centre) − Σ u(neighbours) = b`.
In matrix form that is `Ax = b`.

**Q8. What is a sparse matrix? How sparse is yours?**
A matrix that is mostly zeros, so only the non-zeros are stored. Ours has
at most 7 non-zeros per row out of 2 million columns, so **more than
99.99 % zeros**.

**Q9. What is CSR format?**
Compressed Sparse Row uses three arrays:

* `values`: the non-zeros, row by row;
* `col_idx`: the column of each non-zero;
* `row_ptr`: where each row starts.

It saves memory and lets SpMV read the data sequentially.

**Q10. What is SPD and why does it matter?**
Symmetric (A = Aᵀ) and positive definite (xᵀAx > 0 for every non-zero x).
**CG is only guaranteed to converge for SPD matrices.**

**Q11. How do you know your matrix is SPD?**
Proof: it is symmetric, has a positive diagonal and is diagonally dominant
(6 ≥ sum of six 1s), strictly so on boundary rows, and the grid is
connected, which together imply SPD. Tests check it too: a **Cholesky
factorisation succeeds** (that only works for SPD matrices), and the
smallest eigenvalue matches its exact formula.

**Q12. Explain the Conjugate Gradient algorithm.**
Start with a guess x₀ and residual r = b − Ax (the error signal), and set
the search direction p = r. In each iteration:

1. α = (r·r)/(p·Ap), the step size;
2. x = x + αp;
3. r = r − αAp;
4. β = (r_new·r_new)/(r·r);
5. p = r + βp.

Stop when ‖r‖/‖b‖ < 10⁻⁸. Each step uses **one matrix-vector product,
two dot products and three vector updates**.

**Q13. Why does CG work? What is special about it?**
It chooses search directions that are A-conjugate, so it never undoes
earlier progress. It minimises the error over a growing Krylov subspace.
In exact arithmetic it finishes in at most n steps; in practice it gets
close much sooner.

**Q14. What is the residual? What is the relative residual?**
r = b − Ax measures how wrong the current answer is. The relative
residual is ‖r‖/‖b‖, which doesn't depend on scale. We stop at 10⁻⁸.

**Q15. Why does the iteration count grow with grid size (41 → 368)?**
The condition number κ grows like (grid size)², and CG needs roughly √κ
iterations, so **iterations grow linearly with grid size**. Our data
confirms it: about 2.3–2.6 × N.

**Q16. Why b = A × 1?**
Then the **exact answer is known** (all ones), so we can measure the true
error, not just the residual. HPCG uses the same trick.

**Q17. What happens if the matrix is not SPD?**
p·Ap can become ≤ 0. The code detects this ("breakdown") and stops
safely. It also detects NaN or infinity. These cases are tested.

### C. Parallel programming

**Q18. Difference between OpenMP and MPI?**
OpenMP uses **threads in one process sharing memory**, parallelised with
`#pragma omp` directives. MPI uses **separate processes with private
memory** that exchange explicit messages, and it can run across many
computers.

**Q19. How did you parallelise CG with OpenMP?**
Each loop (SpMV, vector updates) is `#pragma omp parallel for
schedule(static)`. Dot products use `reduction(+:sum)`, so each thread
sums privately and the results are combined at the end. There are **no
race conditions**.

**Q20. What is a race condition? How did you avoid it?**
Two threads updating the same variable at the same time, giving a wrong
result. We avoid it because each thread writes **only its own rows**, and
dot products use reductions instead of a shared counter.

**Q21. What is a reduction?**
Combining partial results from all threads (or processes) into one value,
for example a sum. OpenMP uses `reduction(+:x)`; MPI uses
`MPI_Allreduce`.

**Q22. Why static scheduling?**
All iterations cost the same, and static scheduling gives each thread the
**same rows in every loop**, which keeps its data in its cache.

**Q23. How is the work divided in MPI?**
The rows are split into contiguous blocks, one per process. If the
division is uneven, the first (n mod P) processes get one extra row.
**Each process generates only its own rows**, so no process holds the
whole matrix.

**Q24. Why do MPI processes need to communicate in CG?**

1. SpMV for a row needs vector values from neighbouring rows, which may
   belong to another process. That needs an exchange.
2. Dot products need the **sum over all processes**, which needs
   `MPI_Allreduce`, twice per iteration.

**Q25. What is MPI_Allreduce?**
A collective operation: every process contributes a value, the values are
combined (summed), and **every process receives the result**. We use it
for α and β so that all processes agree.

**Q26. Allgatherv vs halo exchange?**
* **Allgatherv**: every process gets the *entire* vector each iteration.
  It is simple but sends O(n) data per process, so it doesn't scale.
* **Halo**: each process gets only the boundary values (one grid plane
  from each neighbour) using `MPI_Isend`/`MPI_Irecv`.

At 4 processes on 64³ that is 196,608 values versus 4,096.

**Q27. Why did MPI with 8 processes using Allgatherv get slower than sequential?**
Communication took **51 % of the time**. Each process received about 1.8
million values (14.7 MB) every iteration, which cost more than the
computation saved.

**Q28. How do you make sure MPI processes don't deadlock?**
Every decision (stop or continue, α, β) uses values from `MPI_Allreduce`,
which are **identical on all processes**. So all processes run the same
sequence of collective calls. The tests check that the iteration counts
are equal on every rank.

**Q29. What is hybrid MPI + OpenMP? Why use it?**
MPI between processes (or nodes) and OpenMP threads inside each process.
It means fewer processes, which gives fewer messages and less duplicated
memory, and it uses shared memory within a node. On real clusters this is
the standard approach.

**Q30. What is MPI_THREAD_FUNNELED?**
A thread-support level where the program can have threads but **only the
main thread calls MPI**. Our hybrid code calls MPI only outside OpenMP
regions. It requests this level with `MPI_Init_thread` and checks it.

**Q31. Why `--bind-to none`?**
By default Open MPI pins each process to one core. In hybrid mode that
would force all of a process's threads onto **one core**. `--bind-to none`
lets the operating system spread them out.

### D. Performance analysis

**Q32. Define speedup and efficiency.**
* Speedup S(p) = T(sequential) / T(p).
* Efficiency E(p) = S(p)/p × 100 %, where p = processes × threads.

Example: 8 threads gave S = 1.40, so E = 17.5 %.

**Q33. Why is your speedup only 1.4× on 8 cores?**
**Memory bandwidth.** CG does about 0.13 flops per byte, so it is limited
by how fast memory delivers data. We measured bandwidth growing only 1.32×
from 1 to 8 cores, and the speedup (1.40×) matches it. The cores are
waiting for memory.

**Q34. Is low efficiency a sign of bad code?**
**No.** The roofline shows we reach about 90 % of the measured memory
bandwidth. The code is near the hardware limit; the hardware is the
bottleneck. This is exactly what HPCG demonstrates.

**Q35. What is the roofline model?**
Attainable performance = min(peak FLOPS, arithmetic intensity × memory
bandwidth). At low intensity (CG: 0.13 flops per byte) the bandwidth line
is the limit: 0.133 × 28.9 GB/s ≈ 3.8 GFLOPS. We measured 3.5.

**Q36. What is arithmetic intensity? How did you get 0.13?**
Flops divided by bytes moved. Per iteration we do 2·nnz + 10n flops and
move about 12·nnz + 96n bytes (12 bytes per non-zero: an 8-byte value and
a 4-byte index; 96 bytes per row for the vectors). With about 7 non-zeros
per row that gives about 0.13.

**Q37. What is STREAM triad?**
A standard memory-bandwidth test: `a[i] = b[i] + s·c[i]` over big arrays,
counting 24 bytes per element. We use it to measure the bandwidth "roof".

**Q38. What are Amdahl's and Gustafson's laws?**
* **Amdahl:** with a serial fraction s, speedup ≤ 1/(s + (1−s)/p), the
  limit for a *fixed* problem size (strong scaling).
* **Gustafson:** if the problem grows with p (weak scaling), the scaled
  speedup can keep growing.

Our limit is neither of these: it is the shared memory bandwidth.

**Q39. Strong scaling vs weak scaling?**
* **Strong:** the same problem, more cores, with the goal of less time.
* **Weak:** the problem grows with the cores, with the goal of the same
  time.

We did both. Weak scaling fails on one laptop because all cores share
one memory bus; on a cluster each node brings its own memory.

**Q40. Why is 16 threads slower than 8?**
The laptop has 16 *logical* CPUs but fewer physical cores (hyper-threading,
plus a mix of performance and efficiency cores). Extra hyper-threads add
**no memory bandwidth** and compete with each other, and with the VM and
Windows.

**Q41. Why are small problems faster per equation (up to 3.9× speedup)?**
They **fit in the CPU cache**, which is much faster than main memory. That
gives misleadingly good numbers, so benchmarks like HPCG require problems
much larger than the cache. Our main problem is 263 MB.

**Q42. What is GFLOPS and how do you calculate it?**
Billions of floating-point operations per second:
`GFLOPS = (counted operations) / (time × 10⁹)`, with 2·nnz + 10n
operations per iteration. Communication is not counted as useful work.

**Q43. How did you measure time fairly?**
* The timer covers only the solve, not matrix generation.
* There is 1 warm-up run, then 5 timed runs, and we report the
  **median**.
* For MPI we report the **slowest process** (MPI_MAX).
* Every version solves the same problem to the same tolerance, in the
  same 296 iterations.

**Q44. Why the median, not the average?**
The median is not affected by outliers. One slow run, for example caused
by a background process, doesn't distort it.

**Q45. Why does MPI with 1 process take longer than sequential (6.48 s vs 5.95 s)?**
It is the overhead of the distributed code path: copying into the
exchange buffer, bookkeeping and 592 `MPI_Allreduce` calls (two per
iteration). It is small and constant.

### E. Implementation, testing, tools

**Q46. Which tools and languages did you use, and why?**
* **C++17:** fast, with direct control of memory.
* **OpenMP and Open MPI:** the industry standards.
* **CMake:** portable builds.
* **Python, Pandas and Matplotlib:** analysis and graphs.
* **Docker:** the Windows laptop had no compiler, so a Linux container
  makes the setup reproducible anywhere.

**Q47. How do you know your results are correct?**
* 10 automated tests: matrix properties, the exact solution, and parallel
  results matching sequential.
* Every benchmark run checks the true residual and the error from the
  exact answer (about 7.7×10⁻⁸).
* All 50 configurations converged with the same 296 iterations.

**Q48. Why compare with a tolerance and not exact equality?**
Floating-point addition isn't associative, so parallel sums in a
different order give slightly different last digits. We even observed
OpenMP giving bit-different results between identical runs. We compare
within 10⁻⁸.

**Q49. Why store column indices as 32-bit integers?**
SpMV is memory-bound, and 4-byte indices instead of 8-byte cut the
matrix traffic by 25 % (12 instead of 16 bytes per non-zero).

**Q50. What is kernel fusion?**
Combining two loops into one, for example SpMV together with its dot
product, so the data is read from memory once instead of twice. All four
versions use the same fusion, so the comparison is fair.

**Q51. What are the limitations?**
1. Single laptop: MPI communication goes through shared memory, not a
   real network.
2. No multigrid preconditioner, a 7-point stencil and a 1D split, so it is
   not official HPCG.
3. The laptop and VM add timing noise.
4. No hardware counters (`perf` doesn't work in Docker), so memory traffic
   is modelled.

**Q52. Future work?**
* Add the multigrid preconditioner and the 27-point stencil.
* Use a 3D domain decomposition.
* Overlap communication with computation.
* Run on a real multi-node cluster.
* Add a GPU version (CUDA).
* Measure hardware counters on bare metal.

**Q53. What would change on a real supercomputer?**
Each node has its own memory bandwidth, so MPI across nodes would scale
much better (weak scaling). The network latency of `MPI_Allreduce` would
then become the main bottleneck. The halo approach would matter even
more.

**Q54. What did you personally learn?**
(Make this personal.) For example: that parallel speed depends on memory
and communication, not just core count; how to measure performance
fairly; and how to verify parallel code against a sequential baseline.

### If you don't know an answer

Say: *"I'm not certain; my understanding is … and that is something I
would verify in the report / code."* Then point to `report/project_report.md`,
which has 24 sections, or to the relevant file. Honesty scores better
than guessing.

---

## Part 7 – If something goes wrong during the demo

| Problem | Fix |
|---|---|
| "Docker Desktop is not running" | Start Docker Desktop, wait for *Engine running*, then run the command again |
| `the input device is not a TTY` | You are in Git Bash. Use **PowerShell**, or add `-NoPause` to `demo.ps1` |
| "running scripts is disabled" | Use the full command `powershell -ExecutionPolicy Bypass -File scripts\demo.ps1` |
| `not enough slots` from mpirun | Use fewer processes (`-np 2`) |
| Hybrid is very slow | Make sure `--bind-to none` is in the command |
| Times look worse than the report | Plug in the charger and close apps. Explain that live timings vary and the saved medians are the official results |
| Nothing works at all | Show the **saved** results instead: `results/dashboard.html`, `results/summary.md`, `results/logs/`, `results/test_results.txt`. All of it comes from real runs |

---

## Part 8 – Structure of a 5-minute presentation (if asked to explain first)

1. **Problem (30 s):** HPL ranks supercomputers by arithmetic speed, but
   real science is limited by memory, which is why HPCG exists.
2. **What we built (45 s):** sparse matrix with 2 million equations, CG
   solver, 4 versions (sequential, OpenMP, MPI with 2 communication
   schemes, hybrid), 10 tests, automated benchmarks and graphs.
3. **Live demo (2 min):** Part 3.
4. **Results (1 min):** 1.40× speedup on 8 cores, which equals the 1.32×
   bandwidth growth; halo is 1.69× faster than Allgatherv; roofline shows
   about 90 % of the hardware limit.
5. **Conclusion (30 s):** "For sparse codes, bytes per second matter more
   than flops per second. That is why HPCG complements HPL."

Slide-by-slide notes are in `report/presentation_outline.md`.
