# Step-by-step walkthrough

This guide takes you from a fresh clone to your own graphs. For each step
it says **what to run**, **what happens**, **why it is done**, and **what
you should see**.

---

## Step 0 – Choose an environment

| You have… | Use |
|---|---|
| Linux (Ubuntu/Debian) or WSL | the native toolchain, Step 1A |
| Windows or macOS with Docker Desktop | the container, Step 1B |

**Why:** The code needs a C++17 compiler with OpenMP, an MPI library and
CMake. These are standard on Linux; elsewhere the container provides them.

## Step 1A – Native install (Linux)

```bash
sudo apt install build-essential cmake openmpi-bin libopenmpi-dev python3 python3-numpy python3-pandas python3-matplotlib
```

```bash
bash scripts/setup.sh
```

**What happens:** `setup.sh` checks for g++, cmake and make. It compiles a
tiny OpenMP program to prove OpenMP really works, then checks `mpicxx` and
`mpirun`, the Python packages, and (optionally) `perf`.

**Expected:** `All required dependencies found.` Anything missing is
printed with the exact install command.

## Step 1B – Container (Windows/macOS)

```bash
docker build -t hpcg-analyzer -f docker/Dockerfile docker
```

**What happens:** This builds an Ubuntu 24.04 image with g++, Open MPI,
CMake and the Python libraries (about 300 MB, built once). From then on,
prefix any command with `bash scripts/in_docker.sh`, for example:

```bash
bash scripts/in_docker.sh bash scripts/setup.sh
```

`in_docker.sh` mounts the project at `/work`, so files the container
creates (such as `build/` and `results/`) appear in your folder.

## Step 2 – Build

```bash
bash scripts/build.sh --clean
```

**What happens:** CMake detects OpenMP and MPI, then compiles:

* `hpcg_core`: a static library holding all solvers;
* `hpcg_benchmark`: the main program;
* `stream_triad`: the bandwidth probe;
* the test programs.

The configure step prints `HPCG: OpenMP=ON MPI=ON build=Release flags='-O3 -DNDEBUG'`.

**Why Release / -O3:** Benchmarks must use optimised code. Debug builds
can be 5–10× slower.

**If OpenMP or MPI is missing:** CMake prints a warning and those modes
are *not built*. The program then refuses to run them instead of silently
running sequential code.

## Step 3 – Test

```bash
cd build && ctest --output-on-failure && cd ..
```

**What happens:** CTest runs 10 targets:

| Test | What it proves |
|---|---|
| `test_matrix` | CSR structure, nnz formula, symmetry, positive definiteness (Cholesky + exact smallest eigenvalue), SpMV correctness, row-block slicing, `b = A·1`, uneven split |
| `test_sequential` | known 3×3 solution, convergence to `x = 1`, residual history, zero RHS, breakdown, NaN and Inf detection, iteration limit, fixed-iteration mode, stability on 30³ |
| `test_openmp` | thread team really created, solution matches sequential for 1, 2, 4 and all threads, reproducibility to rounding, error handling |
| `test_mpi_np1…4` | correct row coverage, correct halo and Allgatherv data, MPI and hybrid (both schemes) match sequential, identical iteration count on every rank, zero RHS and breakdown on all ranks |
| `e2e_*` | the real executable runs end to end in sequential, OpenMP and MPI-halo modes |

**Expected:** `100% tests passed, 0 tests failed out of 10`.

**Why 3 MPI ranks:** With 3 ranks the rows do not divide evenly, so the
uneven-distribution code path is exercised.

## Step 4 – Try single runs

```bash
./build/hpcg_benchmark --info
```

This prints the compiler, OpenMP version, MPI library, CPU count and
memory.

```bash
./build/hpcg_benchmark --mode sequential --grid 30
```

```bash
OMP_NUM_THREADS=4 ./build/hpcg_benchmark --mode openmp --grid 30
```

```bash
mpirun -np 4 ./build/hpcg_benchmark --mode mpi --grid 30 --comm halo
```

```bash
OMP_NUM_THREADS=2 mpirun --bind-to none -np 2 ./build/hpcg_benchmark --mode hybrid --grid 30
```

**What you see:** a header with problem metadata (rows, nnz, sparsity,
estimated memory), one line per timed repetition, then a `RESULT` line:

```
RESULT mode=openmp comm=none grid=30x30x30 rows=27000 nnz=183600 procs=1 threads=4
       iters=76 status=converged relres=9.617e-09 true_relres=9.617e-09 max_err=1.325e-08
       t_median=0.002267s comm_frac=0.000 GFLOPS=21.580 mem=3.3MiB
```

**How to read it:**

* `iters`: CG iterations needed.
* `status=converged`: the tolerance was reached.
* `relres` / `true_relres`: the recursive and recomputed residuals, which
  should agree.
* `max_err`: distance from the exact solution (all ones).
* `t_median`: median solve time.
* `comm_frac`: share of time spent in MPI communication.

**Why 30³ looks so fast:** 27,000 rows fit in the CPU cache. Use
`--grid 128` to see realistic, memory-bound performance (see the report,
§19.5).

**Exit codes:**

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | bad arguments |
| 2 | mode unavailable, or problem too big for memory |
| 3 | not converged |
| 4 | breakdown or NaN |

## Step 5 – Run the benchmark campaign

```bash
bash scripts/run_benchmarks.sh
```

The script measures memory bandwidth first, then runs these experiments:

| Experiment | Configurations | Purpose |
|---|---|---|
| Strong scaling (128³, tol 1e-8, 5 repetitions) | sequential; OpenMP 1–16 threads; MPI 1–8 procs × {allgather, halo}; hybrid 2×2…8×2 | speedup and efficiency on a fixed problem |
| Size sweep 16³ → 160³ | sequential, OpenMP 8t, MPI-halo 4p | cache vs. DRAM regime, iteration growth |
| Weak scaling 64×64×(64·p), fixed 100 iterations | OpenMP and MPI-halo | constant work per unit |
| Convergence histories | sequential 64³ and 128³ | residual vs. iteration |

It also records the hardware and software in `results/system_info.txt`
and the exit status of every run in `results/run_status.csv`.

**Duration:** about 15–20 minutes. For a 1-minute test run:

```bash
QUICK=1 bash scripts/run_benchmarks.sh
```

**Customise:** `GRID=96 REPS=3 SIZES="32 64 96" bash scripts/run_benchmarks.sh`.

**Why 128³:** The problem is 263 MB, far larger than any CPU cache, so it
measures the memory-bound regime that HPCG targets, and it still fits
comfortably in laptop RAM.

## Step 6 – Analyse and plot

```bash
python3 python/analyze.py
```

```bash
python3 python/visualize.py
```

```bash
python3 python/dashboard.py
```

* `analyze.py` computes speedup, efficiency, weak-scaling efficiency,
  effective bandwidth and arithmetic intensity. It writes
  `results/summary.csv` and a Markdown version, `results/summary.md`.
* `visualize.py` draws 11 graphs into `results/graphs/`.
* `dashboard.py` builds `results/dashboard.html`. Open it in a browser.

**Rules the scripts follow:** speedup always uses the sequential run with
the same grid and tolerance; non-converged runs are excluded; any graph
whose data is missing is skipped rather than faked.

## Step 7 – Everything at once

```bash
bash scripts/run_all.sh
```

This runs Steps 1–6 in order and stops at the first failure.

## Step 8 – Read the results

Start with `results/summary.md`, then the graphs, then the report:

* `report/project_report.md`: full report. §19 covers results, §20 the
  discussion.
* `report/methodology.md`: exact measurement rules.
* `report/literature_review.md`: background and references.
* `docs/GLOSSARY.md`: definitions of every term.
* `docs/DESIGN_DECISIONS.md`: why the code is built the way it is.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `mpirun` refuses to run as root | inside Docker this is pre-allowed; elsewhere run as a normal user |
| `There are not enough slots` | fewer cores than requested processes: use fewer `-np`, or add `--oversubscribe` (timings are then meaningless) |
| hybrid mode is very slow | Open MPI pinned threads to one core: add `--bind-to none` |
| `problem ... needs ~X MiB` | memory guard: choose a smaller `--grid` (or `--force` if you are sure) |
| `bash\r: command not found` | Windows line endings: the repo's `.gitattributes` enforces LF; re-clone or run `sed -i 's/\r$//' scripts/*.sh` |
| speedups look huge | problem too small (cache-resident): use `--grid 96` or larger |
