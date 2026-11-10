#!/usr/bin/env bash
# demo.sh - guided live demonstration for presentations / viva (~3 minutes).
#
# Runs each part of the project with a short on-screen explanation and
# waits for ENTER between steps so you can talk.
#   bash scripts/in_docker.sh bash scripts/demo.sh      (Windows + Docker)
#   bash scripts/demo.sh                                 (native Linux)
# Options: DEMO_GRID=64 (problem size), DEMO_NO_PAUSE=1 (run without waiting)
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
G="${DEMO_GRID:-64}"
BIN=build/hpcg_benchmark
B=$'\e[1m'; C=$'\e[36m'; Y=$'\e[33m'; N=$'\e[0m'

step() {  # step "title" "explanation"
    echo
    echo "${B}${C}==================================================================${N}"
    echo "${B}${C} $1${N}"
    echo "${B}${C}==================================================================${N}"
    echo "${Y}$2${N}"
    if [ "${DEMO_NO_PAUSE:-0}" != "1" ]; then read -r -p "   [press ENTER to run] " _; fi
}
run() { echo "${B}\$ $*${N}"; "$@"; }

step "0. Build (only if needed)" \
"CMake compiles the C++17 code with -O3, OpenMP and MPI into build/hpcg_benchmark."
if [ ! -x "$BIN" ]; then run bash scripts/build.sh; else echo "build/hpcg_benchmark already built - skipping."; fi

step "1. The machine we are running on" \
"Compiler, OpenMP and MPI versions, number of CPUs and memory - recorded with every result."
run "$BIN" --info

step "2. Automated tests (correctness first)" \
"10 test targets: matrix is symmetric positive definite, CG gives the exact answer,
OpenMP / MPI / hybrid give the same solution as sequential, MPI on 1-4 processes."
(cd build && run ctest)

step "3. Sequential baseline (grid ${G}^3)" \
"One core solves A x = b with Conjugate Gradient. Watch: iters, status=converged,
true_relres (accuracy), max_err (distance from the exact answer x = 1), t_median (time)."
run "$BIN" --mode sequential --grid "$G" --reps 3

step "4. OpenMP - 4 threads sharing memory" \
"Same problem, same tolerance. Loops are split across 4 threads; dot products use reductions.
Expect the same iteration count and answer, and a lower time."
OMP_NUM_THREADS=4 run "$BIN" --mode openmp --grid "$G" --reps 3

step "5a. MPI - 4 processes, Allgatherv (every process gets the WHOLE vector)" \
"Distributed memory: rows split over 4 processes. Every iteration all processes exchange
the full vector p. Note comm_frac = share of time spent communicating."
run mpirun --bind-to none -np 4 "$BIN" --mode mpi --comm allgather --grid "$G" --reps 3

step "5b. MPI - 4 processes, halo exchange (only neighbour data)" \
"Same problem, but each process only receives the boundary planes it needs.
Compare 'receives ... doubles' and comm_frac with 5a - much less communication."
run mpirun --bind-to none -np 4 "$BIN" --mode mpi --comm halo --grid "$G" --reps 3

step "6. Hybrid - 2 MPI processes x 2 OpenMP threads" \
"MPI between processes, OpenMP threads inside each. Only the main thread calls MPI (FUNNELED)."
OMP_NUM_THREADS=2 run mpirun --bind-to none -np 2 "$BIN" --mode hybrid --comm halo --grid "$G" --reps 3

step "7. Why speedup is limited: memory bandwidth" \
"STREAM triad measures how many GB/s the memory delivers with 1, 4 and 8 threads.
CG does only ~0.13 flops per byte, so it is limited by this bandwidth, not by CPU cores."
run build/stream_triad --threads 1,4,8 --mb 64 --reps 5

step "8. Full results (already measured, 50 configurations, grid 128^3)" \
"The complete campaign takes ~20 min (bash scripts/run_all.sh). Its results are saved:"
run sed -n '3,25p' results/summary.md

echo
echo "${B}Demo finished.${N} Graphs: results/graphs/   Dashboard: results/dashboard.html"
