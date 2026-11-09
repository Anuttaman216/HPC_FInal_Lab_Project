#!/usr/bin/env bash
# run_benchmarks.sh - automated benchmark campaign.
#
# Experiments (all comparable runs share grid, tolerance and iteration cap):
#   strong : fixed problem (GRID^3), tolerance-based stop
#            sequential | OpenMP threads | MPI (allgather, halo) | hybrid
#   size   : performance vs. matrix size (sequential, OpenMP, MPI halo)
#   weak   : fixed work per processing unit (WEAK_BASE^3 rows per unit,
#            fixed WEAK_ITERS iterations, tol = 0) for OpenMP and MPI halo
#   bandwidth: STREAM-style triad for the roofline model
#   convergence: residual history of one sequential solve
#
# Environment overrides: GRID, REPS, TOL, MAX_ITERS, SIZES, WEAK_BASE,
# WEAK_ITERS, MAX_UNITS, ALLOW_OVERSUBSCRIBE=1, QUICK=1 (small smoke run),
# APPEND=1 (keep existing CSV).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${BIN:-$ROOT/build/hpcg_benchmark}"
STREAM_BIN="${STREAM_BIN:-$ROOT/build/stream_triad}"
RESULTS="$ROOT/results"
LOGS="$RESULTS/logs"
CSV="$RESULTS/benchmark.csv"
STATUS="$RESULTS/run_status.csv"

if [ "${QUICK:-0}" = "1" ]; then
    GRID="${GRID:-32}"; REPS="${REPS:-2}"; SIZES="${SIZES:-16 24 32}"
    WEAK_BASE="${WEAK_BASE:-24}"; WEAK_ITERS="${WEAK_ITERS:-20}"
else
    GRID="${GRID:-128}"; REPS="${REPS:-5}"; SIZES="${SIZES:-16 32 48 64 96 128 160}"
    WEAK_BASE="${WEAK_BASE:-64}"; WEAK_ITERS="${WEAK_ITERS:-100}"
fi
TOL="${TOL:-1e-8}"
MAX_ITERS="${MAX_ITERS:-5000}"
NPROC="$(nproc 2>/dev/null || echo 1)"
MAX_UNITS="${MAX_UNITS:-$NPROC}"
# --bind-to none: let the OS place ranks/threads. OpenMPI's default core
# binding would pin every hybrid rank's threads to a single core.
MPIRUN="${MPIRUN:-mpirun} --bind-to none"

[ -x "$BIN" ] || { echo "error: $BIN not found - run scripts/build.sh first" >&2; exit 1; }
mkdir -p "$LOGS" "$RESULTS/graphs"

if [ "${APPEND:-0}" != "1" ]; then
    for f in "$CSV" "$STATUS"; do
        [ -s "$f" ] && mv "$f" "$f.bak.$(date +%Y%m%d%H%M%S)"
    done
fi
[ -s "$STATUS" ] || echo "timestamp,label,mode,comm,grid,procs,threads,exit_code,status,log" > "$STATUS"

# ------------------------------------------------------------- metadata
{
    echo "# System information recorded $(date -Iseconds)"
    echo "## OS";        uname -a; [ -r /etc/os-release ] && grep PRETTY_NAME /etc/os-release
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo "NOTE: running under a Microsoft hypervisor (WSL2 / Docker Desktop VM)."
    fi
    echo "## CPU (lscpu)"; lscpu 2>/dev/null || echo "lscpu unavailable"
    echo "## Memory";     grep -E 'MemTotal|MemAvailable' /proc/meminfo 2>/dev/null
    echo "## Compiler";   g++ --version 2>/dev/null | head -n1
    echo "## CMake";      cmake --version 2>/dev/null | head -n1
    echo "## MPI";        mpirun --version 2>/dev/null | head -n1
    echo "## Python";     python3 --version 2>/dev/null
    python3 -c "import numpy,pandas,matplotlib;print('numpy',numpy.__version__,'pandas',pandas.__version__,'matplotlib',matplotlib.__version__)" 2>/dev/null
    echo "## Benchmark build"; "$BIN" --info
    echo "## perf"
    if command -v perf >/dev/null 2>&1 && perf stat -e task-clock true >/dev/null 2>&1; then
        echo "perf available"
    else
        echo "perf not usable in this environment (no matching linux-tools / perf_event access); hardware counters not collected"
    fi
    echo "## Campaign parameters"
    echo "GRID=$GRID REPS=$REPS TOL=$TOL MAX_ITERS=$MAX_ITERS SIZES='$SIZES' WEAK_BASE=$WEAK_BASE WEAK_ITERS=$WEAK_ITERS MAX_UNITS=$MAX_UNITS"
} > "$RESULTS/system_info.txt" 2>&1
echo "System metadata -> results/system_info.txt"

# ------------------------------------------------------------- helpers
ok_runs=0; failed_runs=0; skipped_runs=0

# run LABEL MODE COMM PROCS THREADS GRIDSPEC... -- extra args
run() {
    local label="$1" mode="$2" comm="$3" procs="$4" threads="$5"; shift 5
    local units=$((procs * threads))
    local tag="${label}_${mode}_${comm}_p${procs}_t${threads}_$(echo "$*" | tr -c 'a-zA-Z0-9' '_' | cut -c1-40)"
    local log="$LOGS/$tag.log"
    if [ "$units" -gt "$MAX_UNITS" ] && [ "${ALLOW_OVERSUBSCRIBE:-0}" != "1" ]; then
        echo "SKIP  $tag: needs $units processing units but only $MAX_UNITS available (set ALLOW_OVERSUBSCRIBE=1 to force)"
        echo "$(date -Iseconds),$label,$mode,$comm,\"$*\",$procs,$threads,,skipped_insufficient_cores,$log" >> "$STATUS"
        skipped_runs=$((skipped_runs + 1))
        return 0
    fi
    local cmd=("$BIN" --mode "$mode" --csv "$CSV" --label "$label" --threads "$threads" "$@")
    [ "$comm" != "none" ] && cmd+=(--comm "$comm")
    local start=$SECONDS
    if [ "$mode" = "mpi" ] || [ "$mode" = "hybrid" ]; then
        OMP_NUM_THREADS="$threads" $MPIRUN -np "$procs" "${cmd[@]}" >"$log" 2>&1
    else
        OMP_NUM_THREADS="$threads" "${cmd[@]}" >"$log" 2>&1
    fi
    local code=$?
    local status="ok"
    [ $code -ne 0 ] && status="failed"
    echo "$(date -Iseconds),$label,$mode,$comm,\"$*\",$procs,$threads,$code,$status,$log" >> "$STATUS"
    if [ $code -eq 0 ]; then
        ok_runs=$((ok_runs + 1))
        printf "OK    %-60s %4ds  %s\n" "$tag" $((SECONDS - start)) "$(grep -o 't_median=[0-9.]*s' "$log" | head -n1)"
    else
        failed_runs=$((failed_runs + 1))
        echo "FAIL  $tag (exit $code) - see $log"; tail -n 5 "$log" | sed 's/^/      /'
    fi
}

pow2_upto() {  # 1 2 4 8 ... <= $1, plus $1 itself
    local m=$1 v=1 out=""
    while [ $v -le "$m" ]; do out="$out $v"; v=$((v * 2)); done
    case " $out " in *" $m "*) ;; *) out="$out $m" ;; esac
    echo $out
}

COMMON=(--tol "$TOL" --max-iters "$MAX_ITERS" --reps "$REPS" --warmup 1 --quiet)

# ------------------------------------------------------------- bandwidth
if [ -x "$STREAM_BIN" ]; then
    echo "== Memory bandwidth (STREAM-style triad) =="
    "$STREAM_BIN" --threads "$(pow2_upto "$NPROC" | tr ' ' ',')" --mb 128 --reps 10 \
        > "$RESULTS/bandwidth.csv" 2> "$LOGS/stream_triad.log" && cat "$RESULTS/bandwidth.csv"
fi

# ------------------------------------------------------------- strong scaling
echo "== Strong scaling: grid ${GRID}^3, tol $TOL, $REPS reps =="
G=(--grid "$GRID")
run strong sequential none 1 1 "${G[@]}" "${COMMON[@]}"
for t in $(pow2_upto "$NPROC"); do run strong openmp none 1 "$t" "${G[@]}" "${COMMON[@]}"; done
for comm in allgather halo; do
    for p in 1 2 4 8; do run strong mpi "$comm" "$p" 1 "${G[@]}" "${COMMON[@]}"; done
done
for pt in "2 2" "2 4" "4 2" "2 8" "4 4" "8 2"; do
    set -- $pt; run strong hybrid halo "$1" "$2" "${G[@]}" "${COMMON[@]}"
done

# ------------------------------------------------------------- size sweep
echo "== Performance vs. matrix size: grids $SIZES =="
SREPS=(--tol "$TOL" --max-iters "$MAX_ITERS" --reps 3 --warmup 1 --quiet)
for g in $SIZES; do
    run size sequential none 1 1 --grid "$g" "${SREPS[@]}"
    run size openmp none 1 8 --grid "$g" "${SREPS[@]}"
    run size mpi halo 4 1 --grid "$g" "${SREPS[@]}"
done

# ------------------------------------------------------------- weak scaling
echo "== Weak scaling: ${WEAK_BASE}^3 rows per unit, $WEAK_ITERS fixed iterations =="
WREPS=(--tol 0 --max-iters "$WEAK_ITERS" --reps 3 --warmup 1 --quiet)
for t in $(pow2_upto "$NPROC"); do
    run weak openmp none 1 "$t" --nx "$WEAK_BASE" --ny "$WEAK_BASE" --nz $((WEAK_BASE * t)) "${WREPS[@]}"
done
for p in 1 2 4 8; do
    run weak mpi halo "$p" 1 --nx "$WEAK_BASE" --ny "$WEAK_BASE" --nz $((WEAK_BASE * p)) "${WREPS[@]}"
done

# ------------------------------------------------------------- convergence
echo "== Convergence history (sequential, grid 64 and ${GRID}) =="
for g in 64 "$GRID"; do
    "$BIN" --mode sequential --grid "$g" --tol "$TOL" --max-iters "$MAX_ITERS" --reps 1 --warmup 0 \
        --quiet --history "$RESULTS/convergence_g${g}.csv" > "$LOGS/convergence_g${g}.log" 2>&1 \
        && echo "OK    convergence_g${g}" || echo "FAIL  convergence_g${g}"
done

echo
echo "Runs: $ok_runs ok, $failed_runs failed, $skipped_runs skipped. CSV: results/benchmark.csv"
[ "$failed_runs" -eq 0 ]
