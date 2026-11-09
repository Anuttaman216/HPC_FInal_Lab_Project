#!/usr/bin/env bash
# setup.sh - check the build/run dependencies and print install hints.
# Exit status: 0 if everything required is present, 1 otherwise.
set -u

ok=1
have() { command -v "$1" >/dev/null 2>&1; }
report() {  # name, found(0/1), hint
    if [ "$2" -eq 1 ]; then printf "  [ok]      %s\n" "$1"; else printf "  [missing] %s  -> %s\n" "$1" "$3"; fi
}

echo "== Required =="
for tool in g++ cmake make; do
    if have "$tool"; then report "$tool ($($tool --version | head -n1))" 1 ""
    else report "$tool" 0 "sudo apt install build-essential cmake"; ok=0; fi
done

echo "== OpenMP =="
tmp=$(mktemp -d)
printf '#include <omp.h>\nint main(){return omp_get_max_threads()>0?0:1;}\n' > "$tmp/omp.cpp"
if have g++ && g++ -fopenmp "$tmp/omp.cpp" -o "$tmp/omp" 2>/dev/null && "$tmp/omp"; then
    report "OpenMP (g++ -fopenmp)" 1 ""
else
    report "OpenMP" 0 "install a GCC with libgomp (sudo apt install build-essential)"; ok=0
fi
rm -rf "$tmp"

echo "== MPI (optional but needed for mpi/hybrid modes) =="
if have mpicxx && have mpirun; then
    report "MPI ($(mpirun --version 2>/dev/null | head -n1))" 1 ""
else
    report "MPI" 0 "sudo apt install openmpi-bin libopenmpi-dev"
fi

echo "== Python analysis =="
if have python3; then
    for m in numpy pandas matplotlib; do
        if python3 -c "import $m" 2>/dev/null; then report "python3 $m" 1 ""
        else report "python3 $m" 0 "pip install $m  (or sudo apt install python3-$m)"; ok=0; fi
    done
else
    report "python3" 0 "sudo apt install python3"; ok=0
fi

echo "== Optional =="
if have perf && perf stat -e task-clock true >/dev/null 2>&1; then
    report "linux perf (usable)" 1 ""
else
    report "linux perf" 0 "optional; needs linux-tools-\$(uname -r) and perf_event access (not available in Docker/WSL by default)"
fi
if have lscpu; then report "lscpu" 1 ""; else report "lscpu" 0 "sudo apt install util-linux"; fi

echo
if [ "$ok" -eq 1 ]; then echo "All required dependencies found."; else echo "Some required dependencies are missing (see hints above)."; fi
echo "No Linux toolchain? Use the Docker image:  docker build -t hpcg-analyzer -f docker/Dockerfile ."
[ "$ok" -eq 1 ]
