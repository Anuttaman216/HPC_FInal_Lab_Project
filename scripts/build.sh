#!/usr/bin/env bash
# build.sh - configure and build with CMake (Release by default).
#   scripts/build.sh            # incremental Release build in ./build
#   scripts/build.sh --clean    # remove ./build first
#   BUILD_TYPE=Debug scripts/build.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"

if [ "${1:-}" = "--clean" ]; then
    echo "Removing $BUILD_DIR"
    rm -rf "$BUILD_DIR"
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" ${CMAKE_EXTRA_ARGS:-}
cmake --build "$BUILD_DIR" -j "$JOBS"
echo "Built: $BUILD_DIR/hpcg_benchmark"
