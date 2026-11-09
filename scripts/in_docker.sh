#!/usr/bin/env bash
# in_docker.sh - run a command inside the reproducible Ubuntu container
# (for hosts without a native Linux toolchain, e.g. Windows + Docker Desktop).
#   scripts/in_docker.sh bash scripts/run_all.sh
#   scripts/in_docker.sh            # interactive shell
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${HPCG_IMAGE:-hpcg-analyzer}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    docker build -t "$IMAGE" -f "$ROOT/docker/Dockerfile" "$ROOT/docker"
fi
HOST_DIR="$ROOT"
# Git Bash on Windows: give Docker a Windows path and stop MSYS path mangling.
if command -v cygpath >/dev/null 2>&1; then HOST_DIR="$(cygpath -w "$ROOT")"; export MSYS_NO_PATHCONV=1; fi
TTY=""; [ -t 0 ] && [ -t 1 ] && TTY="-it"
if [ $# -eq 0 ]; then set -- bash; fi
exec docker run --rm $TTY --shm-size=1g -v "$HOST_DIR:/work" -w /work "$IMAGE" "$@"
