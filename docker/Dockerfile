# Reproducible Linux build/run environment for the HPCG-inspired CG benchmark.
#   docker build -t hpcg-analyzer -f docker/Dockerfile .
#   docker run --rm -it -v "$PWD":/work -w /work hpcg-analyzer bash
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    OMPI_MCA_rmaps_base_oversubscribe=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
        openmpi-bin libopenmpi-dev \
        python3 python3-numpy python3-pandas python3-matplotlib \
        linux-tools-common procps util-linux \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work
CMD ["bash"]
