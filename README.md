# Microsoft Internship Project

![Publish Documentation](https://github.com/Axeloooo/Microsoft-Internship/actions/workflows/documentation.yml/badge.svg)
![Continuous Integration](https://github.com/Axeloooo/Microsoft-Internship/actions/workflows/continuous_integration.yml/badge.svg)

## Overview

This repository hosts CUDA benchmarks for BLAS operations and custom collective communication algorithms developed during the Microsoft internship.

## Documentation

Full documentation is available via mdBook under the `docs/src` folder:

- **Getting Started**: [docs/src/README.md](docs/src/README.md)
- **BLAS Kernels**:
  - [AXPY](docs/src/blas/axpy.md)
  - [Dot Product](docs/src/blas/dot.md)
  - [Reduction](docs/src/blas/reduction.md)
  - [GEMM](docs/src/blas/gemm.md)
- **Collective Kernels**:
  - [One-Shot AllReduce](docs/src/collectives/custom_oneshot_allreduce.md)
  - [Two-Shot AllReduce](docs/src/collectives/custom_twoshot_allreduce.md)
  - [User-Buffers AllReduce](docs/src/collectives/userbuffers.md)

## System Specifications

Benchmarks were executed on:

- CPU: Intel Xeon Platinum 8480C
- GPU: NVIDIA H200
- Threads per block: 256 (BLAS)
- MPI ranks per node: variable (Collectives)

## Building and Running

### BLAS Benchmarks

Compile and run BLAS kernels via Makefile:

```bash
# Build all BLAS targets
make blas TARGET=axpy      # Vector AXPY (Y = aX + Y)
make blas TARGET=reduce    # Parallel reduction
make blas TARGET=dot       # Dot product
make blas TARGET=gemm      # Matrix multiplication (GEMM)
```

### Collective Benchmarks

Under `trt_allreduce/scripts`:

```bash
cd trt_allreduce/scripts
chmod +x build.sh benchmark.sh
./build.sh                  # Build collective benchmarks
./benchmark.sh -n 4 -d fp32  # Run with 4 MPI ranks, FP32 precision
./benchmark.sh -n 4 -d fp16  # Run with 4 MPI ranks, FP16 precision
```

## Development Environment

Use Docker Compose or standalone Docker:

```bash
# Docker Compose
docker compose up -d

# Standalone Docker
docker run --gpus all -it \
  --user $(id -u):$(id -g) \
  --name microsoft-internship-axel \
  --network host \
  --cap-add=SYS_PTRACE \
  --device=/dev/infiniband \
  --device=/dev/dri \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  --ipc=host \
  --shm-size=32g \
  nvcr.io/nvidia/pytorch:24.12-py3
```

## License

This project is licensed under the Apache License 2.0 - see [LICENSE](LICENSE).

## Contributors

- Axel Omar Sanchez Peralta ([Axeloooo](https://github.com/Axeloooo))
