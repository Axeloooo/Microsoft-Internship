# GEMM Kernel

---

This document describes the GEMM (`C = α*AB + βC`) implementations in `benchmarks/blas/benchmark_gemm.cu`.

## Table of Contents

- [Overview](#overview)
- [Technical Details](#technical-details)
- [Compilation Instructions](#compilation-instructions)
- [Running the Benchmark](#running-the-benchmark)
- [Performance Results](#performance-results)

## Overview

General matrix-matrix multiplication multiplies two dense matrices A (M x K) and B (K x N) to produce C (M x N). We provide:

- **CPU**: Host reference implementation.
- **Tiled GPU Kernel**: Partition matrices into tiles for shared memory reuse.
- **cuBLAS**: Vendor-optimized routine.

## Technical Details

- Kernel launch:

```cpp
// blockDim = dim3(16, 16);        // 16 threads per block in x and y
// gridDim  = dim3((N + 15) / 16, (M + 15) / 16);  // one block per tile

gemm_tiled<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

- Shared memory: Blocks load tiles of A and B into shared memory.
- Loop unrolling and register blocking.
- Data type: `float`.

## Compilation Instructions

Build the GEMM benchmark:

```bash
make build
```

## Running the Benchmark

Execute the benchmark binary:

```bash
make blas TARGET=gemm
```

This runs the tiled GPU kernel and cuBLAS for comparison.

## Performance Results

Typical benchmarks were run with sizes M=N=K=2048.

| Kernel | Latency (ms) | GFLOPs | % of cuBLAS |
| ------ | ------------ | ------ | ----------- |
