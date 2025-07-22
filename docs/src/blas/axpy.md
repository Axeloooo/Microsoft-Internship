# AXPY Kernel

---

This document describes the AXPY (`Y = a*X + Y`) implementations in `benchmarks/blas/benchmark_axpy.cu`.

## Table of Contents

- [Overview](#overview)
- [Technical Details](#technical-details)
- [Compilation Instructions](#compilation-instructions)
- [Running the Benchmark](#running-the-benchmark)
- [Performance Results](#performance-results)

## Overview

AXPY computes the element-wise operation `Y[i] = a * X[i] + Y[i]` for vectors of size `N`. We provide two CUDA GPU kernels alongside a CPU reference:

- **CPU**: Single-threaded host implementation.
- **Naive GPU Kernel**: One thread per element without vectorization.
- **Vectorized GPU Kernel**: Processes multiple elements per thread using vector loads/writes.
- **cuBLAS**: Vendor-optimized implementation (for reference).

## Technical Details

- Kernel launch:

```cpp
// block = dim3(BLOCK_SIZE);
// grid  = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

// Naive AXPY kernel
axpy_naive<<<grid, block>>>(N, a, x, y);

// Vectorized AXPY (4 elements per thread)
axpy_vec<<<grid, block>>>(N/4, a, x, y);
```

- Data types: `float` precision.
- Block size: 256 threads per block.
- Grid size: `ceil(N / elements_per_thread)`.

## Compilation Instructions

Build the AXPY benchmark:

```bash
make build
```

## Running the Benchmark

Execute the benchmark binary:

```bash
make blas TARGET=axpy
```

This runs all kernels in sequence and prints latency and bandwidth.

## Performance Results

Benchmarks were run with `N=4,194,304` elements and 256 threads per block.

| Kernel                     | Latency (ms) | Bandwidth (TB/s) | % of cuBLAS |
| -------------------------- | ------------ | ---------------- | ----------- |
| CPU                        | 5.69472      | 0.0058922        | 0.48 %      |
| Naive AXPY GPU Kernel      | 0.028864     | 1.1625           | 94.00 %     |
| Vectorized AXPY GPU Kernel | 0.027264     | 1.23072          | 99.53 %     |
| cuBLAS                     | 0.027136     | 1.23653          | 100 %       |
