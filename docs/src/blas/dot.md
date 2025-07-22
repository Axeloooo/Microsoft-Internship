# Dot Product Kernel

---

This document describes the dot product implementations in `benchmarks/blas/benchmark_dot.cu`.

## Table of Contents

- [Overview](#overview)
- [Technical Details](#technical-details)
- [Compilation Instructions](#compilation-instructions)
- [Running the Benchmark](#running-the-benchmark)
- [Performance Results](#performance-results)

## Overview

Dot product computes the scalar result `sum = Σ_i X[i] * Y[i]` for vectors of size `N`. Implementations include:

- **CPU**: Host reference implementation.
- **Naive Kernel**: Basic per-element multiply and accumulation.
- **Optimized Kernel**: Uses warp-level primitives and shared memory to reduce accumulation overhead.
- **cuBLAS**: Vendor-optimized routine for reference.

## Technical Details

- Kernel launch:

```cpp
// block     = dim3(BLOCK_SIZE);
// grid      = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
// optimGrid = dim3((N + (2*BLOCK_SIZE) - 1) / (2*BLOCK_SIZE));

// Naive dot product kernel
naiveSdot<<<grid, block>>>(d_a, d_b, d_c_gpu, N);

// Optimized dot product kernel with shared memory
optimizedSdot<BLOCK_SIZE><<<optimGrid, block, BLOCK_SIZE * sizeof(float)>>>(d_a, d_b, d_c_gpu, N);
```

- Reduction across threads performed using parallel tree or warp shuffles.
- Block size: 256 threads per block.

## Compilation Instructions

Build the Dot Product benchmark:

```bash
make build
```

## Running the Benchmark

Execute the benchmark binary:

```bash
make blas TARGET=dot
```

This runs all dot product kernels and prints latency and bandwidth.

## Performance Results

Benchmarks were run with `N=4,194,304` elements and 256 threads per block.

| Kernel           | Latency (ms) | Bandwidth (TB/s) | % of cuBLAS |
| ---------------- | ------------ | ---------------- | ----------- |
| CPU              | 10.3247      | 0.00324993       | 0.24 %      |
| Naive Kernel     | 7.37251      | 0.00455129       | 0.34 %      |
| Optimized Kernel | 0.026528     | 1.26487          | 94.57 %     |
| cuBLAS           | 0.025088     | 1.33747          | 100 %       |
