# Reduction Kernel

---

This document describes the parallel reduction implementations in `benchmarks/blas/benchmark_reduction.cu`.

## Table of Contents

- [Overview](#overview)
- [Technical Details](#technical-details)
- [Compilation Instructions](#compilation-instructions)
- [Running the Benchmark](#running-the-benchmark)
- [Performance Results](#performance-results)

## Overview

Parallel reduction computes the sum of all elements in a vector of size `N`. We provide a host reference and several optimized GPU kernels:

- **CPU**: Single-threaded host implementation.
- **Naive Reduce GPU**: Basic binary-tree reduction without optimization.
- **Interleaved Addressing Reduce GPU**: Reduces divergent memory accesses by interleaving.
- **Sequence Addressing Reduce GPU**: Performs contiguous block reductions.
- **Add During Load Reduce GPU**: Combines addition during load to improve memory throughput.
- **Unroll Last Warp Reduce GPU**: Unrolls final warp reduction for improved performance.
- **Completely Unrolled Warp Reduce GPU**: Fully unrolled warp-level reduction.
- **Multiple Elements Reduce GPU**: Processes multiple input elements per thread.

## Technical Details

- Kernel launch:

```cpp
// grid = dim3((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
// block = dim3(BLOCK_SIZE);
// numBlocks = (N + 2*BLOCK_SIZE - 1) / (2*BLOCK_SIZE);

// Naive binary-tree reduction
naiveReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(N, input, output);

// Interleaved addressing to improve coalescing
interleavedAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(N, input, output);

// Sequence addressing for contiguous block reduction
sequenceAddressingReduce<<<grid, block, BLOCK_SIZE * sizeof(int)>>>(N, input, output);

// Add-during-load to combine loads and adds
addDuringLoadReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(N, input, output);

// Unroll last warp to eliminate intra-warp sync
unrollLastWarpReduce<<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(N, input, output);

// Fully unroll warp-level reduction loops
completelyUnrollReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(N, input, output);

// Each thread reduces multiple elements before block-wide reduction
multipleElementsReduce<BLOCK_SIZE><<<numBlocks, block, BLOCK_SIZE * sizeof(int)>>>(input, output, N);
```

- Block size: 256 threads.
- Grid size: `ceil(N / (2*blockDim.x))` or variant-dependent.
- Final reduction across partial results on host.

## Compilation Instructions

Build the Reduction benchmark:

```bash
make build
```

## Running the Benchmark

Execute the benchmark binary:

```bash
make blas TARGET=reduce
```

This runs all reduction kernels and reports latency and bandwidth.

## Performance Results

Benchmarks were run with `N=4,194,304` elements and 256 threads per block.

| Kernel                              | Latency (ms) | Bandwidth (TB/s) | % of cuBLAS |
| ----------------------------------- | ------------ | ---------------- | ----------- |
| CPU                                 | 4.4047       | 0.00380893       | N/A         |
| Naive Reduce GPU                    | 0.05024      | 0.333941         | N/A         |
| Interleaved Addressing Reduce GPU   | 0.037216     | 0.450807         | N/A         |
| Sequence Addressing Reduce GPU      | 0.050432     | 0.33267          | N/A         |
| Add During Load Reduce GPU          | 0.02112      | 0.794376         | N/A         |
| Unroll Last Warp Reduce GPU         | 0.019776     | 0.848362         | N/A         |
| Completely Unrolled Warp Reduce GPU | 0.023456     | 0.715263         | N/A         |
| Multiple Elements Reduce GPU        | 0.01968      | 0.852501         | N/A         |
