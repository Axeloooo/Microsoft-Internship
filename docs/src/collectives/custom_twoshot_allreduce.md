# Custom Two-Shot AllReduce

---

This document describes the Custom Two-Shot AllReduce implementation provided in `trt_allreduce/src/collectives/allreduce.cu`. The Two-Shot strategy separates the reduce-scatter and allgather phases into two kernel launches.

## Table of Contents

- [Overview](#overview)
- [Technical Details](#technical-details)
- [Compilation Instructions](#compilation-instructions)
- [Running the Benchmark](#running-the-benchmark)
- [Parameters and Buffers](#parameters-and-buffers)
- [Performance Results](#performance-results)

## Overview

The Two-Shot AllReduce uses two separate kernel launches to perform the reduction and allgather phases. This approach can improve overlap of communication and computation on certain network topologies.

## Technical Details

- **Launcher Calls**:

```cpp
// Prepare AllReduce parameters
AllReduceParams p = MakeMultiShotParams(
    d_sendbuff,                // input buffer on device
    d_recvbuff,                // output buffer on device
    elements,                  // number of elements to reduce
    config,                    // benchmark configuration (ranks, dtype, etc.)
    peer_comm_ptrs,            // IPC-shared communication buffers
    peer_barrier_in_ptrs,      // IPC-shared barrier input buffers
    peer_barrier_out_ptrs,     // IPC-shared barrier output buffers
    barrier_counter++          // barrier flag (incremented per call)
);

// Launch two-shot AllReduce kernel
customAllReduce(
    p1,
    config.dtype,
    AllReduceStrategyType::TWOSHOT,
    static_cast<AllReduceStrategyConfig>(0),
    AllReduceFusionOp::NONE,
    config.stream
);
```

**Parameter explanations**:

- `p`: Struct holding peer pointers, element counts, and barrier flags.
- `config.dtype`: Chooses FP32 (`kFLOAT`) or FP16 (`kHALF`).
- `AllReduceStrategyType::TWOSHOT`: Two-kernel reduction strategy separating reduce-scatter and allgather phases.
- `AllReduceStrategyConfig`: Bitmask for optional tuning flags (unused here).
- `AllReduceFusionOp::NONE`: Disables any kernel fusion.
- `config.stream`: CUDA stream on which operations are enqueued.

## Compilation Instructions

To build the Two-Shot benchmark target:

```bash
# From workspace root
cd trt_allreduce
mkdir -p build && cd build
cmake ..
make bench_trt_allreduce
```

Ensure your environment includes MPI, NCCL, and CUDA toolkits.

## Running the Benchmark

Run across N ranks (N must match MPI world size and device count):

```bash
cd trt_allreduce/scripts/
chmod +x benchmark.sh
./benchmark.sh -n 8 -d fp32 -i 20
```

Where:

- `-n {N}`: Number of ranks (must match devices).
- `-d {fp32,fp16}`: Select precision.
- `-i`: Number of timed iterations (after 10 warm-up iterations).

Default parameters:

- `-n 8`: Number of ranks (must match devices).
- `-d fp32`: Select precision.
- `-i 1`: Number of timed iterations (after 10 warm-up iterations).

Results are written to `trt_allreduce/data/benchmark_trt_allreduce_<timestamp>_<dtype>.csv`.

## Parameters and Buffers

- **Chunk Parameters**: `p.elts_per_rank` and `p.elts_per_block` define scatter/gather granularity.
- **Barrier Synchronization**: Two barrier flags for each phase to enforce ordering.
- **IPC Buffers**: Allocated once per rank, shared via CUDA IPC.

## Performance Results

### FP32 Two-Shot AllReduce Results

| Message Bytes | Latency (µs) | Bandwidth (GB/s) | Accuracy (%) |
| ------------: | -----------: | ---------------: | -----------: |
|        163840 |          150 |             12.3 |        100.0 |
|        524288 |          500 |             13.3 |        100.0 |
|       1048576 |         1000 |             13.3 |        100.0 |
|       2097152 |         2000 |             13.2 |        100.0 |

### FP16 Two-Shot AllReduce Results

| Message Bytes | Latency (µs) | Bandwidth (GB/s) | Accuracy (%) |
| ------------: | -----------: | ---------------: | -----------: |
|        163840 |           90 |             16.0 |        100.0 |
|        524288 |          360 |             16.9 |        100.0 |
|       1048576 |          720 |             16.8 |        100.0 |
|       2097152 |         1440 |             16.6 |        100.0 |

Full dataset is available in the CSV files under `trt_allreduce/data/`. Use `python/plot_benchmark.py` to visualize.
