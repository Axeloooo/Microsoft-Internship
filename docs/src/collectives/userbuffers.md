# Userbuffers AllReduce

---

This document describes the User-Buffers AllReduce implementation provided in `trt_allreduce/src/collectives/allreduce.cu`. The user-buffers strategy uses a pre-allocated shared buffer in device memory to stage data and perform reductions in-place across all ranks.

## Table of Contents

- [Overview](#overview)
- [Technical Details](#technical-details)
- [Compilation Instructions](#compilation-instructions)
- [Running the Benchmark](#running-the-benchmark)
- [Parameters and Buffers](#parameters-and-buffers)
- [Performance Results](#performance-results)

## Overview

The User-Buffers implementation leverages a continuous IPC-allocated buffer (`UBBuffer`) per rank to perform in-place reductions using `allreduce2_userbuff_inplace_launcher`. This strategy is currently supported for FP16 precision only.

## Technical Details

- **Technical Workflow**:

```cpp
// Initialize and allocate UserBuffer (FP16 only)
ub_initialize(config.world_size);
UBBuffer userbuff = ub_allocate(max_bytes);

// Stage input data into the UserBuffer
CUDA_CHECK(cudaMemcpyAsync(
    userbuff.addr,                     // destination: user buffer address
    d_sendbuff,                        // source: device input buffer
    elements * config.element_size,    // total bytes
    cudaMemcpyDeviceToDevice,
    config.stream                      // stream for async ops
));

// Launch in-place UserBuffers AllReduce
allreduce2_userbuff_inplace_launcher(
    userbuff.handle,                   // IPC handle for the buffer
    0,                                 // stage flag (reserved)
    elements,                          // number of elements to reduce
    config.dtype,                      // data type (kHALF only)
    ub_comm(),                         // communicator context
    config.stream                      // CUDA stream
);

// Copy reduced result back to receive buffer
CUDA_CHECK(cudaMemcpyAsync(
    d_recvbuff,                        // destination: device output buffer
    userbuff.addr,                     // source: user buffer address
    elements * config.element_size,    // total bytes
    cudaMemcpyDeviceToDevice,
    config.stream
));
CUDA_CHECK(cudaStreamSynchronize(config.stream));
```

**Parameter explanations**:

- `ub_initialize(world_size)`: Sets up IPC handle pool for all ranks.
- `ub_allocate(max_bytes)`: Allocates a `UBBuffer` of `max_bytes` on device.
- `userbuff.addr`: Device pointer to buffer memory used by the kernel.
- `userbuff.handle`: Opaque handle passed to kernel for buffer access.
- `elements`: Number of data elements to reduce.
- `config.dtype`: Data type (`kHALF`) determining element size.
- `ub_comm()`: Returns `communicator*` for UserBuffers launches.
- `config.stream`: CUDA stream for asynchronous copy and kernel operations.

## Compilation Instructions

To build the User-Buffers benchmark:

```bash
cd trt_allreduce/scripts/
chmod +x build.sh
./build.sh
```

Ensure `ub_allocator.cu` and related headers are compiled into the library.

## Running the Benchmark

Run across N ranks (N must match MPI world size and device count):

**Note**: User-Buffers currently supports FP16 precision only.

```bash
cd trt_allreduce/scripts/
chmod +x benchmark.sh
./benchmark.sh -n 8 -d fp16 -i 20
```

Where:

- `-n {N}`: Number of ranks (must match devices).
- `-d {fp16}`: Select precision.
- `-i`: Number of timed iterations (after 10 warm-up iterations).

Default parameters:

- `-n 8`: Number of ranks (must match devices).
- `-d fp32`: Select precision.
- `-i 1`: Number of timed iterations (after 10 warm-up iterations).

Results are written to `trt_allreduce/data/benchmark_trt_allreduce_<timestamp>_<dtype>.csv`.

## Parameters and Buffers

- **UBBuffer**:
  - `addr`: Base device pointer to the IPC buffer.
  - `size`: Total buffer size in bytes.
  - `invalid()`: Checks handle validity.
- **Peer Communicator**: Context from `ub_comm()` for collective launch.
- **Copy-Back**: After reductions, data is copied from `userbuff.addr` to `d_recvbuff` for validation.

## Performance Results

### FP32 User-Buffers AllReduce Results (Not Supported)

| Message Bytes | Latency (µs) | Bandwidth (GB/s) | Accuracy (%) |
| ------------: | -----------: | ---------------: | -----------: |
|        163840 |          N/A |              N/A |          N/A |
|        524288 |          N/A |              N/A |          N/A |
|       1048576 |          N/A |              N/A |          N/A |
|       2097152 |          N/A |              N/A |          N/A |

### FP16 User-Buffers AllReduce Results

| Message Bytes | Latency (µs) | Bandwidth (GB/s) | Accuracy (%) |
| ------------: | -----------: | ---------------: | -----------: |
|        163840 |           80 |             15.0 |        100.0 |
|        524288 |          320 |             16.5 |        100.0 |
|       1048576 |          640 |             16.4 |        100.0 |
|       2097152 |         1280 |             16.2 |        100.0 |

For full tables and CSVs, see files under `trt_allreduce/data/`. Use `python/plot_benchmark.py` to compare One-Shot, Two-Shot, and User-Buffers strategies visually.
