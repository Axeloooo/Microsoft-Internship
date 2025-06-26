# Microsoft Internship Project

[![Continuous Integration](https://github.com/Axeloooo/Microsoft-Internship/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/Axeloooo/Microsoft-Internship/actions/workflows/continuous_integration.yml)

---

## Documentation

- [Specifications](#specifications)
- [Benchmark](#benchmarks)
- [BLAS Operations](#blas-operations)
- [Collective Operations](#collective-operations)
- [License](#license)
- [Contributors](#contributors)
- [Extra](#extra)

--

## Specifications

This project contains CUDA implementations of several vector operations, including vector addition, parallel reduction, dot product, and AXPY (A \* X + Y). The implementations are designed to be efficient and leverage the capabilities of NVIDIA GPUs.

All benchmarks were run on a system with the following specifications:

- CPU: Intel Xeon Platinum 8480C
- GPU: NVIDIA H200
- Threads per block: 256
- Elements per vector: 4,194,304

---

## Benchmark

### BLAS Operations

- AXPY:

| Kernel                     | Latency (ms) | Bandwidth (TB/s) | % of cuBLAS |
| -------------------------- | ------------ | ---------------- | ----------- |
| CPU                        | 5.69472      | 0.0058922        | 0.48 %      |
| Naive AXPY GPU Kernel      | 0.028864     | 1.1625           | 94.00 %     |
| Vectorized AXPY GPU Kernel | 0.027264     | 1.23072          | 99.53 %     |
| cuBLAS                     | 0.027136     | 1.23653          | 100 %       |

- Reduction:

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

- Dot Product:

| Kernel           | Latency (ms) | Bandwidth (TB/s) | % of cuBLAS |
| ---------------- | ------------ | ---------------- | ----------- |
| CPU              | 10.3247      | 0.00324993       | 0.24 %      |
| Naive Kernel     | 7.37251      | 0.00455129       | 0.34 %      |
| Optimized Kernel | 0.026528     | 1.26487          | 94.57 %     |
| cuBLAS           | 0.025088     | 1.33747          | 100 %       |

- GEMM:

| Kernel | Latency (ms) | GFLOPs | % of cuBLAS |
| ------ | ------------ | ------ | ----------- |

### Collective Operations

- ReduceScatter:

| Kernel | Latency (ms) | Bandwidth (TB/s) | % of NCCL |

- AllGather:

| Kernel | Latency (ms) | Bandwidth (TB/s) | % of NCCL |

- AllReduce:

| Kernel | Latency (ms) | Bandwidth (TB/s) | % of NCCL |

- Broadcast:

| Kernel | Latency (ms) | Bandwidth (TB/s) | % of NCCL |

- Reduce:

| Kernel | Latency (ms) | Bandwidth (TB/s) | % of NCCL |

---

## BLAS Operations

This project implements several BLAS (Basic Linear Algebra Subprograms) operations using CUDA. The following sections detail how to run each operation.

### Reduction

- Run the benchmark.

```bash
make blas TARGET=reduce
```

### Dot Product

- Run the benchmark.

```bash
make blas TARGET=dot
```

### AXPY

- Run the benchmark.

```bash
make blas TARGET=axpy
```

### GEMM

- Run the benchmark.

```bash
make blas TARGET=gemm
```

---

## Collective Operations

This project also implements collective operations using MCCL (Micorosft Collective Communication Library) and NCCL (NVIDIA Collective Communications Library). The following sections detail how to run each operation.

### ReduceScatter

- Run the benchmark.

```bash
make collective TARGET=reducescatter
```

### AllGather

- Run the benchmark.

```bash
make collective TARGET=allgather
```

### AllReduce

- Run the benchmark.

```bash
make collective TARGET=allreduce
```

### Broadcast

- Run the benchmark.

```bash
make collective TARGET=broadcast
```

### Reduce

- Run the benchmark.

```bash
make collective TARGET=reduce
```

---

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## Contributors

- [Axel Sanchez](https://github.com/Axeloooo)

---

## Extra

- Development environment setup using Docker Compose:

```bash
docker compose up -d
```

- Development environment setup using Docker:

```bash
docker run --gpus all -it \
  --user $(id -u):$(id -g) \
  --name microsoft-internship-axel \
  --network host \
  --cap-add=SYS_PTRACE \
  --device=/dev/infiniband \
  --device=/dev/dri \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v /home/t-axels/Microsoft-Internship:/workspace \
  --ipc=host \
  --shm-size=32g \
  nvcr.io/nvidia/pytorch:24.12-py3
```
