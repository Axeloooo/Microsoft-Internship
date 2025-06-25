# Microsoft Internship Project

[![Continuous Integration](https://github.com/Axeloooo/Microsoft-Internship/actions/workflows/continuous_integration.yml/badge.svg)](https://github.com/Axeloooo/Microsoft-Internship/actions/workflows/continuous_integration.yml)

## Documentation

- [Specifications](#specifications)
- [Benchmark](#benchmarks)
- [Vector Addition](#vector-addition)
- [Parallel Reduction](#parallel-reduction)
- [Dot Product](#dot-product)
- [AXPY](#axpy)
- [License](#license)
- [Contributors](#contributors)

## Specifications

This project contains CUDA implementations of several vector operations, including vector addition, parallel reduction, dot product, and AXPY (A \* X + Y). The implementations are designed to be efficient and leverage the capabilities of NVIDIA GPUs.

All benchmarks were run on a system with the following specifications:

- CPU: Intel Xeon Platinum 8480C
- GPU: NVIDIA H200
- Threads per block: 256
- Elements per vector: 4,194,304

## Benchmark

- AXPY:

| Kernel                     | Latency (ms) | Bandwidth (TB/s) | % of cuBLAS |
| -------------------------- | ------------ | ---------------- | ----------- |
| CPU                        | 5.69472      | 0.0058922        | 0.48 %      |
| Naive AXPY GPU Kernel      | 0.028864     | 1.1625           | 94.00 %     |
| Vectorized AXPY GPU Kernel | 0.027264     | 1.23072          | 99.53 %     |
| cuBLAS                     | 0.027136     | 1.23653          | 100 %       |

- Vector Addition:

| Kernel     | Latency (ms) | Bandwidth (TB/s) | % of cuBLAS |
| ---------- | ------------ | ---------------- | ----------- |
| CPU        | 14.2268      | 0.0035378        | N/A         |
| GPU Kernel | 0.026176     | 1.92282          | N/A         |

- Parallel Reduction:

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

## Vector Addition

1. Run the CUDA code.

```bash
make add
```

2. Clean up the build files.

```bash
make clean
```

## Parallel Reduction

1. Run the CUDA code.

```bash
make reduce
```

2. Clean up the build files.

```bash
make clean
```

## Dot Product

1. Run the CUDA code.

```bash
make dot
```

2. Clean up the build files.

```bash
make clean
```

## AXPY

1. Run the CUDA code.

```bash
make axpy
```

2. Clean up the build files.

```bash
make clean
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributors

- [Axel Sanchez](https://github.com/Axeloooo)

## Extra

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
