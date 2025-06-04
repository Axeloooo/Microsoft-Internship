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

| Kernel       | Time (ms) | Bandwith (TB/s) |
| ------------ | --------- | --------------- |
| CPU          | 5.85363   | 0.00286612      |
| Naive Kernel | 0.026656  | 0.629397        |
| cuBLAS       | 0.025312  | 0.662817        |

- Vector Addition:

| Kernel       | Time (ms) | Bandwith (TB/s) |
| ------------ | --------- | --------------- |
| CPU          | 15.2313   | 0.0011015       |
| Naive Kernel | 0.023456  | 0.715263        |

- Parallel Reduction:

| Kernel                              | Time (ms) | Bandwith (TB/s) |
| ----------------------------------- | --------- | --------------- |
| CPU                                 | 4.43878   | 0.00377969      |
| Naive Reduce GPU                    | 0.050048  | 0.335222        |
| Interleaved Addressing Reduce GPU   | 0.035648  | 0.470636        |
| Sequence Addressing Reduce GPU      | 0.050016  | 0.335437        |
| Add During Load Reduce GPU          | 0.019872  | 0.844264        |
| Unroll Last Warp Reduce GPU         | 0.018656  | 0.899293        |
| Completely Unrolled Warp Reduce GPU | 0.01792   | 0.936229        |
| Multiple Elements Reduce GPU        | 0.01888   | 0.888624        |

- Dot Product:

| Kernel           | Time (ms) | Bandwith (TB/s) |
| ---------------- | --------- | --------------- |
| CPU              | 9.43277   | 0.00177861      |
| Naive Kernel     | 7.37485   | 0.00227492      |
| Optimized Kernel | 0.025856  | 0.648871        |
| cuBLAS           | 0.025664  | 0.653726        |

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
