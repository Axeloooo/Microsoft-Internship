# Microsoft Internship Project

## Reduction Kernel

- Compile

```shell
nvcc -o build/reduction src/reduction.cu
```

- Run

```shell
./build/reduction
```

- Profile

```shell
nsys profile -o logs/reduction ./build/reduction
```
