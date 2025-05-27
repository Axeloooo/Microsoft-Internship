#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                  \
    {                                                                     \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess)                                           \
        {                                                                 \
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    }

#define CHECK_CUBLAS(call)                                                              \
    {                                                                                   \
        cublasStatus_t status = call;                                                   \
        if (status != CUBLAS_STATUS_SUCCESS)                                            \
        {                                                                               \
            fprintf(stderr, "cuBLAS error in %s:%d: %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    }
