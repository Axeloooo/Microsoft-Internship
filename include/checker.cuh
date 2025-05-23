#pragma once

#include <stdlib.h>
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
