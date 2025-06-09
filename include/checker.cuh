#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CHECK_CUDA(call)                                                         \
    {                                                                            \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl;                   \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    }

#define CHECK_CUBLAS(call)                                                                    \
    {                                                                                         \
        cublasStatus_t status = (call);                                                       \
        if (status != CUBLAS_STATUS_SUCCESS)                                                  \
        {                                                                                     \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": error code " \
                      << status << std::endl;                                                 \
            std::exit(EXIT_FAILURE);                                                          \
        }                                                                                     \
    }

#define NCCLCHECK(cmd)                                           \
    do                                                           \
    {                                                            \
        ncclResult_t res = cmd;                                  \
        if (res != ncclSuccess)                                  \
        {                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n",            \
                   __FILE__, __LINE__, ncclGetErrorString(res)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)
