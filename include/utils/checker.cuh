/**
 * @file checker.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Header file for CUDA and cuBLAS error checking macros.
 * @version 0.1
 * @date 2025-06-13
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifndef CHECKER_CUH
#define CHECKER_CUH

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

/**
 * @brief Macro to check CUDA function calls for errors.
 *
 */
#define CUDA_CHECK(call)                                                                                                  \
    do {                                                                                                                  \
        cudaError_t err = (call);                                                                                         \
        if (err != cudaSuccess) {                                                                                         \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                                                                      \
        }                                                                                                                 \
    } while (0)

/**
 * @brief Macro to check MPI function calls for errors.
 *
 */
#define MPI_CHECK(cmd)                                                                               \
    do {                                                                                             \
        int err = (cmd);                                                                             \
        if (err != MPI_SUCCESS) {                                                                    \
            std::cerr << "MPI error in " << __FILE__ << ":" << __LINE__ << ": " << err << std::endl; \
            std::exit(EXIT_FAILURE);                                                                 \
        }                                                                                            \
    } while (0)

/**
 * @brief Macro to check cuBLAS function calls for errors.
 *
 */
#define CHECK_CUBLAS(call)                                                                                            \
    {                                                                                                                 \
        cublasStatus_t status = (call);                                                                               \
        if (status != CUBLAS_STATUS_SUCCESS) {                                                                        \
            std::cerr << "cuBLAS error in " << __FILE__ << ":" << __LINE__ << ": error code " << status << std::endl; \
            std::exit(EXIT_FAILURE);                                                                                  \
        }                                                                                                             \
    }

/**
 * @brief Macro to check NCCL function calls for errors.
 *
 */
#define NCCL_CHECK(cmd)                                                                                                               \
    do {                                                                                                                              \
        ncclResult_t res = cmd;                                                                                                       \
        if (res != ncclSuccess) {                                                                                                     \
            std::cerr << "Failed, NCCL error " << __FILE__ << ":" << __LINE__ << " '" << ncclGetErrorString(res) << "'" << std::endl; \
            std::exit(EXIT_FAILURE);                                                                                                  \
        }                                                                                                                             \
    } while (0)

#endif