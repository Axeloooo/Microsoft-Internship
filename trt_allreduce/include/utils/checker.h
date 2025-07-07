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

#include <cstdlib>
#include <iostream>

namespace trt_allreduce::utils {

#ifndef CUDA_CHECK
#define CUDA_CHECK(call)                                                                                                  \
    do {                                                                                                                  \
        cudaError_t err = (call);                                                                                         \
        if (err != cudaSuccess) {                                                                                         \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(err) << std::endl; \
            std::exit(EXIT_FAILURE);                                                                                      \
        }                                                                                                                 \
    } while (0)
#endif

#ifndef NCCL_CHECK
#define NCCL_CHECK(call)                                                                                                              \
    do {                                                                                                                              \
        ncclResult_t res = (call);                                                                                                    \
        if (res != ncclSuccess) {                                                                                                     \
            std::cerr << "Failed, NCCL error " << __FILE__ << ":" << __LINE__ << " '" << ncclGetErrorString(res) << "'" << std::endl; \
            std::exit(EXIT_FAILURE);                                                                                                  \
        }                                                                                                                             \
    } while (0)
#endif

#ifndef MPI_CHECK
#define MPI_CHECK(call)                                                                              \
    do {                                                                                             \
        int err = (call);                                                                            \
        if (err != MPI_SUCCESS) {                                                                    \
            std::cerr << "MPI error in " << __FILE__ << ":" << __LINE__ << ": " << err << std::endl; \
            std::exit(EXIT_FAILURE);                                                                 \
        }                                                                                            \
    } while (0)
#endif

}  // namespace trt_allreduce::utils
