/**
 * @file cublasVersionCheck.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Version check macros for cuBLAS to determine which header to include.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

// We don't want to include cublas_api.h. It contains the CUBLAS_VER_* macro
// definition which is not sufficient to determine if we include cublas.h,
// cublas_v2.h or cublasLt.h.

#define TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH) (MAJOR * 10000 + MINOR * 100 + PATCH)
#define TLLM_CUBLAS_VER_LE(MAJOR, MINOR, PATCH)                                    \
    TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH) \
    <= TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define TLLM_CUBLAS_VER_LT(MAJOR, MINOR, PATCH) TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH) < TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define TLLM_CUBLAS_VER_GE(MAJOR, MINOR, PATCH)                                    \
    TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH) \
    >= TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
#define TLLM_CUBLAS_VER_GT(MAJOR, MINOR, PATCH) TLLM_CUBLAS_VERSION_CALC(CUBLAS_VER_MAJOR, CUBLAS_VER_MINOR, CUBLAS_VER_PATCH) > TLLM_CUBLAS_VERSION_CALC(MAJOR, MINOR, PATCH)
