/**
 * @file axpy.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief SAXPY (Single-Precision A·X Plus Y) operation implementation using CUDA.
 * @version 0.1
 * @date 2025-06-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

#include "blas/axpy.cuh"

/**
 * @brief Naive SAXPY kernel implementation
 *
 * @param alpha Scalar multiplier
 * @param x Vector x
 * @param y Vector y
 * @param n Number of elements in vectors x and y
 * @return __global__
 */
__global__ void naiveSaxpy(const float alpha, const float *x, float *y, const int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = tid; i < n; i += stride) {
        y[i] = alpha * x[i] + y[i];
    }
}

/**
 * @brief Vectorized SAXPY operation using float4 for better performance.
 *
 * @param alpha Scalar multiplier for vector x.
 * @param x4 Pointer to the input vector x, which is expected to be aligned to float4.
 * @param y4 Pointer to the output vector y, which will be updated with the result.
 * @param nv Number of float4 elements in the vectors (i.e., n/4 if n is the total number of elements).
 * @return __global__
 */
__global__ void vectorizedSaxpy(const float alpha, const float4 *x4, float4 *y4, const int nv) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i = tid; i < nv; i += stride) {
        float4 xv = x4[i];
        float4 yv = y4[i];

        yv.x = alpha * xv.x + yv.x;
        yv.y = alpha * xv.y + yv.y;
        yv.z = alpha * xv.z + yv.z;
        yv.w = alpha * xv.w + yv.w;

        y4[i] = yv;
    }
}