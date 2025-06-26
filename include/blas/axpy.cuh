/**
 * @file axpy.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CUDA implementation of the A·X Plus Y (SAXPY) operation.
 * @version 0.1
 * @date 2025-06-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

/**
 * @brief SAXPY (Single-Precision A·X Plus Y) operation.
 *
 * @param alpha Scalar multiplier for vector x.
 * @param x Pointer to the input vector x.
 * @param y Pointer to the output vector y, which will be updated with the result.
 * @param n Number of elements in the vectors.
 * @return __global__
 */
__global__ void naiveSaxpy(const float alpha, const float *x, float *y, const int n);

/**
 * @brief Vectorized SAXPY operation using float4 for better performance.
 *
 * @param alpha Scalar multiplier for vector x.
 * @param x4 Pointer to the input vector x, which is expected to be aligned to float4.
 * @param y4 Pointer to the output vector y, which will be updated with the result.
 * @param nv Number of float4 elements in the vectors (i.e., n/4 if n is the total number of elements).
 * @return __global__
 */
__global__ void vectorizedSaxpy(const float alpha, const float4 *x4, float4 *y4, const int nv);