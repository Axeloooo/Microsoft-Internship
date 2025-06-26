/**
 * @file dot.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CUDA implementation of the dot product of two vectors.
 * @version 0.1
 * @date 2025-06-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

/**
 * @brief Naive implementation of the dot product of two vectors
 *
 * @param a Vector a
 * @param b Vector b
 * @param c Result vector
 * @param n Size of the vectors
 * @return __global__
 */
__global__ void naiveSdot(const float *a, const float *b, float *c, const int n);

/**
 * @brief Template for warp-level reduction in shared memory.
 *
 * @tparam BLOCK_SIZE Size of the block
 * @param sharedMemory Shared memory array used for reduction
 * @param tid Thread ID within the block
 * @return __device__
 */
template <unsigned int BLOCK_SIZE>
__device__ void warpTemplateReduce(volatile float *sharedMemory, const unsigned int tid);

/**
 * @brief Optimized implementation of the dot product of two vectors using shared memory and warp-level reduction.
 *
 * @tparam BLOCK_SIZE Size of the block
 * @param a Vector a
 * @param b Vector b
 * @param c Result vector
 * @param n Size of the vectors
 * @return __global__
 */
template <unsigned int BLOCK_SIZE>
__global__ void optimizedSdot(const float *a, const float *b, float *c, const int n);