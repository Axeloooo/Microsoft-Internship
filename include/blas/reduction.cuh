/**
 * @file reduction.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CUDA implementation of various reduction algorithms.
 * @version 0.1
 * @date 2025-06-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

/**
 * @brief Naive reduction kernel that sums all elements in the input array.
 *
 * @param g_in_data Pointer to the input data in global memory.
 * @param g_out_data Pointer to the output data in global memory, where the result will be stored.
 * @return __global__
 */
__global__ void naiveReduce(int *g_in_data, int *g_out_data);

/**
 * @brief Interleaved addressing reduction kernel that sums elements in the input array using interleaved addressing.
 *
 * @param g_in_data Pointer to the input data in global memory.
 * @param g_out_data Pointer to the output data in global memory, where the result will be stored.
 * @return __global__
 */
__global__ void interleaveAddressingReduce(int *g_in_data, int *g_out_data);

/**
 * @brief Sequence addressing reduction kernel that sums elements in the input array using sequence addressing.
 *
 * @param g_in_data Pointer to the input data in global memory.
 * @param g_out_data Pointer to the output data in global memory, where the result will be stored.
 * @return __global__
 */
__global__ void sequenceAddressingReduce(int *g_in_data, int *g_out_data);

/**
 * @brief Reduction kernel that performs a reduction during the loading phase.
 *
 * @param g_in_data Pointer to the input data in global memory.
 * @param g_out_data Pointer to the output data in global memory, where the result will be stored.
 * @return __global__
 */
__global__ void addDuringLoadReduce(int *g_in_data, int *g_out_data);

/**
 * @brief Warp-level reduction function that reduces data within a warp.
 *
 * @param sdata Pointer to shared memory used for reduction.
 * @param tid Thread ID within the warp.
 * @return __device__
 */
__device__ void warpReduce(volatile int *sdata, int tid);

/**
 * @brief Unroll last warp reduction kernel that performs a reduction on the last warp of threads.
 *
 * @param g_in_data Pointer to the input data in global memory.
 * @param g_out_data Pointer to the output data in global memory, where the result will be stored.
 * @return __global__
 */
__global__ void unrollLastWarpReduce(int *g_in_data, int *g_out_data);

/**
 * @brief Template for warp-level reduction in shared memory.
 *
 * @tparam BLOCK_SIZE Size of the block (number of threads per block).
 * @param sdata Pointer to shared memory used for reduction.
 * @param tid Thread ID within the block.
 * @return __device__
 */
template <unsigned int BLOCK_SIZE>
__device__ void warpTemplateReduce(volatile int *sdata, unsigned int tid);

/**
 * @brief Completely unroll reduction kernel that performs a reduction by completely unrolling the loop.
 *
 * @tparam BLOCK_SIZE Size of the block (number of threads per block).
 * @param g_in_data Pointer to the input data in global memory.
 * @param g_out_data Pointer to the output data in global memory, where the result will be stored.
 * @return __global__
 */
template <unsigned int BLOCK_SIZE>
__global__ void completelyUnrollReduce(int *g_in_data, int *g_out_data);

/**
 * @brief Multiple elements reduction kernel that reduces multiple elements in a block of threads.
 *
 * @tparam BLOCK_SIZE Size of the block (number of threads per block).
 * @param g_in_data Pointer to the input data in global memory.
 * @param g_out_data Pointer to the output data in global memory, where the result will be stored.
 * @param n Number of elements to reduce in the input data.
 * @return __global__
 */
template <unsigned int BLOCK_SIZE>
__global__ void multipleElementsReduce(int *g_in_data, int *g_out_data, unsigned int n);