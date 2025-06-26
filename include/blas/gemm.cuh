/**
 * @file gemm.cuh
 * @author Axel Omar Sanchez Peralta
 * @brief CUDA implementation of General Matrix Multiplication (GEMM).
 * @version 0.1
 * @date 2025-06-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

/**
 * @brief Naive implementation of GEMM (General Matrix Multiplication).
 *
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 * @param alpha Scalar multiplier for the product of A and B.
 * @param A Pointer to matrix A in row-major order.
 * @param B Pointer to matrix B in row-major order.
 * @param beta Scalar multiplier for matrix C.
 * @param C Pointer to matrix C in row-major order, which will be updated with the result.
 * @return __global__
 */
__global__ void NaiveGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

/**
 * @brief Optimized GEMM implementation using general memory coalescing.
 *
 * @tparam BLOCK_SIZE Size of the block (number of threads per block).
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 * @param alpha Scalar multiplier for the product of A and B.
 * @param A Pointer to matrix A in row-major order.
 * @param B Pointer to matrix B in row-major order.
 * @param beta Scalar multiplier for matrix C.
 * @param C Pointer to matrix C in row-major order, which will be updated with the result.
 * @return __global__
 */
template <unsigned int BLOCK_SIZE>
__global__ void CoalescingGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

/**
 * @brief Optimized GEMM implementation using cache optimization techniques.
 *
 * @tparam BLOCK_SIZE Size of the block (number of threads per block).
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 * @param alpha Scalar multiplier for the product of A and B.
 * @param A Pointer to matrix A in row-major order.
 * @param B Pointer to matrix B in row-major order.
 * @param beta Scalar multiplier for matrix C.
 * @param C Pointer to matrix C in row-major order, which will be updated with the result.
 * @return __global__
 */
template <unsigned int BLOCK_SIZE>
__global__ void CacheGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

/**
 * @brief Tiling 1D GEMM implementation using shared memory for better performance.
 *
 * @tparam BM Block size for matrix A.
 * @tparam BN Block size for matrix B.
 * @tparam BK Block size for matrix K.
 * @tparam TM Thread size for matrix multiplication.
 * @param M Number of rows in matrix A and C.
 * @param N Number of columns in matrix B and C.
 * @param K Number of columns in matrix A and rows in matrix B.
 * @param alpha Scalar multiplier for the product of A and B.
 * @param A Pointer to matrix A in row-major order.
 * @param B Pointer to matrix B in row-major order.
 * @param beta Scalar multiplier for matrix C.
 * @param C Pointer to matrix C in row-major order, which will be updated with the result.
 * @return __global__
 */
template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int TM>
__global__ void Tiling1DGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);