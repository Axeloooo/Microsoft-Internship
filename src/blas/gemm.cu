#include <assert.h>  // required for assert macro in device/host code

#include "blas/gemm.cuh"

/**
 * @brief Naive implementation of single-precision matrix multiplication (SGEMM)
 *
 * @param M Number of rows in matrix A and C
 * @param N Number of columns in matrix B and C
 * @param K Number of columns in matrix A and rows in matrix B
 * @param alpha Scalar multiplier for the product of A and B
 * @param A Matrix A, which is of size MxK
 * @param B Matrix B, which is of size KxN
 * @param beta  Scalar multiplier for the existing values in C
 * @param  C Matrix C, which is the result of the multiplication
 * @return __global__
 */
__global__ void NaiveGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (uint i = 0; i < K; ++i) {
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

template <unsigned int BLOCK_SIZE>
__global__ void CoalescingGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const int cRow = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int cCol = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (cRow < M && cCol < N) {
        float tmp = 0.0;
        for (uint i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }
        C[cRow * N + cCol] = alpha * tmp + beta * C[cRow * N + cCol];
    }
}

template <unsigned int BLOCK_SIZE>
__global__ void CacheGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // the output block that we want to compute in this threadblock
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // allocate buffer for current block in fast shared mem
    // shared mem is shared between all threads in a block
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // the inner row & col that we're accessing in this thread
    const uint threadCol = threadIdx.x % BLOCK_SIZE;
    const uint threadRow = threadIdx.x / BLOCK_SIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCK_SIZE * K;                      // row=cRow, col=0
    B += cCol * BLOCK_SIZE;                          // row=0, col=cCol
    C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE;  // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
        // Have each thread load one of the elements in A & B
        // Make the threadCol (=threadIdx.x) the consecutive index
        // to allow global memory access coalescing
        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

        // block threads in this block until cache is fully populated
        __syncthreads();
        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // execute the dotproduct on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCK_SIZE + dotIdx] * Bs[dotIdx * BLOCK_SIZE + threadCol];
        }
        // need to sync again at the end, to avoid faster threads
        // fetching the next block into the cache before slower threads are done
        __syncthreads();
    }
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

template <unsigned int BM, unsigned int BN, unsigned int BK, unsigned int TM>
__global__ void Tiling1DGemm(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // If we flip x and y here we get ~30% less performance for large matrices.
    // The current, 30% faster configuration ensures that blocks with sequential
    // blockIDs access columns of B sequentially, while sharing the same row of A.
    // The slower configuration would share columns of A, but access into B would
    // be non-sequential. So the faster configuration has better spatial locality
    // and hence a greater L2 hit rate.
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    // each warp will calculate 32*TM elements, with 32 being the columnar dim.
    const int threadCol = threadIdx.x % BN;
    const int threadRow = threadIdx.x / BN;

    // allocate space for the current blocktile in SMEM
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Move blocktile to beginning of A's row and B's column
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // better exploit the cache sizes
    assert(BM * BK == blockDim.x);
    assert(BN * BK == blockDim.x);
    const uint innerColA = threadIdx.x % BK;  // warp-level GMEM coalescing
    const uint innerRowA = threadIdx.x / BK;
    const uint innerColB = threadIdx.x % BN;  // warp-level GMEM coalescing
    const uint innerRowB = threadIdx.x / BN;

    // allocate thread-local cache for results in registerfile
    float threadResults[TM] = {0.0};

    // outer loop over block tiles
    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        // populate the SMEM caches
        As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
        Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
        __syncthreads();

        // advance blocktile
        A += BK;
        B += BK * N;

        // calculate per-thread results
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            // we make the dotproduct loop the outside loop, which facilitates
            // reuse of the Bs entry, which we can cache in a tmp var.
            float tmpB = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; ++resIdx) {
                threadResults[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    // write out the results
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        C[(threadRow * TM + resIdx) * N + threadCol] = alpha * threadResults[resIdx] + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}

template __global__ void CoalescingGemm<16>(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

template __global__ void CacheGemm<16>(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);

template __global__ void Tiling1DGemm<32, 32, 32, 32>(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C);