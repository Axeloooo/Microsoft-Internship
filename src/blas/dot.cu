#include <cuda_runtime.h>

#include "blas/dot.cuh"

/**
 * @brief Naive implementation of the dot product of two vectors
 *
 * @param a Vector a
 * @param b Vector b
 * @param c Result vector
 * @param n Size of the vectors
 * @return __global__
 */
__global__ void naiveSdot(const float *a, const float *b, float *c, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        atomicAdd(c, a[i] * b[i]);
    }
}

template <unsigned int BLOCK_SIZE>
__device__ void warpTemplateReduce(volatile float *sharedMemory, const unsigned int tid) {
    if (BLOCK_SIZE >= 64) sharedMemory[tid] += sharedMemory[tid + 32];
    if (BLOCK_SIZE >= 32) sharedMemory[tid] += sharedMemory[tid + 16];
    if (BLOCK_SIZE >= 16) sharedMemory[tid] += sharedMemory[tid + 8];
    if (BLOCK_SIZE >= 8) sharedMemory[tid] += sharedMemory[tid + 4];
    if (BLOCK_SIZE >= 4) sharedMemory[tid] += sharedMemory[tid + 2];
    if (BLOCK_SIZE >= 2) sharedMemory[tid] += sharedMemory[tid + 1];
}

template <unsigned int BLOCK_SIZE>
__global__ void optimizedSdot(const float *a, const float *b, float *c, const int n) {
    extern __shared__ float sharedMemory[];

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sharedMemory[tid] = (a[i] * b[i]) + (a[i + blockDim.x] * b[i + blockDim.x]);
    __syncthreads();

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            sharedMemory[tid] += sharedMemory[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            sharedMemory[tid] += sharedMemory[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            sharedMemory[tid] += sharedMemory[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) warpTemplateReduce<BLOCK_SIZE>(sharedMemory, tid);

    if (tid == 0) {
        atomicAdd(c, sharedMemory[0]);
    }
}

template __device__ void warpTemplateReduce<256>(volatile float *sharedMemory, const unsigned int tid);

template __global__ void optimizedSdot<256>(const float *a, const float *b, float *c, const int n);
