#include "blas/reduction.cuh"

/**
 * @brief Naive reduction kernel that sums up elements in a block of threads
 *
 * @param g_in_data Vector of integers to be reduced
 * @param g_out_data Pointer to the output integer where the result will be stored
 * @return __global__
 */
__global__ void naiveReduce(int *g_in_data, int *g_out_data) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        if (tid % (2 * stride) == 0) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&g_out_data[0], sdata[0]);
    }
}

__global__ void interleaveAddressingReduce(int *g_in_data, int *g_out_data) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index + stride < blockDim.x) {
            sdata[index] += sdata[index + stride];  // Memory issue caused by shared memory bank conflicts which can occur when multiple threads access the same memory bank
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&g_out_data[0], sdata[0]);
    }
}

__global__ void sequenceAddressingReduce(int *g_in_data, int *g_out_data) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[tid] = g_in_data[i];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)  // Thread idle issue by halving the number of threads in each iteration
    {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&g_out_data[0], sdata[0]);
    }
}

__global__ void addDuringLoadReduce(int *g_in_data, int *g_out_data) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)  // For loop overhead issue, fix by unrolling warp of 32 threads
    {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(&g_out_data[0], sdata[0]);
    }
}

__device__ void warpReduce(volatile int *sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}

__global__ void unrollLastWarpReduce(int *g_in_data, int *g_out_data) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)  // Optimize using template since we know the block size will always be 512 or 256 or 128, etc
    {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid < 32) {
        warpReduce(sdata, tid);
    }

    if (tid == 0) {
        atomicAdd(&g_out_data[0], sdata[0]);
    }
}

template <unsigned int BLOCK_SIZE>
__device__ void warpTemplateReduce(volatile int *sdata, unsigned int tid) {
    if (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32];
    if (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16];
    if (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8];
    if (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4];
    if (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2];
    if (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1];
}

template <unsigned int BLOCK_SIZE>
__global__ void completelyUnrollReduce(int *g_in_data, int *g_out_data) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sdata[tid] = g_in_data[i] + g_in_data[i + blockDim.x];
    __syncthreads();

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) warpTemplateReduce<BLOCK_SIZE>(sdata, tid);

    if (tid == 0) {
        atomicAdd(&g_out_data[0], sdata[0]);
    }
}

template <unsigned int BLOCK_SIZE>
__global__ void multipleElementsReduce(int *g_in_data, int *g_out_data, unsigned int n) {
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (BLOCK_SIZE * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n) {
        sdata[tid] += g_in_data[i] + g_in_data[i + BLOCK_SIZE];
        i += gridSize;
    }
    __syncthreads();

    if (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 256) {
        if (tid < 128) {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (BLOCK_SIZE >= 128) {
        if (tid < 64) {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) warpTemplateReduce<BLOCK_SIZE>(sdata, tid);

    if (tid == 0) {
        atomicAdd(&g_out_data[0], sdata[0]);
    }
}

template __device__ void warpTemplateReduce<256>(volatile int *sdata, unsigned int tid);

template __global__ void completelyUnrollReduce<256>(int *g_in_data, int *g_out_data);

template __global__ void multipleElementsReduce<256>(int *g_in_data, int *g_out_data, unsigned int n);