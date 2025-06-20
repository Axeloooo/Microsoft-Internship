/**
 * @brief Naive and optimized implementations of the dot product of two vectors
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

template <unsigned int blockSize>
__device__ void warpTemplateReduce(volatile float *sharedMemory, const unsigned int tid) {
    if (blockSize >= 64) sharedMemory[tid] += sharedMemory[tid + 32];
    if (blockSize >= 32) sharedMemory[tid] += sharedMemory[tid + 16];
    if (blockSize >= 16) sharedMemory[tid] += sharedMemory[tid + 8];
    if (blockSize >= 8) sharedMemory[tid] += sharedMemory[tid + 4];
    if (blockSize >= 4) sharedMemory[tid] += sharedMemory[tid + 2];
    if (blockSize >= 2) sharedMemory[tid] += sharedMemory[tid + 1];
}

template <int blockSize>
__global__ void optimizedSdot(const float *a, const float *b, float *c, const int n) {
    extern __shared__ float sharedMemory[];

    const unsigned int tid = threadIdx.x;
    const unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    sharedMemory[tid] = (a[i] * b[i]) + (a[i + blockDim.x] * b[i + blockDim.x]);
    __syncthreads();

    if (blockSize >= 512) {
        if (tid < 256) {
            sharedMemory[tid] += sharedMemory[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (tid < 128) {
            sharedMemory[tid] += sharedMemory[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (tid < 64) {
            sharedMemory[tid] += sharedMemory[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32) warpTemplateReduce<blockSize>(sharedMemory, tid);

    if (tid == 0) {
        atomicAdd(c, sharedMemory[0]);
    }
}