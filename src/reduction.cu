#include <iostream>
#include <chrono>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 4194304
#define BLOCK_SIZE 256

#define CHECK_CUDA(call)                                                                               \
    {                                                                                                  \
        cudaError_t err = call;                                                                        \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                                        \
        }                                                                                              \
    }

template <unsigned int blockSize>
__device__ void warpReduce(volatile int *sdata, unsigned int tid)
{
    if (blockSize >= 64)
        sdata[tid] += sdata[tid + 32];
    if (blockSize >= 32)
        sdata[tid] += sdata[tid + 16];
    if (blockSize >= 16)
        sdata[tid] += sdata[tid + 8];
    if (blockSize >= 8)
        sdata[tid] += sdata[tid + 4];
    if (blockSize >= 4)
        sdata[tid] += sdata[tid + 2];
    if (blockSize >= 2)
        sdata[tid] += sdata[tid + 1];
}

template <int blockSize>
__global__ void reduction(int *g_in_data, int *g_out_data, unsigned int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockSize * 2) + tid;
    unsigned int gridSize = blockDim.x * 2 * gridDim.x;
    sdata[tid] = 0;

    while (i < n)
    {
        sdata[tid] += g_in_data[i] + g_in_data[i + blockSize];
        i += gridSize;
    }
    __syncthreads();

    if (blockSize >= 512)
    {
        if (tid < 256)
        {
            sdata[tid] += sdata[tid + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256)
    {
        if (tid < 128)
        {
            sdata[tid] += sdata[tid + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128)
    {
        if (tid < 64)
        {
            sdata[tid] += sdata[tid + 64];
        }
        __syncthreads();
    }

    if (tid < 32)
        warpReduce<blockSize>(sdata, tid);

    if (tid == 0)
    {
        g_out_data[blockIdx.x] = sdata[0];
    }
}

int main()
{
    // Set bytes and number of blocks
    size_t bytes = N * sizeof(int);
    const int numBlocks = (N + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);

    // Allocate memory on CPU
    int *h_in = new int[N];
    int *h_out = new int[(N + BLOCK_SIZE - 1) / BLOCK_SIZE];

    int *d_in, *d_out;

    // Initialize data
    srand(42);
    for (int i = 0; i < N; i++)
    {
        h_in[i] = rand() % 100;
    }

    // Allocate memory on GPU
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, (N + BLOCK_SIZE - 1) / BLOCK_SIZE * sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Perfom warm-ups
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; i++)
    {
        reduction<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
        cudaDeviceSynchronize();
    }

    // Launch kernel
    std::cout << "Running kernel..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    reduction<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count() / 1000.0;

    // Copy output to host
    CHECK_CUDA(cudaMemcpy(h_out, d_out, (N + BLOCK_SIZE - 1) / BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Final reduction on the host
    int finalResult = h_out[0];
    for (int i = 1; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
    {
        finalResult += h_out[i];
    }

    // Print results
    double bandwidth = (duration > 0) ? (bytes / duration / 1e6) : 0;
    std::cout << "Reduced result: " << finalResult << std::endl;
    std::cout << "Time elapsed: " << duration << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << bandwidth << " GB/s" << std::endl;

    // Free Memory
    delete[] (h_in);
    delete[] (h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}