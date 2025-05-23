#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "timer.cuh"
#include "checker.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void initData(int *vector, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        vector[i] = rand() % 100;
    }
}

void checkResult(int cpuResult, int gpuResult)
{
    if (fabs(cpuResult - gpuResult) > 1e-5)
    {
        std::cerr << "Results do not match!" << std::endl;
        std::cerr << "CPU result: " << cpuResult << ", GPU result: " << gpuResult << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Results are correct!" << std::endl;
}

void reductionCpu(int *h_in_data, int *h_out_data, unsigned int n)
{
    int sum = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        sum += h_in_data[i];
    }
    h_out_data[0] = sum;
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
__global__ void reductionGpu(int *g_in_data, int *g_out_data, unsigned int n)
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
    size_t bytes = N * sizeof(int);
    const int numBlocks = (N + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);

    int *h_in = new int[N];
    int *h_out_cpu = new int[1];
    int *h_out_gpu = new int[(N + BLOCK_SIZE - 1) / BLOCK_SIZE];

    initData(h_in, N);

    int *d_in, *d_out;

    // Allocate memory on GPU
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, (N + BLOCK_SIZE - 1) / BLOCK_SIZE * sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Perfom warm up
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; i++)
    {
        reductionGpu<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform CPU computation
    std::cout << "Performing CPU computation..." << std::endl;
    CudaTimer cpuTimer;
    cpuTimer.start();
    reductionCpu(h_in, h_out_cpu, N);
    cpuTimer.stop();
    float cpuDuration = cpuTimer.getElapsedMilliseconds();
    std::cout << "CPU reduction time: " << cpuDuration << " ms" << std::endl;

    // Perform GPU computation
    std::cout << "Performing GPU computation..." << std::endl;
    CudaTimer gpuTimer;
    gpuTimer.start();
    reductionGpu<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    gpuTimer.stop();
    float gpuDuration = gpuTimer.getElapsedMilliseconds();
    std::cout << "GPU reduction time: " << gpuDuration << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << (bytes / gpuDuration / 1e6) << " GB/s" << std::endl;
    std::cout << "Speedup: " << (cpuDuration / gpuDuration) << "x" << std::endl;
    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, (N + BLOCK_SIZE - 1) / BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Final reductionGpu on the host
    int finalResult = h_out_gpu[0];
    for (int i = 1; i < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i)
    {
        finalResult += h_out_gpu[i];
    }

    // Check results
    checkResult(h_out_cpu[0], finalResult);

    // Print results
    std::cout << "Reduced result: " << finalResult << std::endl;

    // Free Memory
    delete[] (h_in);
    delete[] (h_out_cpu);
    delete[] (h_out_gpu);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}