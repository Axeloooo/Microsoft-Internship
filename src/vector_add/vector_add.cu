#include <iostream>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "timer.cuh"
#include "checker.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void initData(float *vector, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        vector[i] = (float)rand() / RAND_MAX;
    }
}

void checkResult(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5)
        {
            std::cerr << "Results do not match!" << std::endl;
            std::cerr << "Error at index " << i << ": " << c[i] << " != " << a[i] + b[i] << std::endl;
            return;
        }
    }
    std::cout << "Results are correct!" << std::endl;
}

void vectorAddCpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void vectorAddGpu(float *a, float *b, float *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    size_t bytes = N * sizeof(int);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c_cpu = new float[N];
    float *h_c_gpu = new float[N];

    initData(h_a, N);
    initData(h_b, N);

    float *d_a, *d_b, *d_c;

    // Allocate memory on device
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Perform warm up
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 3; i++)
    {
        vectorAddGpu<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Perform CPU computation
    std::cout << "Performing CPU computation..." << std::endl;
    CudaTimer cpuTimer;
    cpuTimer.start();
    vectorAddCpu(h_a, h_b, h_c_cpu, N);
    cpuTimer.stop();
    float cpuDuration = cpuTimer.getElapsedMilliseconds();
    std::cout << "CPU computation time: " << cpuDuration << " ms" << std::endl;

    // Perform GPU computation
    std::cout << "Performing GPU computation..." << std::endl;
    CudaTimer gpuTimer;
    gpuTimer.start();
    vectorAddGpu<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    gpuTimer.stop();
    float gpuDuration = gpuTimer.getElapsedMilliseconds();
    std::cout << "GPU computation time: " << gpuDuration << " ms" << std::endl;
    std::cout << "Effective bandwidth: " << (bytes / gpuDuration / 1e6) << " GB/s" << std::endl;

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_a, h_b, h_c_gpu, N);

    // Free memory
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}