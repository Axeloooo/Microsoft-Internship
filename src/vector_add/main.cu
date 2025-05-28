#include <iostream>
#include <cmath>
#include <cuda_runtime.h>

#include "timer.cuh"
#include "checker.cuh"
#include "vector_add.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void initData(float *vector, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        vector[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void checkResult(const float *a, const float *b, const float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (std::fabs(c[i] - (a[i] + b[i])) > 1e-5)
        {
            std::cerr << "Results do not match!" << std::endl;
            std::cerr << "Error at index " << i << ": " << c[i] << " != " << a[i] + b[i] << "\n"
                      << std::endl;
            return;
        }
    }
    std::cout << "Results are correct!" << "\n"
              << std::endl;
}

void vectorAddCpu(const float *a, const float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    size_t bytes = N * sizeof(float);

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

    // Perform CPU computation
    CudaTimer cpuTimer;
    cpuTimer.start();
    vectorAddCpu(h_a, h_b, h_c_cpu, N);
    cpuTimer.stop();
    cpuTimer.print(&bytes, "CPU");
    std::cout << "CPU computation done." << "\n"
              << std::endl;

    // Perform warm up
    std::cout << "Warming up GPUs..." << "\n"
              << std::endl;
    for (int i = 0; i < 3; i++)
    {
        vectorAddGpu<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform GPU computation
    CudaTimer gpuTimer;
    gpuTimer.start();
    vectorAddGpu<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    gpuTimer.stop();
    gpuTimer.print(&bytes, "GPU");

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