#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>

#include "timer.cuh"
#include "checker.cuh"

#define N 4194304

void initData(float *vector, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        vector[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

void checkResult(const float &cpuResult, const float &gpuResult)
{
    float diff = std::fabs(cpuResult - gpuResult);
    float relError = diff / std::fabs(cpuResult);
    if (relError > 1e-3)
    {
        std::cerr << "Results do not match!" << std::endl;
        std::cerr << "CPU result: " << cpuResult << ", GPU result: " << gpuResult << " (relative error: " << relError << ")" << "\n"
                  << std::endl;
        exit(EXIT_FAILURE);
    }
    std::cout << "Results are correct!" << "\n"
              << std::endl;
}

void dotProductCpu(const float *a, const float *b, float *c, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += static_cast<double>(a[i]) * b[i];
    *c = static_cast<float>(sum);
}

int main()
{
    size_t bytes = N * sizeof(float);

    float *h_a = new float[N];
    float *h_b = new float[N];
    float h_c_cpu, h_c_gpu;

    // Initialize data
    initData(h_a, N);
    initData(h_b, N);

    float *d_a, *d_b;

    // Initialize cuBLAS handle
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // Allocate memory on GPU
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Perform CPU computation
    CudaTimer cpuTimer;
    cpuTimer.start();
    dotProductCpu(h_a, h_b, &h_c_cpu, N);
    cpuTimer.stop();
    cpuTimer.print(&bytes, "CPU");
    std::cout << "CPU result: " << h_c_cpu << "\n"
              << std::endl;

    // Perform warm up
    std::cout << "Warming up GPUs..." << "\n"
              << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        CHECK_CUBLAS(cublasSdot(cublasHandle, N, d_a, 1, d_b, 1, &h_c_gpu));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform GPU computation
    CudaTimer gpuTimer;
    gpuTimer.start();
    CHECK_CUBLAS(cublasSdot(cublasHandle, N, d_a, 1, d_b, 1, &h_c_gpu));
    CHECK_CUDA(cudaDeviceSynchronize());
    gpuTimer.stop();
    gpuTimer.print(&bytes, "GPU");

    // Check results
    checkResult(h_c_cpu, h_c_gpu);

    // Destroy cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    // Free memory
    delete[] h_a;
    delete[] h_b;
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}