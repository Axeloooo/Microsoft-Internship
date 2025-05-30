#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "checker.cuh"
#include "timer.cuh"
#include "data.cuh"
#include "axpy.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void axpyCpu(float alpha, const float *x, float *y, int n)
{
    for (int i = 0; i < n; i++)
    {
        y[i] = alpha * x[i] + y[i];
    }
}

int main()
{
    // Initialize cuBLAS handle
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t bytes = N * sizeof(float);

    float alpha = 2.0f;

    float *h_x = new float[N];
    float *h_y_cpu = new float[N];
    float *h_y_gpu = new float[N];
    float *h_y_cublas = new float[N];

    // Initialize data
    initVector<float>(h_x, N);
    initVector<float>(h_y_cpu, N);
    initVector<float>(h_y_gpu, N);
    initVector<float>(h_y_cublas, N);

    float *d_x, *d_y_cpu, *d_y_gpu, *d_y_cublas;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc(&d_x, bytes));
    CHECK_CUDA(cudaMalloc(&d_y_cpu, bytes));
    CHECK_CUDA(cudaMalloc(&d_y_gpu, bytes));
    CHECK_CUDA(cudaMalloc(&d_y_cublas, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_cpu, h_y_cpu, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_gpu, h_y_gpu, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y_cublas, h_y_cublas, bytes, cudaMemcpyHostToDevice));

    // Perform AXPY on CPU
    CudaTimer cpuTimer("AXPY CPU", &bytes);
    cpuTimer.start();
    axpyCpu(alpha, h_x, h_y_cpu, N);
    cpuTimer.stop();
    cpuTimer.print();

    // Perform warm up
    std::cout << "Warming up GPUs..." << std::endl;
    for (int i = 0; i < 3; ++i)
    {
        axpyGpu<<<grid, block>>>(alpha, d_x, d_y_gpu);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &alpha, d_x, 1, d_y_cublas, 1));
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform AXPY using CUDA kernel
    CudaTimer gpuTimer("AXPY GPU Kernel", &bytes);
    gpuTimer.start();
    axpyGpu<<<grid, block>>>(alpha, d_x, d_y_gpu);
    CHECK_CUDA(cudaGetLastError());
    gpuTimer.stop();
    gpuTimer.print();

    // Perform AXPY using cuBLAS
    CudaTimer cublasTimer("AXPY cuBLAS", &bytes);
    cublasTimer.start();
    CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &alpha, d_x, 1, d_y_cublas, 1));
    cublasTimer.stop();
    cublasTimer.print();

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_y_cpu, d_y_cpu, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_gpu, d_y_gpu, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_y_cublas, d_y_cublas, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    checkVector<float>(h_y_gpu, h_y_cpu, N);
    checkVector<float>(h_y_cublas, h_y_cpu, N);

    // Clean up
    delete[] h_x;
    delete[] h_y_cpu;
    delete[] h_y_gpu;
    delete[] h_y_cublas;
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y_cpu));
    CHECK_CUDA(cudaFree(d_y_gpu));
    CHECK_CUDA(cudaFree(d_y_cublas));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}