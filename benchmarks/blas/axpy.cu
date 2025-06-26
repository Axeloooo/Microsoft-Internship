#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>

#include "blas/axpy.cuh"
#include "utils/checker.cuh"
#include "utils/data.cuh"
#include "utils/timer.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void axpyCpu(const float alpha, const float *x, float *y, const int n) {
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
}

int main() {
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t bytes = N * sizeof(float);
    size_t totalBytes = 2 * bytes;

    int vectorCount = N / 4;
    size_t vectorBytes = vectorCount * sizeof(float4);
    dim3 vectorGrid((N / 4 + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float alpha = 2.0f;

    float *h_x = new float[N];
    float *h_y_cpu = new float[N];
    float *h_y_init = new float[N];
    float *h_y_gpu = new float[N];
    float *h_y_cublas = new float[N];

    // Initialize data
    InitVector<float>(h_x, N);
    InitVector<float>(h_y_init, N);
    InitVector<float>(h_y_cpu, N);
    InitVector<float>(h_y_gpu, N);
    InitVector<float>(h_y_cublas, N);

    float *d_x, *d_y_gpu, *d_y_cublas;
    float4 *d_x4, *d_y4;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y_gpu, bytes));
    CUDA_CHECK(cudaMalloc(&d_y_cublas, bytes));

    // Allocate memory for vectorized version
    CUDA_CHECK(cudaMalloc(&d_x4, vectorBytes));
    CUDA_CHECK(cudaMalloc(&d_y4, vectorBytes));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_gpu, h_y_gpu, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_cublas, h_y_cublas, bytes, cudaMemcpyHostToDevice));

    // Copy data for vectorized version
    CUDA_CHECK(cudaMemcpy(d_x4, h_x, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y4, h_y_gpu, bytes, cudaMemcpyHostToDevice));

    // Perform AXPY on CPU
    CudaTimer cpuTimer("AXPY CPU", totalBytes);
    cpuTimer.Start();
    axpyCpu(alpha, h_x, h_y_cpu, N);
    cpuTimer.Stop();
    cpuTimer.Print();

    // Perform warm up
    std::cout << "Warming up GPUs..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        naiveSaxpy<<<grid, block>>>(alpha, d_x, d_y_gpu, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        vectorizedSaxpy<<<vectorGrid, block>>>(alpha, d_x4, d_y4, vectorCount);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &alpha, d_x, 1, d_y_cublas, 1));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Reset device vectors to initial values after warmup
    CUDA_CHECK(cudaMemcpy(d_y_gpu, h_y_init, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y4, h_y_init, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y_cublas, h_y_init, bytes, cudaMemcpyHostToDevice));

    // Perform Naive AXPY using CUDA kernel
    CudaTimer gpuTimer("Naive AXPY GPU Kernel", totalBytes);
    gpuTimer.Start();
    naiveSaxpy<<<grid, block>>>(alpha, d_x, d_y_gpu, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    gpuTimer.Stop();
    gpuTimer.Print();

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y_gpu, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    VerifyVector<float>(h_y_gpu, h_y_cpu, N);

    // Perform Verctorized AXPY using CUDA kernel
    CudaTimer vectorizedGpuTimer("Vectorized AXPY GPU Kernel", totalBytes);
    vectorizedGpuTimer.Start();
    vectorizedSaxpy<<<vectorGrid, block>>>(alpha, d_x4, d_y4, vectorCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    vectorizedGpuTimer.Stop();
    vectorizedGpuTimer.Print();

    // Copy results back to host for vectorized version
    CUDA_CHECK(cudaMemcpy(h_y_gpu, d_y4, bytes, cudaMemcpyDeviceToHost));

    // Verify results for vectorized version
    VerifyVector<float>(h_y_gpu, h_y_cpu, N);

    // Perform AXPY using cuBLAS
    CudaTimer cublasTimer("AXPY cuBLAS", totalBytes);
    cublasTimer.Start();
    CHECK_CUBLAS(cublasSaxpy(cublasHandle, N, &alpha, d_x, 1, d_y_cublas, 1));
    cublasTimer.Stop();
    cublasTimer.Print();

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_y_cublas, d_y_cublas, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    VerifyVector<float>(h_y_cublas, h_y_cpu, N);

    // Clean up
    delete[] h_x;
    delete[] h_y_cpu;
    delete[] h_y_gpu;
    delete[] h_y_cublas;
    delete[] h_y_init;
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y_gpu));
    CUDA_CHECK(cudaFree(d_y_cublas));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}