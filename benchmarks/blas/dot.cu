#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>

#include "blas/dot.cuh"
#include "utils/checker.cuh"
#include "utils/data.cuh"
#include "utils/timer.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void checkResult(const float &cpuResult, const float &gpuResult) {
    const float diff = std::fabs(cpuResult - gpuResult);
    const float relError = diff / std::fabs(cpuResult);
    if (relError > 1e-2f) {
        std::cerr << "Results do not match!" << std::endl;
        std::cerr << "CPU result: " << cpuResult << ", GPU result: " << gpuResult << " (relative error: " << relError << ")" << std::endl;
    }
    std::cout << "Results are correct! (relative error: " << relError * 100.0f << "%)" << std::endl;
}

void dotProductCpu(const float *a, const float *b, float *c, const int n) {
    double sum = 0.0;
    for (int i = 0; i < n; ++i) sum += static_cast<double>(a[i]) * b[i];
    *c = static_cast<float>(sum);
}

int main() {
    // Initialize cuBLAS handle
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 optimGrid((N + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE));

    size_t bytes = N * sizeof(float);
    size_t totalBytes = 2 * bytes;

    float *h_a = new float[N];
    float *h_b = new float[N];
    float h_c_cpu, h_c_gpu, h_c_cublas = 0.0f;

    // Initialize data
    InitVector<float>(h_a, N);
    InitVector<float>(h_b, N);

    float *d_a, *d_b, *d_c_gpu;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c_gpu, sizeof(float)));

    // Initialize d_c_gpu to zero
    CUDA_CHECK(cudaMemset(d_c_gpu, 0, sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Perform dot product on CPU
    CudaTimer cpuTimer("Dot Product CPU", totalBytes);
    cpuTimer.Start();
    dotProductCpu(h_a, h_b, &h_c_cpu, N);
    cpuTimer.Stop();
    cpuTimer.Print();

    // Perform warm up
    std::cout << "Warming up GPUs..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        CUDA_CHECK(cudaMemset(d_c_gpu, 0, sizeof(float)));
        naiveSdot<<<grid, block>>>(d_a, d_b, d_c_gpu, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemset(d_c_gpu, 0, sizeof(float)));
        optimizedSdot<BLOCK_SIZE><<<optimGrid, block, BLOCK_SIZE * sizeof(float)>>>(d_a, d_b, d_c_gpu, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CHECK_CUBLAS(cublasSdot(cublasHandle, N, d_a, 1, d_b, 1, &h_c_cublas));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Perform naive dot product using CUDA kernel
    CudaTimer naiveGpuTimer("Naive Dot Product GPU Kernel", totalBytes);
    naiveGpuTimer.Start();
    CUDA_CHECK(cudaMemset(d_c_gpu, 0, sizeof(float)));
    naiveSdot<<<grid, block>>>(d_a, d_b, d_c_gpu, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    naiveGpuTimer.Stop();
    naiveGpuTimer.Print();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_c_gpu, d_c_gpu, sizeof(float), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_c_cpu, h_c_gpu);

    // Perform optimized dot product using CUDA kernel
    CudaTimer optimizedGpuTimer("Optimized Dot Product GPU Kernel", totalBytes);
    optimizedGpuTimer.Start();
    CUDA_CHECK(cudaMemset(d_c_gpu, 0, sizeof(float)));
    optimizedSdot<BLOCK_SIZE><<<optimGrid, block, BLOCK_SIZE * sizeof(float)>>>(d_a, d_b, d_c_gpu, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    optimizedGpuTimer.Stop();
    optimizedGpuTimer.Print();

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_c_gpu, d_c_gpu, sizeof(float), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_c_cpu, h_c_gpu);

    // Perform dot product using cuBLAS
    CudaTimer cublasTimer("Dot Product cuBLAS", totalBytes);
    cublasTimer.Start();
    CHECK_CUBLAS(cublasSdot(cublasHandle, N, d_a, 1, d_b, 1, &h_c_cublas));
    cublasTimer.Stop();
    cublasTimer.Print();

    checkResult(h_c_cpu, h_c_cublas);

    // Clean up
    delete[] h_a;
    delete[] h_b;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c_gpu));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}