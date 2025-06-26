/**
 * @file benchmark_gemm.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft)
 * @brief A benchmark for SGEMM (Single-Precision General Matrix Multiply) using CUDA and cuBLAS.
 * @version 0.1
 * @date 2025-06-26
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>

#include "blas/gemm.cuh"
#include "utils/checker.cuh"
#include "utils/data.cuh"
#include "utils/timer.cuh"

#define M 1024
#define N 1024
#define K 1024
#define BLOCK_DIM 16

void sgemmCpu(const int m, const int n, const int k, const float alpha, const float *A, const float *B, const float beta, float *C) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float tmp = 0.0f;
            for (int p = 0; p < k; ++p) {
                tmp += A[i * k + p] * B[p * n + j];
            }
            C[i * n + j] = alpha * tmp + beta * C[i * n + j];
        }
    }
}

int main() {
    // Initialize cuBLAS
    cublasHandle_t cublasHandle;
    CHECK_CUBLAS(cublasCreate(&cublasHandle));

    // Set dimensions for kernel launch
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((M + BLOCK_DIM - 1) / BLOCK_DIM, (N + BLOCK_DIM - 1) / BLOCK_DIM);

    // Calculate memory sizes
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);
    size_t totalBytes = bytesA + bytesB + bytesC;

    // Calculate FLOPS: 2 operations (multiply and add) per matrix element
    double totalFlops = 2.0 * M * N * K;

    // Allocate host memory
    float *h_A = new float[M * K];
    float *h_B = new float[K * N];
    float *h_C = new float[M * N];
    float *h_C_cpu = new float[M * N];
    float *h_C_cublas = new float[M * N];

    // Initialize matrices
    InitializeMatrix<float>(h_A, M, K);
    InitializeMatrix<float>(h_B, K, N);
    InitializeMatrix<float>(h_C, M, N);

    // Copy C matrix for CPU and cuBLAS computations
    std::memcpy(h_C_cpu, h_C, bytesC);
    std::memcpy(h_C_cublas, h_C, bytesC);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_cublas;
    CUDA_CHECK(cudaMalloc(&d_A, bytesA));
    CUDA_CHECK(cudaMalloc(&d_B, bytesB));
    CUDA_CHECK(cudaMalloc(&d_C, bytesC));
    CUDA_CHECK(cudaMalloc(&d_C_cublas, bytesC));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytesB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_cublas, h_C, bytesC, cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 1.0f;

    // Perform CPU computation
    CudaTimer cpuTimer("SGEMM CPU", totalBytes, totalFlops);
    cpuTimer.Start();
    sgemmCpu(M, N, K, alpha, h_A, h_B, beta, h_C_cpu);
    cpuTimer.Stop();
    cpuTimer.Print();

    // Warm up GPU
    std::cout << "Warming up GPUs..." << std::endl;
    for (int i = 0; i < 3; ++i) {
        NaiveGemm<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CoalescingGemm<BLOCK_DIM><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Reset matrices after warmup
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_C_cublas, h_C, bytesC, cudaMemcpyHostToDevice));

    // Perform naive SGEMM
    CudaTimer naiveTimer("Naive SGEMM GPU", totalBytes, totalFlops);
    naiveTimer.Start();
    NaiveGemm<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    naiveTimer.Stop();
    naiveTimer.Print();

    // Copy results back and verify
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    VerifyMatrix<float>(h_C_cpu, h_C, M, N);

    // Reset output matrix CHECK THIS
    CUDA_CHECK(cudaMemcpy(d_C, h_C, bytesC, cudaMemcpyHostToDevice));

    // Perform coalescing SGEMM
    CudaTimer coalescingTimer("Coalescing SGEMM GPU", totalBytes, totalFlops);
    coalescingTimer.Start();
    CoalescingGemm<BLOCK_DIM><<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    coalescingTimer.Stop();
    coalescingTimer.Print();

    // Copy results back and verify
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytesC, cudaMemcpyDeviceToHost));
    VerifyMatrix<float>(h_C_cpu, h_C, M, N);

    // Perform cuBLAS SGEMM
    CudaTimer cublasTimer("cuBLAS SGEMM", totalBytes, totalFlops);
    cublasTimer.Start();
    CHECK_CUBLAS(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C_cublas, N));
    CUDA_CHECK(cudaDeviceSynchronize());
    cublasTimer.Stop();
    cublasTimer.Print();

    // Copy results back and verify
    CUDA_CHECK(cudaMemcpy(h_C_cublas, d_C_cublas, bytesC, cudaMemcpyDeviceToHost));
    VerifyMatrix<float>(h_C_cpu, h_C_cublas, M, N);

    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_C_cpu;
    delete[] h_C_cublas;
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_C_cublas));
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    return 0;
}