#include <cuda_runtime.h>

#include <cmath>
#include <iostream>

#include "add.cuh"
#include "checker.cuh"
#include "data.cuh"
#include "timer.cuh"

#define N 4194304
#define BLOCK_SIZE 256

void vectorAddCpu(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    size_t bytes = N * sizeof(float);
    size_t totalBytes = 3 * bytes;

    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c_cpu = new float[N];
    float *h_c_gpu = new float[N];

    // Initialize data
    InitVector<float>(h_a, N);
    InitVector<float>(h_b, N);

    float *d_a, *d_b, *d_c_gpu;

    // Allocate memory on device
    CHECK_CUDA(cudaMalloc(&d_a, bytes));
    CHECK_CUDA(cudaMalloc(&d_b, bytes));
    CHECK_CUDA(cudaMalloc(&d_c_gpu, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // Perform Vector Addition on CPU
    CudaTimer cpuTimer("Vector Addition CPU", totalBytes);
    cpuTimer.Start();
    vectorAddCpu(h_a, h_b, h_c_cpu, N);
    cpuTimer.Stop();
    cpuTimer.Print();

    // Perform warm up
    std::cout << "Warming up GPUs..." << std::endl;
    for (int i = 0; i < 3; i++) {
        add<<<grid, block>>>(d_a, d_b, d_c_gpu, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform GPU computation
    CudaTimer gpuTimer("Vector Addition GPU Kernel", totalBytes);
    gpuTimer.Start();
    add<<<grid, block>>>(d_a, d_b, d_c_gpu, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    gpuTimer.Stop();
    gpuTimer.Print();

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c_gpu, bytes, cudaMemcpyDeviceToHost));

    // Verify results
    VerifyVector<float>(h_c_cpu, h_c_gpu, N);

    // Clean up
    delete[] h_a;
    delete[] h_b;
    delete[] h_c_cpu;
    delete[] h_c_gpu;
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c_gpu));

    return 0;
}