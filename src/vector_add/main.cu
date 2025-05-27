#include <stdio.h>
#include <stdlib.h>
#include <math.h>
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
        vector[i] = (float)rand() / RAND_MAX;
    }
}

void checkResult(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (fabs(c[i] - (a[i] + b[i])) > 1e-5)
        {
            fprintf(stderr, "Results do not match!\n");
            fprintf(stderr, "Error at index %d: %f != %f\n", i, c[i], a[i] + b[i]);
            return;
        }
    }
    printf("Results are correct!\n");
}

void vectorAddCpu(float *a, float *b, float *c, int n)
{
    for (int i = 0; i < n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

int main()
{
    size_t bytes = N * sizeof(int);

    float *h_a = (float *)malloc(N * sizeof(float));
    float *h_b = (float *)malloc(N * sizeof(float));
    float *h_c_cpu = (float *)malloc(N * sizeof(float));
    float *h_c_gpu = (float *)malloc(N * sizeof(float));

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
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++)
    {
        vectorAddGpu<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    // Perform CPU computation
    printf("Performing CPU computation...\n");
    CudaTimer cpuTimer;
    startTimer(&cpuTimer);
    vectorAddCpu(h_a, h_b, h_c_cpu, N);
    stopTimer(&cpuTimer);
    printTimer(&cpuTimer, &bytes);

    // Perform GPU computation
    printf("Performing GPU computation...\n");
    CudaTimer gpuTimer;
    startTimer(&gpuTimer);
    vectorAddGpu<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    stopTimer(&gpuTimer);
    printTimer(&gpuTimer, &bytes);

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_c_gpu, d_c, bytes, cudaMemcpyDeviceToHost));

    // Check results
    checkResult(h_a, h_b, h_c_gpu, N);

    // Destroy timers
    destroyTimer(&cpuTimer);
    destroyTimer(&gpuTimer);

    // Free memory
    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    return 0;
}