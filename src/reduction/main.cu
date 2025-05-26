#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#include "timer.cuh"
#include "checker.cuh"
#include "reduction.cuh"

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

void checkResult(int *cpuResult, int *gpuResult)
{
    if (fabs(*cpuResult - *gpuResult) > 1e-5)
    {
        fprintf(stderr, "Results do not match!\n");
        fprintf(stderr, "CPU result: %d, GPU result: %d\n", *cpuResult, *gpuResult);
        exit(EXIT_FAILURE);
    }
    printf("Results are correct!\n");
}

void reductionCpu(int *h_in_data, int *h_out_data, unsigned int n)
{
    h_out_data[0] = 0;
    for (unsigned int i = 0; i < n; i++)
    {
        h_out_data[0] += h_in_data[i];
    }
}

int main()
{
    size_t bytes = N * sizeof(int);
    const int numBlocks = (N + (2 * BLOCK_SIZE) - 1) / (2 * BLOCK_SIZE);

    int *h_in = (int *)malloc(N * sizeof(int));
    int *h_out_cpu = (int *)malloc(sizeof(int));
    int *h_out_gpu = (int *)malloc(sizeof(int));

    initData(h_in, N);

    int *d_in, *d_out;

    // Allocate memory on GPU
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // Perfom warm up
    printf("Warming up...\n");
    for (int i = 0; i < 3; i++)
    {
        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        reductionGpu<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Perform CPU computation
    printf("Performing CPU computation...\n");
    CudaTimer cpuTimer;
    startTimer(&cpuTimer);
    reductionCpu(h_in, h_out_cpu, N);
    stopTimer(&cpuTimer);
    printTimer(&cpuTimer, &bytes);

    // Perform GPU computation
    printf("Performing GPU computation...\n");
    CudaTimer gpuTimer;
    startTimer(&gpuTimer);
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    reductionGpu<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    stopTimer(&gpuTimer);
    printTimer(&gpuTimer, &bytes);

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(&h_out_cpu[0], &h_out_gpu[0]);

    // Print results
    printf("Reduced result: %d\n", h_out_gpu[0]);

    // Destroy timers
    destroyTimer(&cpuTimer);
    destroyTimer(&gpuTimer);

    // Free Memory
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}