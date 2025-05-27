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
    const int gridDim = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
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
        naiveReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
        multipleElementsReduce<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
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

    // Perform naive reduce GPU computation
    printf("Performing naive reduce GPU computation...\n");
    CudaTimer naiveReduceTimer;
    startTimer(&naiveReduceTimer);
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    naiveReduce<<<gridDim, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out);
    CHECK_CUDA(cudaDeviceSynchronize());
    stopTimer(&naiveReduceTimer);
    printTimer(&naiveReduceTimer, &bytes);

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(&h_out_cpu[0], &h_out_gpu[0]);

    // Perform multiple elements reduce GPU computation
    printf("Performing multiple elements reduce GPU computation...\n");
    CudaTimer multipleElementsReduceTimer;
    startTimer(&multipleElementsReduceTimer);
    CHECK_CUDA(cudaMemset(d_out, 0, sizeof(int)));
    multipleElementsReduce<BLOCK_SIZE><<<numBlocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(d_in, d_out, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    stopTimer(&multipleElementsReduceTimer);
    printTimer(&multipleElementsReduceTimer, &bytes);

    // Copy data from device to host
    CHECK_CUDA(cudaMemcpy(h_out_gpu, d_out, sizeof(int), cudaMemcpyDeviceToHost));

    // Check results
    checkResult(&h_out_cpu[0], &h_out_gpu[0]);

    // Destroy CUDA timers
    destroyTimer(&cpuTimer);
    destroyTimer(&naiveReduceTimer);
    destroyTimer(&multipleElementsReduceTimer);

    // Free Memory
    free(h_in);
    free(h_out_cpu);
    free(h_out_gpu);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    return 0;
}