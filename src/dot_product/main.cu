#include <stdio.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "timer.cuh"
#include "checker.cuh"

#define N 4194304

void initData(float *vector, int n)
{
    srand(42);
    for (int i = 0; i < n; i++)
    {
        vector[i] = (float)rand() / RAND_MAX;
    }
}

void checkResult(float *cpuResult, float *gpuResult)
{
    float diff = fabs(*cpuResult - *gpuResult);
    float relError = diff / fabs(*cpuResult);
    if (relError > 1e-3) // using relative tolerance 0.1%
    {
        fprintf(stderr, "Results do not match!\n");
        fprintf(stderr, "CPU result: %f, GPU result: %f (relative error: %f)\n", *cpuResult, *gpuResult, relError);
        exit(EXIT_FAILURE);
    }
    printf("Results are correct!\n");
}

void dotProductCpu(float *a, float *b, float *c, int n)
{
    double sum = 0.0;
    for (int i = 0; i < n; ++i)
        sum += static_cast<double>(a[i]) * b[i];
    *c = static_cast<float>(sum);
}

int main()
{
    size_t bytes = N * sizeof(float);

    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
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
    printf("Performing CPU computation...\n");
    CudaTimer cpuTimer;
    startTimer(&cpuTimer);
    dotProductCpu(h_a, h_b, &h_c_cpu, N);
    stopTimer(&cpuTimer);
    printTimer(&cpuTimer, &bytes);

    // Perform GPU computation
    printf("Performing GPU computation...\n");
    CudaTimer gpuTimer;
    startTimer(&gpuTimer);
    CHECK_CUBLAS(cublasSdot(cublasHandle, N, d_a, 1, d_b, 1, &h_c_gpu));
    CHECK_CUDA(cudaDeviceSynchronize());
    stopTimer(&gpuTimer);
    printTimer(&gpuTimer, &bytes);

    // Check results
    checkResult(&h_c_cpu, &h_c_gpu);

    // Destroy timers
    destroyTimer(&cpuTimer);
    destroyTimer(&gpuTimer);

    // Destroy cuBLAS handle
    CHECK_CUBLAS(cublasDestroy(cublasHandle));

    // Free memory
    free(h_a);
    free(h_b);
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));

    return 0;
}