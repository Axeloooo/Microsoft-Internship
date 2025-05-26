#pragma once

#include <cuda_runtime.h>

typedef struct
{
    cudaEvent_t startEvent, stopEvent;
    float elapsedMs;
} CudaTimer;

void startTimer(CudaTimer *timer)
{
    cudaEventCreate(&timer->startEvent);
    cudaEventCreate(&timer->stopEvent);
    timer->elapsedMs = 0.0f;
    cudaEventRecord(timer->startEvent);
}

void stopTimer(CudaTimer *timer)
{
    cudaEventRecord(timer->stopEvent);
    cudaEventSynchronize(timer->stopEvent);
    cudaEventElapsedTime(&timer->elapsedMs, timer->startEvent, timer->stopEvent);
}

void printTimer(const CudaTimer *timer, const size_t *bytes)
{
    printf("Reduction time: %f ms\n", timer->elapsedMs);
    printf("Effective bandwidth: %f GB/s\n", (double)(*bytes) / (timer->elapsedMs) / 1e6);
}

void destroyTimer(CudaTimer *timer)
{
    cudaEventDestroy(timer->startEvent);
    cudaEventDestroy(timer->stopEvent);
}
