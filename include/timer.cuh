#pragma once

#include <iostream>
#include <cuda_runtime.h>

class CudaTimer
{
private:
    cudaEvent_t startEvent, stopEvent;
    float elapsedMs;

public:
    CudaTimer() : elapsedMs(0.0f)
    {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~CudaTimer()
    {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void start()
    {
        cudaEventRecord(startEvent);
    }

    void stop()
    {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    }

    float getElapsedMilliseconds() const
    {
        return elapsedMs;
    }
};
