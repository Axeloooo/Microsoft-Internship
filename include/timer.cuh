#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>

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
        elapsedMs = 0.0f;
        cudaEventRecord(startEvent);
    }

    void stop()
    {
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&elapsedMs, startEvent, stopEvent);
    }

    void print(const size_t *bytes, const std::string &label) const
    {
        std::cout << label << ":" << std::endl;
        std::cout << "Computation time: " << elapsedMs << " ms" << std::endl;
        std::cout << "Effective bandwidth: " << (static_cast<double>(*bytes) / elapsedMs / 1e6) << " GB/s" << std::endl;
    }
};
