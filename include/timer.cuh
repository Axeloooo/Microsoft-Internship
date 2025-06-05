#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <string>

class CudaTimer
{
private:
    cudaEvent_t startEvent, stopEvent;
    std::string label;
    size_t bytes;
    float elapsedMs;

public:
    CudaTimer(const std::string &label, const size_t *bytes) : elapsedMs(0.0f), label(label), bytes(*bytes)
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

    void print() const
    {
        std::cout << label << ":" << std::endl;
        std::cout << "Latency: " << elapsedMs << " ms" << std::endl;

        double bandwidthTBps = static_cast<double>(bytes) / elapsedMs / 1e9;

        std::cout << "Bandwidth: " << bandwidthTBps << " TB/s" << std::endl;
    }
};
