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
    double flops;
    float elapsedMs;

public:
    CudaTimer(const std::string &label, const size_t &bytes)
        : label(label), bytes(bytes), flops(0), elapsedMs(0.0f)
    {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    CudaTimer(const std::string &label, const size_t &bytes, double &flops)
        : label(label), bytes(bytes), flops(flops), elapsedMs(0.0f)
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

        double bandwidth = static_cast<double>(bytes) / elapsedMs / 1e9;

        std::cout << "Bandwidth: " << bandwidth << " TB/s" << std::endl;

        if (flops > 0)
        {
            double tflops = (flops * 1e-12) / (elapsedMs * 1e-3);
            std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
        }
    }
};
