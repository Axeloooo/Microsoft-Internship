/**
 * @file timer.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CUDA timer utility for measuring execution time and performance
 * metrics.
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "timer.cuh"

CudaTimer::CudaTimer(const std::string &label, const size_t &bytes)
    : label_(label), bytes_(bytes), flops_(0), elapsed_ms_(0.0f) {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

CudaTimer::CudaTimer(const std::string &label, const size_t &bytes,
                     double &flops)
    : label_(label), bytes_(bytes), flops_(flops), elapsed_ms_(0.0f) {
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

CudaTimer::~CudaTimer() {
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
}

void CudaTimer::Start() {
    elapsed_ms_ = 0.0f;
    cudaEventRecord(start_event_);
}

void CudaTimer::Stop() {
    cudaEventRecord(stop_event_);
    cudaEventSynchronize(stop_event_);
    cudaEventElapsedTime(&elapsed_ms_, start_event_, stop_event_);
}

void CudaTimer::Print() const {
    std::cout << label_ << ":" << std::endl;
    std::cout << "Latency: " << elapsed_ms_ << " ms" << std::endl;

    double bandwidth = static_cast<double>(bytes_) / elapsed_ms_ / 1e9;

    std::cout << "Bandwidth: " << bandwidth << " TB/s" << std::endl;

    if (flops_ > 0) {
        double tflops = (flops_ * 1e-12) / (elapsed_ms_ * 1e-3);
        std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    }
}
