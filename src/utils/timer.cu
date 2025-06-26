/**
 * @file timer.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CUDA timer utility for measuring execution time and performance metrics.
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_runtime.h>

#include <iostream>
#include <string>

#include "utils/checker.cuh"
#include "utils/timer.cuh"

/**
 * @brief Construct a new Cuda Timer:: Cuda Timer object
 *
 * @param label A label for the timer, useful for debugging and logging.
 * @param bytes The number of bytes processed during the operation.
 */
CudaTimer::CudaTimer(const std::string &label, const unsigned int &bytes) : label_(label), bytes_(bytes), flops_(0.0), elapsed_ms_(0.0f) {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

/**
 * @brief Construct a new Cuda Timer object
 *
 * @param label A label for the timer, useful for debugging and logging.
 * @param bytes The number of bytes processed during the operation.
 * @param flops The FLOPS (floating-point operations per second) value.
 */
CudaTimer::CudaTimer(const std::string &label, const unsigned int &bytes, const double &flops) : label_(label), bytes_(bytes), flops_(flops), elapsed_ms_(0.0f) {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

/**
 * @brief Destroy the Cuda Timer:: Cuda Timer object
 *
 */
CudaTimer::~CudaTimer() {
    CUDA_CHECK(cudaEventDestroy(start_event_));
    CUDA_CHECK(cudaEventDestroy(stop_event_));
}

/**
 * @brief Start the timer.
 *
 */
void CudaTimer::Start() {
    elapsed_ms_ = 0.0f;
    CUDA_CHECK(cudaEventRecord(start_event_));
}

/**
 * @brief Stop the timer and calculate the elapsed time.
 *
 */
void CudaTimer::Stop() {
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_, start_event_, stop_event_));
}

/**
 * @brief Print the results of the timer.
 *
 */
void CudaTimer::Print() const {
    std::cout << label_ << ":" << std::endl;
    std::cout << "Latency: " << elapsed_ms_ << " ms" << std::endl;

    const double bandwidth = static_cast<double>(bytes_) / elapsed_ms_ / 1e9;

    std::cout << "Bandwidth: " << bandwidth << " TB/s" << std::endl;

    if (flops_ > 0.0) {
        const double tflops = (flops_ * 1e-12) / (elapsed_ms_ * 1e-3);
        std::cout << "Performance: " << tflops << " TFLOPS" << std::endl;
    }
}
