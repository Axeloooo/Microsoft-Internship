/**
 * @file cuda_timer.cu
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

#include "utils/checker.h"
#include "utils/cuda_timer.cuh"

namespace trt_allreduce::utils {

/**
 * @brief Construct a new Cuda Timer:: Cuda Timer object
 *
 * @param label A label for the timer, useful for debugging and logging.
 */
CudaTimer::CudaTimer(const std::string &label) : label_(label), elapsed_ms_(0.0f) {
    CUDA_CHECK(cudaEventCreate(&start_event_));
    CUDA_CHECK(cudaEventCreate(&stop_event_));
}

/**
 * @brief Destroy the Cuda Timer object
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
void CudaTimer::Start() { CUDA_CHECK(cudaEventRecord(start_event_)); }

/**
 * @brief Stop the timer and return the elapsed time in milliseconds.
 *
 * @return float
 */
float CudaTimer::Stop() {
    CUDA_CHECK(cudaEventRecord(stop_event_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms_, start_event_, stop_event_));
    return elapsed_ms_;
}

}  // namespace trt_allreduce::utils
