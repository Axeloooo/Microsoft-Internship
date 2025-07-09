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

#include "utils/checker.h"
#include "utils/cuda_timer.cuh"

namespace trt_allreduce::utils {

/**
 * @brief Construct a new Cuda Timer:: Cuda Timer object
 *
 */
CudaTimer::CudaTimer() : stream_(0) {}

/**
 * @brief Construct a new Cuda Timer:: Cuda Timer object
 *
 * @param stream The CUDA stream to associate with the timer. If not provided, the default stream will be used.
 */
CudaTimer::CudaTimer(cudaStream_t stream = 0) : stream_(stream) {
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
void CudaTimer::Start() { CUDA_CHECK(cudaEventRecord(start_event_, stream_)); }

/**
 * @brief Stop the timer and return the elapsed time in milliseconds.
 *
 * @return float
 */
float CudaTimer::Stop() {
    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_event_, stop_event_));
    return ms;
}

}  // namespace trt_allreduce::utils
