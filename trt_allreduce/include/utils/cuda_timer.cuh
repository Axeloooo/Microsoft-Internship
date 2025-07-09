/**
 * @file cuda_timer.cuh
 * @author your name (you@domain.com)
 * @brief A header file for a CUDA timer class that measures the time taken by CUDA operations.
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_runtime.h>

namespace trt_allreduce::utils {

/**
 * @brief A class to measure the time taken by CUDA operations.
 *
 */
class CudaTimer {
   private:
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

   public:
    /**
     * @brief Construct a new Cuda Timer object
     *
     */
    CudaTimer();

    /**
     * @brief Construct a new Cuda Timer object
     *
     * @param stream The CUDA stream to associate with the timer. If not provided, the default stream will be used.
     */
    CudaTimer(cudaStream_t stream);

    /**
     * @brief Destroy the Cuda Timer object
     *
     */
    ~CudaTimer();

    /**
     * @brief Start the timer.
     *
     */
    void Start();

    /**
     * @brief Stop the timer and return the elapsed time in milliseconds.
     *
     * @return float
     */
    float Stop();
};

}  // namespace trt_allreduce::utils
