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

#include <iostream>
#include <string>

namespace trt_allreduce::utils {

/**
 * @brief A class to measure the time taken by CUDA operations.
 *
 */
class CudaTimer {
   private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    std::string label_;
    float elapsed_ms_;

   public:
    /**
     * @brief Construct a new Cuda Timer object
     *
     * @param label A label for the timer, useful for debugging and logging.
     */
    CudaTimer(const std::string &label);

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
