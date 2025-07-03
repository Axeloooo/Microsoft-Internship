/**
 * @file timer.cuh
 * @author your name (you@domain.com)
 * @brief A header file for a CUDA timer class that measures the time taken by CUDA operations.
 * @version 0.1
 * @date 2025-06-12
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#ifndef TIMER_CUH
#define TIMER_CUH

#include <cuda_runtime.h>

#include <iostream>
#include <string>

/**
 * @brief A class to measure the time taken by CUDA operations.
 *
 */
class CudaTimer {
   private:
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;
    std::string label_;
    unsigned int bytes_;
    double flops_;
    float elapsed_ms_;

   public:
    /**
     * @brief Construct a new Cuda Timer object
     *
     * @param label A label for the timer, useful for debugging and logging.
     * @param bytes The number of bytes processed during the operation.
     */
    CudaTimer(const std::string &label, const unsigned int &bytes);

    /**
     * @brief Construct a new Cuda Timer object
     *
     * @param label A label for the timer, useful for debugging and logging.
     * @param bytes The number of bytes processed during the operation.
     * @param flops The FLOPS (floating-point operations per second) value.
     */
    CudaTimer(const std::string &label, const unsigned int &bytes, const double &flops);

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
     * @brief Stop the timer and calculate the elapsed time.
     *
     */
    void Stop();

    /**
     * @brief Print the results of the timer.
     *
     */
    void Print() const;
};

#endif
