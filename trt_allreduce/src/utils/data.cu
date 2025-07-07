/**
 * @file data.cu
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for initializing vectors with random values.
 * @version 0.1
 * @date 2025-07-04
 *
 * @copyright Copyright (c) 2025
 *
 */

#include <cuda_fp16.h>

#include "utils/data.h"

namespace trt_allreduce::utils {

/**
 * @brief Initialize a vector with random values.
 *
 * @tparam float
 * @param vector Pointer to the vector to be initialized.
 * @param n Number of elements in the vector.
 */
template <>
void InitVector<float>(float *vector, const std::size_t &n) {
    for (std::size_t i = 0; i < n; ++i) {
        vector[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

/**
 * @brief Initialize a vector with random values for __half data type.
 *
 * @tparam  __half
 * @param vector Pointer to the vector to be initialized.
 * @param n Number of elements in the vector.
 */
template <>
void InitVector<__half>(__half *vector, const std::size_t &n) {
    for (std::size_t i = 0; i < n; ++i) {
        float val = static_cast<float>(rand()) / RAND_MAX;
        vector[i] = __float2half(val);
    }
}
}  // namespace trt_allreduce::utils