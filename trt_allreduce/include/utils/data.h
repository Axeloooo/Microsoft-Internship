/**
 * @file data.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Header file for data utilities.
 * @version 0.1
 * @date 2025-07-04
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

namespace trt_allreduce::utils {
/**
 * @brief Initialize a vector with random values
 *
 * @tparam T Data type for the vector (e.g., float, __half)
 * @param vector Pointer to the vector data
 * @param n Number of elements in the vector
 */
template <typename T>
void InitVector(T *vector, const std::size_t &n);

}  // namespace trt_allreduce::utils