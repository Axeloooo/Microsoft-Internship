/**
 * @file kvCacheIndex.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Header file for KVCacheIndex utilities.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <cstdint>

#include "common/assert.h"

namespace trt_allreduce::kernels {

class KVCacheIndex {
   public:
    using UnderlyingType = std::int32_t;

    // Flag indicating KVCacheIndex refers to secondary pool
    static constexpr UnderlyingType kSecondaryPoolFlag = static_cast<UnderlyingType>(1) << (8 * sizeof(UnderlyingType) - 1);

    explicit KVCacheIndex(UnderlyingType value, bool isSecondary = false) : value{isSecondary ? value | kSecondaryPoolFlag : value} { TLLM_CHECK_DEBUG(value >= 0); }

    __host__ __device__ [[nodiscard]] UnderlyingType get() const { return value & (~kSecondaryPoolFlag); }

    __host__ __device__ [[nodiscard]] bool isPrimary() const { return (value & kSecondaryPoolFlag) == 0; }

   private:
    UnderlyingType value;
};

}  // namespace trt_allreduce::kernels