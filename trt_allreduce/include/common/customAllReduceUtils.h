/**
 * @file customAllReduceUtils.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for custom all-reduce operations
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cstddef>

#include "common/envUtils.h"

namespace trt_allreduce::utils::customAllReduceUtils {

constexpr size_t NUM_POINTERS_PER_RANK = 7;

// WARNING: MUST BE KEPT IN SYNC with trt_allreduce/plugin/plugin.py
inline size_t getMaxRequiredWorkspaceSize(int worldSize) noexcept {
    if (common::getEnvForceDeterministicAllReduce()) {
        return common::getEnvAllReduceWorkspaceSize();
    }
    if (worldSize <= 2) {
        return 16 * 1000 * 1000;
    }
    return 8 * 1000 * 1000;
}

}  // namespace trt_allreduce::utils::customAllReduceUtils