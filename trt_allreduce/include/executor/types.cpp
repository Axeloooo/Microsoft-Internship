/**
 * @file types.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Implementation of operator overloads for CapacitySchedulerPolicy and ContextChunkingPolicy.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "executor/types.h"

#include <iostream>

namespace trt_allreduce::executor {
std::ostream& operator<<(std::ostream& os, CapacitySchedulerPolicy policy) {
    switch (policy) {
        case CapacitySchedulerPolicy::kMAX_UTILIZATION:
            os << "MAX_UTILIZATION";
            break;
        case CapacitySchedulerPolicy::kGUARANTEED_NO_EVICT:
            os << "GUARANTEED_NO_EVICT";
            break;
        case CapacitySchedulerPolicy::kSTATIC_BATCH:
            os << "STATIC_BATCH";
            break;
    }
    return os;
}

std::ostream& operator<<(std::ostream& os, ContextChunkingPolicy policy) {
    switch (policy) {
        case ContextChunkingPolicy::kEQUAL_PROGRESS:
            os << "EQUAL_PROGRESS";
            break;
        case ContextChunkingPolicy::kFIRST_COME_FIRST_SERVED:
            os << "FIRST_COME_FIRST_SERVED";
            break;
    }
    return os;
}
}  // namespace trt_allreduce::executor