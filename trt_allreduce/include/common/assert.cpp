/**
 * @file assert.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Implementation of debug configuration checks.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "common/assert.h"

namespace {

bool initCheckDebug() {
    auto constexpr kDebugEnabled = "TLLM_DEBUG_MODE";
    auto const debugEnabled = std::getenv(kDebugEnabled);
    return debugEnabled && debugEnabled[0] == '1';
}
}  // namespace

bool DebugConfig::isCheckDebugEnabled() {
    static bool const debugEnabled = initCheckDebug();
    return debugEnabled;
}