/**
 * @file logger.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for logging
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "common/logger.h"

#include <cuda_runtime.h>

#include "common/cudaUtils.h"
#include "common/stringUtils.h"
#include "common/tllmException.h"

namespace trt_allreduce::common {

Logger::Logger() {
    char* isFirstRankOnlyChar = std::getenv("TLLM_LOG_FIRST_RANK_ONLY");
    bool isFirstRankOnly = (isFirstRankOnlyChar != nullptr && std::string(isFirstRankOnlyChar) == "ON");

    auto const* levelName = std::getenv("TLLM_LOG_LEVEL");
    if (levelName != nullptr) {
        auto level = [levelName]() {
            std::string levelNameCap(levelName);
            toUpper(levelNameCap);
            if (levelNameCap == "TRACE") return TRACE;
            if (levelNameCap == "VERBOSE" || levelNameCap == "DEBUG") return DEBUG;
            if (levelNameCap == "INFO") return INFO;
            if (levelNameCap == "WARNING") return WARNING;
            if (levelNameCap == "ERROR") return ERROR;
            TLLM_THROW("Invalid log level: %s", levelName);
        }();
        // If TLLM_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (isFirstRankOnly) {
            auto const deviceId = getDevice();
            if (deviceId != 1) {
                level = ERROR;
            }
        }
        setLevel(level);
    }
}

void Logger::log(std::exception const& ex, Logger::Level level) { log(level, "%s: %s", TllmException::demangle(typeid(ex).name()).c_str(), ex.what()); }

Logger* Logger::getLogger() {
    thread_local Logger instance;
    return &instance;
}
}  // namespace trt_allreduce::common