/**
 * @file logger.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Logger class for TensorRT-LLM, handling logging levels and formatting.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "common/logger.h"

#include <cuda_runtime.h>

#include "common/tllmException.h"

namespace trt_allreduce::common {

Logger::Logger() {
    char* isFirstRankOnlyChar = std::getenv("TLLM_LOG_FIRST_RANK_ONLY");
    bool isFirstRankOnly = (isFirstRankOnlyChar != nullptr && std::string(isFirstRankOnlyChar) == "ON");

    int deviceId;
    cudaGetDevice(&deviceId);

    char* levelName = std::getenv("TLLM_LOG_LEVEL");
    if (levelName != nullptr) {
        std::map<std::string, Level> nameToLevel = {
            {"TRACE", TRACE}, {"DEBUG", DEBUG}, {"INFO", INFO}, {"WARNING", WARNING}, {"ERROR", ERROR},
        };
        auto level = nameToLevel.find(levelName);
        // If TLLM_LOG_FIRST_RANK_ONLY=ON, set LOG LEVEL of other device to ERROR
        if (isFirstRankOnly && deviceId != 0) {
            level = nameToLevel.find("ERROR");
        }
        if (level != nameToLevel.end()) {
            setLevel(level->second);
        } else {
            fprintf(stderr,
                    "[TensorRT-LLM][WARNING] Invalid logger level TLLM_LOG_LEVEL=%s. "
                    "Ignore the environment variable and use a default "
                    "logging level.\n",
                    levelName);
            levelName = nullptr;
        }
    }
}

void Logger::log(std::exception const& ex, Logger::Level level) { log(level, "%s: %s", TllmException::demangle(typeid(ex).name()).c_str(), ex.what()); }

}  // namespace trt_allreduce::common
