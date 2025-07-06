/**
 * @file stringUtils.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief String utility functions for TensorRT-LLM, including formatted string creation.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "common/stringUtils.h"

#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <string>

#include "common/assert.h"

namespace trt_allreduce::common {

namespace {
std::string vformat(char const* fmt, va_list args) {
    va_list args0;
    va_copy(args0, args);
    auto const size = vsnprintf(nullptr, 0, fmt, args0);
    if (size <= 0) return "";

    std::string stringBuf(size, char{});
    auto const size2 = std::vsnprintf(&stringBuf[0], size + 1, fmt, args);

    TLLM_CHECK_WITH_INFO(size2 == size, std::strerror(errno));

    return stringBuf;
}

}  // namespace

std::string fmtstr(char const* format, ...) {
    va_list args;
    va_start(args, format);
    std::string result = vformat(format, args);
    va_end(args);
    return result;
};

}  // namespace trt_allreduce::common
