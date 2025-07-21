/**
 * @file stringUtils.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for string manipulation
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "common/stringUtils.h"

#include <cerrno>
#include <cstdarg>
#include <cstring>
#include <iostream>
#include <string>

#include "common/assert.h"

namespace trt_allreduce::common {

void fmtstr_(char const* format, fmtstr_allocator alloc, void* target, va_list args) {
    va_list args0;
    va_copy(args0, args);

    size_t constexpr init_size = 2048;
    char fixed_buffer[init_size];
    auto const size = std::vsnprintf(fixed_buffer, init_size, format, args0);
    TLLM_CHECK_WITH_INFO(size >= 0, std::string(std::strerror(errno)));
    if (size == 0) {
        return;
    }

    auto* memory = alloc(target, size);

    if (static_cast<size_t>(size) < init_size) {
        std::memcpy(memory, fixed_buffer, size + 1);
    } else {
        auto const size2 = std::vsnprintf(memory, size + 1, format, args);
        TLLM_CHECK_WITH_INFO(size2 == size, std::string(std::strerror(errno)));
    }
}

std::unordered_set<std::string> str2set(std::string const& input, char delimiter) {
    std::unordered_set<std::string> values;
    if (!input.empty()) {
        std::stringstream valStream(input);
        std::string val;
        while (std::getline(valStream, val, delimiter)) {
            if (!val.empty()) {
                values.insert(val);
            }
        }
    }
    return values;
};

}  // namespace trt_allreduce::common