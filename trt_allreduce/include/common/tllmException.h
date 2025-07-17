/**
 * @file tllmException.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for exception handling
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#define NEW_TLLM_EXCEPTION(...) trt_allreduce::common::TllmException(__FILE__, __LINE__, trt_allreduce::common::fmtstr(__VA_ARGS__).c_str())

namespace trt_allreduce::common {

class TllmException : public std::runtime_error {
   public:
    static auto constexpr MAX_FRAMES = 128;

    explicit TllmException(char const* file, std::size_t line, char const* msg);

    ~TllmException() noexcept override;

    [[nodiscard]] std::string getTrace() const;

    static std::string demangle(char const* name);

   private:
    std::array<void*, MAX_FRAMES> mCallstack{};
    int mNbFrames;
};

}  // namespace trt_allreduce::common
