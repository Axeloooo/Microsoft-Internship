/**
 * @file tllmException.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Exception class for TensorRT-LLM, providing detailed error messages and call stack tracing.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>

#include "common/stringUtils.h"

#define NEW_TLLM_EXCEPTION(...) tensorrt_llm::common::TllmException(__FILE__, __LINE__, tensorrt_llm::common::fmtstr(__VA_ARGS__))

namespace tensorrt_llm::common {

class TllmException : public std::runtime_error {
   public:
    static auto constexpr MAX_FRAMES = 128;

    explicit TllmException(char const* file, std::size_t line, std::string const& msg);

    ~TllmException() noexcept override;

    [[nodiscard]] std::string getTrace() const;

    static std::string demangle(char const* name);

   private:
    std::array<void*, MAX_FRAMES> mCallstack{};
    int mNbFrames;
};

}  // namespace tensorrt_llm::common
