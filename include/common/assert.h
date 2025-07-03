/**
 * @file assert.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Assertion macros for TensorRT-LLM runtime.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <string>

#include "common/stringUtils.h"
#include "common/tllmException.h"

namespace tensorrt_llm::common {
[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "") {
    throw TllmException(file, line, fmtstr("[TensorRT-LLM][ERROR] Assertion failed: %s", info.c_str()));
}

}  // namespace tensorrt_llm::common

#if defined(_WIN32)
#define TLLM_LIKELY(x) (__assume((x) == 1), (x))
#else
#define TLLM_LIKELY(x) __builtin_expect((x), 1)
#endif

#define TLLM_CHECK(val)                                                                                                      \
    do {                                                                                                                     \
        TLLM_LIKELY(static_cast<bool>(val)) ? ((void)0) : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, #val); \
    } while (0)

#define TLLM_CHECK_WITH_INFO(val, info)                                                                                      \
    do {                                                                                                                     \
        TLLM_LIKELY(static_cast<bool>(val)) ? ((void)0) : tensorrt_llm::common::throwRuntimeError(__FILE__, __LINE__, info); \
    } while (0)

#define TLLM_THROW(...)                        \
    do {                                       \
        throw NEW_TLLM_EXCEPTION(__VA_ARGS__); \
    } while (0)

#define TLLM_WRAP(ex) NEW_TLLM_EXCEPTION("%s: %s", tensorrt_llm::common::TllmException::demangle(typeid(ex).name()).c_str(), ex.what())
