/**
 * @file common.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Common definitions and type aliases for the runtime module.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>

namespace tensorrt_llm::runtime {

// typedefs
// Note that we use unsigned size types as recommended by TensorRT:
// https://github.com/NVIDIA/TensorRT/blob/main/CODING-GUIDELINES.md#signed-vs-unsigned-integers
using SizeType = std::int32_t;

// Token ID type
using TokenIdType = std::int32_t;

template <typename T>
using StringPtrMap = std::unordered_map<std::string, std::shared_ptr<T>>;

}  // namespace tensorrt_llm::runtime
