/**
 * @file common.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Common definitions and type aliases for the runtime module
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace trt_allreduce::runtime {

#define FMT_DIM "%ld"

// typedefs
// Note that we use signed size types as recommended by TensorRT:
// https://github.com/NVIDIA/TensorRT/blob/main/CODING-GUIDELINES.md#signed-vs-unsigned-integers
using SizeType32 = std::int32_t;
using SizeType64 = std::int64_t;

enum class RequestType : std::int32_t { kCONTEXT = 0, kGENERATION = 1 };

// Token ID type
using TokenIdType = std::int32_t;

using LoraTaskIdType = std::uint64_t;
using TokenExtraIdType = std::uint64_t;
using VecTokenExtraIds = std::vector<TokenExtraIdType>;

struct UniqueToken {
    TokenIdType tokenId;
    TokenExtraIdType tokenExtraId;

    bool operator==(UniqueToken const& other) const noexcept { return (tokenId == other.tokenId && tokenExtraId == other.tokenExtraId); }
};

using VecUniqueTokens = std::vector<UniqueToken>;

template <typename T>
using StringPtrMap = std::unordered_map<std::string, std::shared_ptr<T>>;

}  // namespace trt_allreduce::runtime
