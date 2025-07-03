/**
 * @file allocator.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Interface for memory allocation and management in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cassert>
#include <string>

#include "common/logger.h"

namespace tensorrt_llm {
namespace common {

enum class ReallocType {
    INCREASE,
    REUSE,
    DECREASE,
};

class IAllocator {
   public:
    virtual ~IAllocator() = default;

    // no copying
    IAllocator(const IAllocator&) = delete;
    IAllocator& operator=(const IAllocator&) = delete;

    template <typename T>
    [[nodiscard]] T* reMalloc(T* ptr, size_t sizeBytes, const bool setZero = true) {
        TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
        // TODO martinma: why do we need this size extension?
        auto const sizeAligned = ((sizeBytes + 31) / 32) * 32;  // make the buffer align with 32 bytes
        if (contains(ptr)) {
            auto const realloc = reallocType(ptr, sizeAligned);
            if (realloc == ReallocType::INCREASE) {
                TLLM_LOG_DEBUG("ReMalloc the buffer %p since it is too small.", ptr);
                free(&ptr);
                return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
            } else if (realloc == ReallocType::DECREASE) {
                TLLM_LOG_DEBUG("ReMalloc the buffer %p to release unused memory to memory pools.", ptr);
                free(&ptr);
                return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
            } else {
                assert(realloc == ReallocType::REUSE);
                TLLM_LOG_DEBUG("Reuse original buffer %p with size %d and do nothing for reMalloc.", ptr, sizeAligned);
                if (setZero) {
                    memSet(ptr, 0, sizeAligned);
                }
                return ptr;
            }
        } else {
            TLLM_LOG_DEBUG("Cannot find buffer %p, mallocing new one.", ptr);
            return reinterpret_cast<T*>(malloc(sizeAligned, setZero));
        }
    }

    virtual void free(void** ptr) = 0;

    template <typename T>
    void free(T** ptr) {
        free(reinterpret_cast<void**>(ptr));
    }

   protected:
    IAllocator() = default;

    virtual void* malloc(std::size_t size, bool setZero) = 0;
    virtual void memSet(void* ptr, int val, std::size_t size) = 0;

    virtual bool contains(void const* ptr) const = 0;
    virtual ReallocType reallocType(void const* ptr, size_t size) const = 0;
};

}  // namespace common

}  // namespace tensorrt_llm
