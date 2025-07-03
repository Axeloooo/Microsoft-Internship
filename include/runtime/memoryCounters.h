/**
 * @file memoryCounters.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief MemoryCounters class for tracking memory usage in TensorRT-LLM runtime.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>

#include "common/assert.h"
#include "runtime/iBuffer.h"

namespace tensorrt_llm::runtime {

class MemoryCounters {
   public:
    using SizeType = std::size_t;
    using DiffType = std::ptrdiff_t;

    MemoryCounters() = default;

    [[nodiscard]] SizeType getGpu() const { return mGpu; }

    [[nodiscard]] SizeType getCpu() const { return mCpu; }

    [[nodiscard]] SizeType getPinned() const { return mPinned; }

    [[nodiscard]] DiffType getGpuDiff() const { return mGpuDiff; }

    [[nodiscard]] DiffType getCpuDiff() const { return mCpuDiff; }

    [[nodiscard]] DiffType getPinnedDiff() const { return mPinnedDiff; }

    template <MemoryType T>
    void allocate(SizeType size) {
        auto const sizeDiff = static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU) {
            mGpu += size;
            mGpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kCPU) {
            mCpu += size;
            mCpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kPINNED) {
            mPinned += size;
            mPinnedDiff = sizeDiff;
        } else {
            TLLM_THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    void allocate(MemoryType memoryType, SizeType size);

    template <MemoryType T>
    void deallocate(SizeType size) {
        auto const sizeDiff = -static_cast<DiffType>(size);
        if constexpr (T == MemoryType::kGPU) {
            mGpu -= std::min(size, mGpu);
            mGpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kCPU) {
            mCpu -= std::min(size, mCpu);
            mCpuDiff = sizeDiff;
        } else if constexpr (T == MemoryType::kPINNED) {
            mPinned -= std::min(size, mPinned);
            mPinnedDiff = sizeDiff;
        } else {
            TLLM_THROW("Unknown memory type: %s", MemoryTypeString<T>::value);
        }
    }

    void deallocate(MemoryType memoryType, SizeType size);

    static MemoryCounters& getInstance() { return mInstance; }

    static std::string bytesToString(SizeType bytes, int precision = 2);

    static std::string bytesToString(DiffType bytes, int precision = 2);

   private:
    SizeType mGpu{}, mCpu{}, mPinned{};
    DiffType mGpuDiff{}, mCpuDiff{}, mPinnedDiff{};
    static thread_local MemoryCounters mInstance;
};

}  // namespace tensorrt_llm::runtime
