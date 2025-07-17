/**
 * @file memoryCounters.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Memory counters implementation for TRT-LLM
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "runtime/memoryCounters.h"

#include <array>
#include <cmath>

#include "common/stringUtils.h"

namespace tc = trt_allreduce::common;

namespace {

auto constexpr kByteUnits = std::array{"B", "KB", "MB", "GB", "TB", "PB", "EB"};

std::string doubleBytesToString(double bytes, int precision) {
    std::uint32_t unitIdx{0};

    while (std::abs(bytes) >= 1024.0 && unitIdx < kByteUnits.size() - 1) {
        bytes /= 1024.0;
        ++unitIdx;
    }
    auto const format = "%." + std::to_string(precision) + "f %s";
    return tc::fmtstr(format.c_str(), bytes, kByteUnits[unitIdx]);
}

}  // namespace

namespace trt_allreduce::runtime {
std::string MemoryCounters::bytesToString(SizeType32 bytes, int precision) { return doubleBytesToString(static_cast<double>(bytes), precision); }

std::string MemoryCounters::bytesToString(DiffType bytes, int precision) { return doubleBytesToString(static_cast<double>(bytes), precision); }

std::string MemoryCounters::toString() const {
    return trt_allreduce::common::fmtstr("[MemUsage] GPU %s, CPU %s, Pinned %s", bytesToString(this->getGpu()).c_str(), bytesToString(this->getCpu()).c_str(),
                                         bytesToString(this->getPinned()).c_str());
}

void MemoryCounters::allocate(MemoryType memoryType, MemoryCounters::SizeType32 size) {
    switch (memoryType) {
        case MemoryType::kGPU:
            allocate<MemoryType::kGPU>(size);
            break;
        case MemoryType::kCPU:
            allocate<MemoryType::kCPU>(size);
            break;
        case MemoryType::kPINNED:
            allocate<MemoryType::kPINNED>(size);
            break;
        case MemoryType::kPINNEDPOOL:
            allocate<MemoryType::kPINNEDPOOL>(size);
            break;
        default:
            TLLM_THROW("Unknown memory type");
    }
}

void MemoryCounters::deallocate(MemoryType memoryType, MemoryCounters::SizeType32 size) {
    switch (memoryType) {
        case MemoryType::kGPU:
            deallocate<MemoryType::kGPU>(size);
            break;
        case MemoryType::kCPU:
            deallocate<MemoryType::kCPU>(size);
            break;
        case MemoryType::kPINNED:
            deallocate<MemoryType::kPINNED>(size);
            break;
        case MemoryType::kPINNEDPOOL:
            deallocate<MemoryType::kPINNEDPOOL>(size);
            break;
        default:
            TLLM_THROW("Unknown memory type");
    }
}

MemoryCounters& MemoryCounters::getInstance() {
    static MemoryCounters mInstance;
    return mInstance;
}
}  // namespace trt_allreduce::runtime