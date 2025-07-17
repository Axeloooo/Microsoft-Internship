/**
 * @file ub_allocator.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Header file for user buffer allocation in the allreduce framework.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once
#include <vector>

#include "runtime/worldConfig.h"
#if ENABLE_MULTI_DEVICE
#include "common/assert.h"
#include "kernels/userbuffers/userbuffers.h"
#endif

namespace trt_allreduce::runtime::ub {
static auto constexpr tensor_prefix = "allreduce_ub_";

struct UBBuffer {
    void* addr;
    int handle;
    size_t size;

    UBBuffer(void* a = nullptr, int h = -1, size_t s = 0) : addr(a), handle(h), size(s) {}

    [[nodiscard]] bool invalid() const { return (addr == nullptr) || (handle == -1) || (size == 0); }
};
#if ENABLE_MULTI_DEVICE
class UserBufferAllocator {
   public:
    static UserBufferAllocator& Instance();

    UserBufferAllocator() = default;

    void initialize(trt_allreduce::runtime::WorldConfig const& world_config);
    bool is_initialized();
    UBBuffer allocate(size_t bytes);
    void deallocate(void* addr);
    UBBuffer get(int idx);
    communicator* comm();

   private:
    UBBuffer register_ub_buffer(size_t bytes);

    communicator* ub_comm_;
    std::vector<UBBuffer> buffers_;
    bool is_initialized_;
    trt_allreduce::runtime::WorldConfig world_config_;
};
#else
using communicator = void;
#endif
};  // namespace trt_allreduce::runtime::ub