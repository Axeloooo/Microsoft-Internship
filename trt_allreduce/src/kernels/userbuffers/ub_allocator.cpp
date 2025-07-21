/**
 * @file ub_allocator.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Implementation of user buffer allocation for NVIDIA's TensorRT All-Reduce.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "kernels/userbuffers/ub_allocator.h"

namespace trt_allreduce::runtime::ub {
#if ENABLE_MULTI_DEVICE
UserBufferAllocator& UserBufferAllocator::Instance() {
    static UserBufferAllocator _;
    return _;
}

void UserBufferAllocator::initialize(trt_allreduce::runtime::WorldConfig const& world_config) {
    if (!is_initialized()) {
        ub_comm_ = nullptr;
        world_config_ = world_config;
        create_communicator_grouped2(&ub_comm_, world_config_);
        TLLM_CHECK(ub_comm_ != nullptr);
        is_initialized_ = true;
    }
}

bool UserBufferAllocator::is_initialized() { return is_initialized_; }

UBBuffer UserBufferAllocator::register_ub_buffer(size_t bytes) {
    TLLM_CHECK(is_initialized());
    void* addr = nullptr;
    int handle = -1;
    handle = register_user_buffer_collective((void**)&addr, bytes, ub_comm_);
    return {addr, handle, bytes};
}

UBBuffer UserBufferAllocator::allocate(size_t bytes) {
    TLLM_CHECK(is_initialized());
    auto ub_buffer = register_ub_buffer(bytes);
    TLLM_CHECK(!ub_buffer.invalid());
    buffers_.push_back(ub_buffer);
    return ub_buffer;
}

void UserBufferAllocator::deallocate(void* addr) {}

UBBuffer UserBufferAllocator::get(int idx) {
    TLLM_CHECK(is_initialized() && idx < buffers_.size() && !buffers_[idx].invalid());
    return buffers_[idx];
}

communicator* UserBufferAllocator::comm() {
    TLLM_CHECK(is_initialized());
    return ub_comm_;
}
#endif
};  // namespace trt_allreduce::runtime::ub
