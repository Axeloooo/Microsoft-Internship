/**
 * @file userbuffersManager.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Implementation of user buffer management for NVIDIA's TensorRT All-Reduce.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "kernels/userbuffers/userbuffersManager.h"

namespace trt_allreduce::runtime::ub {

void UserBufferDeleter::operator()(void* ptr) { UserBuffersManager::get_instance().release_buffer(ptr); }

UserBuffersManager& UserBuffersManager::get_instance() {
    static UserBuffersManager allocator;
    return allocator;
}

void UserBuffersManager::initialize(int64_t tp_size, int64_t pp_size, int64_t cp_size, int64_t rank, int64_t gpus_per_node, int64_t buffer_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    trt_allreduce::runtime::WorldConfig world_config(tp_size, pp_size, cp_size, rank, gpus_per_node);
    trt_allreduce::runtime::ub::ub_initialize(world_config);
    TLLM_CHECK(trt_allreduce::runtime::ub::ub_is_initialized());
    buffer_size_ = buffer_size;
}

std::pair<UBBufferPtr, UBBuffer> UserBuffersManager::allocate_userbuffers(int64_t buffer_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    TLLM_CHECK(buffer_size <= buffer_size_);

    // Check for all unused buffers
    int i = 0;
    for (auto& buffer : buffers_) {
        if (buffer.second) {
            i++;
            continue;
        }
        buffer.second = true;
        TLLM_LOG_DEBUG("Reusing buffer %d", i);
        return std::make_pair(std::unique_ptr<void, UserBufferDeleter>(buffer.first.addr), buffer.first);
    }

    auto new_ub = trt_allreduce::runtime::ub::ub_allocate(buffer_size_);
    TLLM_CHECK(!new_ub.invalid());
    buffers_.push_back({new_ub, true});
    TLLM_LOG_DEBUG("Creating new buffer %d", static_cast<int>(buffers_.size() - 1));
    return std::make_pair(std::unique_ptr<void, UserBufferDeleter>(new_ub.addr), new_ub);
}

void UserBuffersManager::release_buffer(void* addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto buffer_iter = std::find_if(buffers_.begin(), buffers_.end(), [addr](auto const& buffer) { return buffer.first.addr == addr; });
    // The UB should be assigned to a tensor
    TLLM_CHECK(buffer_iter != buffers_.end());
    TLLM_CHECK(buffer_iter->second);
    TLLM_CHECK(!buffer_iter->first.invalid());
    TLLM_LOG_DEBUG("Releasing buffer %d", static_cast<int>(std::distance(buffers_.begin(), buffer_iter)));
    buffer_iter->second = false;
}

trt_allreduce::runtime::ub::UBBuffer UserBuffersManager::search_buffer(void* addr) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto buffer_iter = std::find_if(buffers_.begin(), buffers_.end(), [addr](auto const& buffer) { return buffer.first.addr == addr; });
    if (buffer_iter == buffers_.end()) {
        return trt_allreduce::runtime::ub::UBBuffer();
    }
    return buffer_iter->first;
}

trt_allreduce::runtime::ub::communicator* UserBuffersManager::comm() { return trt_allreduce::runtime::ub::ub_comm(); }

void initialize_userbuffers_manager(int64_t tp_size, int64_t pp_size, int64_t cp_size, int64_t rank, int64_t gpus_per_node, int64_t buffer_size) {
    UserBuffersManager::get_instance().initialize(tp_size, pp_size, cp_size, rank, gpus_per_node, buffer_size);
}

}  // namespace trt_allreduce::runtime::ub
