/**
 * @file userbuffersManager.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief UserBuffersManager class declaration
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once
#include <memory>
#include <mutex>
#include <vector>

#include "kernels/userbuffers/ub_interface.h"
#include "runtime/worldConfig.h"

namespace trt_allreduce::runtime::ub {

class UserBufferDeleter {
   public:
    void operator()(void* ptr);
};

using UBBufferPtr = std::unique_ptr<void, UserBufferDeleter>;
using UBBufferInfo = std::pair<trt_allreduce::runtime::ub::UBBuffer, bool>;

class UserBuffersManager {
   public:
    static UserBuffersManager& get_instance();
    UserBuffersManager() = default;

    //! @brief Initialize the userbuffers manager.
    //! @param tp_size Tensor parallel size.
    //! @param pp_size Pipeline parallel size.
    //! @param cp_size Compute parallel size.
    //! @param rank The rank of the current GPU.
    //! @param gpus_per_node The number of GPUs per node.
    //! @param buffer_size The size of the buffer to allocate. All buffers allocated by this manager will have this
    //! size.
    void initialize(int64_t tp_size, int64_t pp_size, int64_t cp_size, int64_t rank, int64_t gpus_per_node, int64_t buffer_size);

    //! @brief Create a UB tensor from the given shape, strides and data type. The function will choose available UB
    //! buffer or create a new one if no available buffer is found.
    //! @param buffer_size The size of the buffer to allocate.
    //! @return A unique_ptr to the buffer and the UBBuffer object.
    //! @note Do not manually call release_buffer with the buffer address in trt_allreduce::runtime::ub::UBBuffer
    std::pair<UBBufferPtr, trt_allreduce::runtime::ub::UBBuffer> allocate_userbuffers(int64_t buffer_size);

    //! @brief Search the buffer from the list of buffers.
    //! @param addr The address of the buffer to search for.
    //! @return The buffer and whether it is assigned to a tensor. If not found, the UBBuffer is invalid.
    trt_allreduce::runtime::ub::UBBuffer search_buffer(void* addr);

    //! @brief Get the communicator.
    //! @return The communicator.
    trt_allreduce::runtime::ub::communicator* comm();

    //! @brief Release the buffer. It does not deallocate the buffer, but just release it to the pool.
    //! @param addr The address of the buffer to release.
    void release_buffer(void* addr);

   private:
    std::mutex mutex_;
    std::vector<UBBufferInfo> buffers_;
    int64_t buffer_size_;
};

void initialize_userbuffers_manager(int64_t tp_size, int64_t pp_size, int64_t cp_size, int64_t rank, int64_t gpus_per_node, int64_t buffer_size);

}  // namespace trt_allreduce::runtime::ub