/**
 * @file cudaMemPool.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CUDA memory pool class definition
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <memory>

/// @brief Forward declaration of cudaMemPool_t to avoid including "driver_types.h"
struct CUmemPoolHandle_st;
using cudaMemPool_t = CUmemPoolHandle_st*;

namespace trt_allreduce::runtime {

class CudaMemPool {
   public:
    explicit CudaMemPool(cudaMemPool_t pool);

    /// @brief Gets the amount of reserved memory in the memory pool stream, WITHOUT synchronizing.
    [[nodiscard]] std::size_t memoryPoolReserved() const;

    /// @brief Gets the amount of used memory in the memory pool, WITHOUT synchronizing.
    [[nodiscard]] std::size_t memoryPoolUsed() const;

    /// @brief Gets the amount of free memory in the memory pool, WITHOUT synchronizing.
    [[nodiscard]] std::size_t memoryPoolFree() const;

    /// @brief Hints the driver to trim the pool. Does not guarantee that the amount of reserved memory will actually
    /// decrease, only guarantees that this amount after trimming will be larger than the provided size.
    void memoryPoolTrimTo(std::size_t size);

    /// @brief Returns the underlying cudaMemPool_t for usage by CUDA APIs.
    [[nodiscard]] cudaMemPool_t getPool() const;

    /// @brief Gets or initializes and gets the primary memory pool for the provided device ID if it was successfully
    /// initialized, nullptr otherwise.
    static std::shared_ptr<trt_allreduce::runtime::CudaMemPool> getPrimaryPoolForDevice(int deviceId);

    /// @brief Returns a value indicating whether memory pools are supported on the device.
    /// @details Memory pools depend on the presence of the UVM driver. On some systems, the UVM driver is explicitly
    /// disabled.
    static bool supportsMemoryPool(int deviceId);

   private:
    class Deleter {
       public:
        void operator()(cudaMemPool_t pool) const;
    };

    using PoolPtr = std::unique_ptr<std::remove_pointer_t<cudaMemPool_t>, Deleter>;

    PoolPtr mPool;
};

}  // namespace trt_allreduce::runtime