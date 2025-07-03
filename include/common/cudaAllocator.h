/**
 * @file cudaAllocator.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CudaAllocator class for managing GPU memory allocation in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <unordered_map>

#include "common/allocator.h"
#include "runtime/bufferManager.h"

namespace tensorrt_llm {

namespace common {

class CudaAllocator : public IAllocator {
   public:
    explicit CudaAllocator(runtime::BufferManager bufferManager);

    ~CudaAllocator() override = default;

    void free(void** ptr) override;

   protected:
    bool contains(void const* ptr) const override { return mPointerMapping.find(ptr) != mPointerMapping.end(); }

    ReallocType reallocType(void const* ptr, size_t size) const override;

    void* malloc(size_t size, bool setZero) override;

    void memSet(void* ptr, int val, size_t size) override;

   private:
    runtime::BufferManager mBufferManager;
    std::unordered_map<void const*, runtime::BufferManager::IBufferPtr> mPointerMapping{};
};

}  // namespace common
}  // namespace tensorrt_llm
