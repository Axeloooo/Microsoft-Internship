/**
 * @file cudaAllocator.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief CudaAllocator class for managing GPU memory allocation in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "cudaAllocator.h"

#include <utility>

#include "common/cudaUtils.h"

using namespace tensorrt_llm::common;
namespace tr = tensorrt_llm::runtime;

CudaAllocator::CudaAllocator(tr::BufferManager bufferManager) : mBufferManager(std::move(bufferManager)) {}

ReallocType CudaAllocator::reallocType(void const* ptr, size_t size) const {
    TLLM_CHECK(contains(ptr));
    auto const currentSize = mPointerMapping.at(ptr)->getSize();
    TLLM_LOG_DEBUG("current_buffer_size: %d, original buffer: %p, new buffer: %d", currentSize, ptr, size);
    if (currentSize < size) {
        return ReallocType::INCREASE;
    } else if (currentSize == size) {
        return ReallocType::REUSE;
    } else {
        return ReallocType::DECREASE;
    }
}

void* CudaAllocator::malloc(std::size_t size, bool const setZero) {
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    auto bufferPtr = mBufferManager.gpu(size);
    if (setZero) {
        mBufferManager.setZero(*bufferPtr);
    }
    void* ptr{bufferPtr->data()};
    TLLM_LOG_DEBUG("malloc buffer %p with size %ld", ptr, size);
    mPointerMapping.insert({ptr, std::move(bufferPtr)});
    return ptr;
}

void CudaAllocator::free(void** ptr) {
    TLLM_LOG_DEBUG(__PRETTY_FUNCTION__);
    mPointerMapping.erase(*ptr);
    *ptr = nullptr;
}

void CudaAllocator::memSet(void* ptr, int const val, size_t const size) { check_cuda_error(cudaMemsetAsync(ptr, val, size, mBufferManager.getStream().get())); }
