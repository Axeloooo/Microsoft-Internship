/**
 * @file bufferManager.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief A helper class for managing memory on host and device.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <NvInferRuntime.h>

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/assert.h"
#include "runtime/cudaStream.h"
#include "runtime/iBuffer.h"
#include "runtime/iTensor.h"

namespace tensorrt_llm::runtime {
//! \brief A helper class for managing memory on host and device.
class BufferManager {
   public:
    using IBufferPtr = IBuffer::UniquePtr;

    using ITensorPtr = ITensor::UniquePtr;

    using CudaStreamPtr = std::shared_ptr<CudaStream>;

    //! \brief Construct a BufferManager.
    //!
    //! \param[in] cudaStream The cuda stream to use for all operations on GPU (allocation, de-allocation, copying,
    //! etc.).
    explicit BufferManager(CudaStreamPtr stream);

    static auto constexpr kBYTE_TYPE = nvinfer1::DataType::kUINT8;

    //! \brief Allocates an `IBuffer` of the given size on the GPU.
    [[nodiscard]] IBufferPtr gpu(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Allocates an `ITensor` of the given dimensions on the GPU.
    [[nodiscard]] ITensorPtr gpu(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Allocates an `IBuffer` of the given size on the CPU.
    [[nodiscard]] static IBufferPtr cpu(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `ITensor` of the given dimensions on the CPU.
    [[nodiscard]] static ITensorPtr cpu(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `IBuffer` of the given size on the CPU.
    [[nodiscard]] static IBufferPtr pinned(std::size_t size, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates a pinned `ITensor` of the given dimensions on the CPU.
    [[nodiscard]] static ITensorPtr pinned(nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE);

    //! \brief Allocates an `IBuffer` of the given size and memory type.
    [[nodiscard]] IBufferPtr allocate(MemoryType memoryType, std::size_t size, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Allocates an `ITensor` of the given dimensions and memory type.
    [[nodiscard]] ITensorPtr allocate(MemoryType memoryType, nvinfer1::Dims dims, nvinfer1::DataType type = kBYTE_TYPE) const;

    //! \brief Create an empty `IBuffer` of the given memory type. It may be resized later.
    [[nodiscard]] IBufferPtr emptyBuffer(MemoryType memoryType, nvinfer1::DataType type = kBYTE_TYPE) const { return allocate(memoryType, 0, type); }

    //! \brief Create an empty `ITensor` of the given memory type. It may be reshaped later.
    [[nodiscard]] ITensorPtr emptyTensor(MemoryType memoryType, nvinfer1::DataType type = kBYTE_TYPE) const { return allocate(memoryType, ITensor::makeShape({}), type); }

    //! \brief Set the contents of the given `buffer` to zero.
    void setZero(IBuffer& buffer) const;

    //! \brief Copy `src` to `dst`.
    void copy(void const* src, IBuffer& dst, MemoryType srcType) const;

    //! \brief Copy `src` to `dst`.
    void copy(IBuffer const& src, void* dst, MemoryType dstType) const;

    //! \brief Copy `src` to `dst`.
    void copy(void const* src, IBuffer& dst) const { return copy(src, dst, IBuffer::memoryType(src)); }

    //! \brief Copy `src` to `dst`.
    void copy(IBuffer const& src, void* dst) const { return copy(src, dst, IBuffer::memoryType(dst)); }

    //! \brief Copy `src` to `dst`.
    void copy(IBuffer const& src, IBuffer& dst) const;

    //! \brief Copy `src` into a new `IBuffer` with a potentially different memory type.
    [[nodiscard]] IBufferPtr copyFrom(IBuffer const& src, MemoryType memoryType) const;

    //! \brief Copy `src` into a new `ITensor` with a potentially different memory type.
    [[nodiscard]] ITensorPtr copyFrom(ITensor const& src, MemoryType memoryType) const;

    //! \brief Copy `src` into a new `IBuffer` with a potentially different memory type.
    template <typename T>
    [[nodiscard]] IBufferPtr copyFrom(std::vector<T> const& src, MemoryType memoryType) const {
        auto buffer = allocate(memoryType, src.size(), TRTDataType<std::remove_cv_t<T>>::value);
        copy(src.data(), *buffer);
        return buffer;
    }

    //! \brief Copy `src` into a new `ITensor` with a potentially different memory type.
    template <typename T>
    [[nodiscard]] ITensorPtr copyFrom(T* src, nvinfer1::Dims dims, MemoryType memoryType) const {
        auto buffer = allocate(memoryType, dims, TRTDataType<std::remove_cv_t<T>>::value);
        copy(src, *buffer);
        return buffer;
    }

    //! \brief Copy `src` into a new `ITensor` with a potentially different memory type.
    template <typename T>
    [[nodiscard]] ITensorPtr copyFrom(std::vector<T> const& src, nvinfer1::Dims dims, MemoryType memoryType) const {
        TLLM_CHECK_WITH_INFO(src.size() == ITensor::volumeNonNegative(dims),
                             common::fmtstr("[TensorRT-LLM][ERROR] Incompatible size %lu and dims %s", src.size(), ITensor::toString(dims).c_str()));
        return copyFrom(src.data(), dims, memoryType);
    }

    //! \brief Get the underlying cuda stream.
    [[nodiscard]] CudaStream const& getStream() const;

   private:
    void static initMemoryPool(int device);

    CudaStreamPtr mStream;
};

}  // namespace tensorrt_llm::runtime
