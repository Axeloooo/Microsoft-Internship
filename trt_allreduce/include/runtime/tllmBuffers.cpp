/**
 * @file tllmBuffers.cpp
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Tensor buffer management for TRT-LLM
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#include "runtime/tllmBuffers.h"

namespace trt_allreduce::runtime {
template <typename TAllocator>
typename PoolAllocator<TAllocator>::PoolType& PoolAllocator<TAllocator>::getPool() {
    static PoolType pool;
    return pool;
}

MulticastTensorView::MulticastTensorView(std::weak_ptr<MulticastTensor> const& tensor, ViewType viewType)
    : mTensor(tensor), mViewType(viewType), mDims(mTensor.lock()->getShape()) {}

MulticastTensorView::MulticastTensorView(MulticastTensorView&& other) noexcept
    : mTensor(std::move(other.mTensor)), mViewType(other.mViewType), mDims(mTensor.lock()->getShape()) {}

MulticastTensorView& MulticastTensorView::operator=(MulticastTensorView&& other) noexcept {
    if (this != &other) {
        // Reset tensor.
        mTensor.reset();
        mTensor.swap(other.mTensor);
        mViewType = other.mViewType;
        mDims = mTensor.lock()->getShape();
    }
    return *this;
}

std::shared_ptr<MulticastBuffer> MulticastTensorView::lock() const {
    auto sp = mTensor.lock();
    TLLM_CHECK(sp != nullptr);
    return sp;
}

///////////////////////////////////////
// MulticastTensorView ITensor methods
///////////////////////////////////////
nvinfer1::Dims const& MulticastTensorView::getShape() const { return mDims; }

void MulticastTensorView::reshape(nvinfer1::Dims const& dims) {
    auto new_size = nonNegative(volume(dims));
    if (new_size > getCapacity()) {
        TLLM_THROW("MulticastTensorView::reshape() cannot be larger than origin tensor.");
    }
    mDims = dims;
}

///////////////////////////////////////
// MulticastTensorView IBuffer methods
///////////////////////////////////////
void* MulticastTensorView::_data() const {
    switch (mViewType) {
        case ViewType::kUNICAST:
            return lock()->data();
        case ViewType::kMULTICAST:
            return lock()->dataMC();
        case ViewType::kIPC_LIST:
            return lock()->dataIpcList();
    }
    TLLM_THROW("Invalid mViewType");
    return nullptr;
}

std::size_t MulticastTensorView::getSize() const { return lock()->getSize(); }

std::size_t MulticastTensorView::getCapacity() const { return lock()->getCapacity(); }

nvinfer1::DataType MulticastTensorView::getDataType() const { return lock()->getDataType(); }

MemoryType MulticastTensorView::getMemoryType() const { return lock()->getMemoryType(); }

// explicit instantiations
template class PoolAllocator<PinnedAllocator>;
}  // namespace trt_allreduce::runtime