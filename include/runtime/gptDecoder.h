/**
 * @file gptDecoder.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Interface and implementation for GPT decoder in TensorRT-LLM.
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

#include "common/cudaAllocator.h"
#include "runtime/bufferManager.h"
#include "runtime/decodingInput.h"
#include "runtime/decodingOutput.h"
#include "runtime/samplingConfig.h"

namespace tensorrt_llm {

namespace layers {
// Forward declaration
template <typename T>
class DynamicDecodeLayer;
}  // namespace layers

namespace runtime {

class IGptDecoder {
   public:
    virtual ~IGptDecoder() = default;

    virtual void setup(SamplingConfig const& samplingConfig, size_t batchSize) = 0;

    virtual bool forward(DecodingOutput& output, DecodingInput const& input) = 0;

    virtual void forwardAsync(DecodingOutput& output, DecodingInput const& input) = 0;

    static void gatherTree(ITensor& finalOutputIds, DecodingOutput const& decodingOutput, DecodingInput const& decodingInput, BufferManager const& manager);

    static std::unique_ptr<IGptDecoder> create(nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream);
};

template <typename T>
class GptDecoder : public virtual IGptDecoder {
   public:
    using CudaStreamPtr = BufferManager::CudaStreamPtr;

    GptDecoder(size_t vocabSize, size_t vocabSizePadded, CudaStreamPtr const& stream);

    void setup(SamplingConfig const& samplingConfig, size_t batchSize) override;

    bool forward(DecodingOutput& output, DecodingInput const& input) override;

    void forwardAsync(DecodingOutput& output, DecodingInput const& input) override;

   private:
    BufferManager mManager;
    common::CudaAllocator mAllocator;
    std::shared_ptr<tensorrt_llm::layers::DynamicDecodeLayer<T>> mDynamicDecodeLayer;
};

inline std::unique_ptr<IGptDecoder> IGptDecoder::create(nvinfer1::DataType dtype, size_t vocabSize, size_t vocabSizePadded, BufferManager::CudaStreamPtr const& stream) {
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return std::make_unique<GptDecoder<float>>(vocabSize, vocabSizePadded, stream);
        case nvinfer1::DataType::kHALF:
            return std::make_unique<GptDecoder<half>>(vocabSize, vocabSizePadded, stream);
        default:
            return nullptr;
    }
}
}  // namespace runtime
}  // namespace tensorrt_llm
