/**
 * @file iStatefulGptDecoder.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Interface for a GPT decoder with support for in-flight batching.
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
#include <utility>
#include <vector>

#include "runtime/cudaStream.h"
#include "runtime/generationInput.h"
#include "runtime/iTensor.h"
#include "runtime/samplingConfig.h"

namespace tensorrt_llm::runtime {

namespace decoder {

class Input {
   public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    explicit Input(TensorPtr logits) : logits{std::move(logits)} { TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor"); }

    // mandatory parameters
    TensorPtr logits;  // [batchSize, maxBeamWidth, vocabSizePadded], on gpu

    // parameters for beam search
    TensorPtr cacheIndirection;  // [batchSize, maxBeamWidth, maxSeqLen] - the k/v cache index for beam search, on gpu
};

class Output {
   public:
    using TensorPtr = std::shared_ptr<ITensor>;

    Output() = default;

    // parameters for beam search
    TensorPtr cacheIndirection;  // [batchSize, maxBeamWidth, maxSeqLen], mandatory in beam search, on gpu
    TensorPtr sequenceLengths;   // [batchSize, maxBeamWidth], mandatory, on gpu
};
}  // namespace decoder

//! GPT decoder class with support for in-flight batching
class IStatefulGptDecoder {
   public:
    using CudaStreamPtr = std::shared_ptr<CudaStream>;
    using TensorPtr = std::shared_ptr<ITensor>;

    //! Setup the decoder before calling `forward()`, also calls reshapeBuffers
    virtual void setup(SizeType maxBatchSize, SizeType maxBeamWidth, SizeType maxSequenceLength, nvinfer1::DataType dtype) = 0;

    //! @brief Initialize the decoder with new batch of inputs.
    virtual void newBatch(GenerationInput const& inputs, SamplingConfig const& samplingConfig) = 0;

    //! @brief Run one step for all requests without blocking the host thread.
    virtual void forwardAsync(decoder::Output& output, decoder::Input const& input) = 0;

    //! @brief Wait for the last call to `forwardAsync` to complete and return whether all sequences have finished.
    virtual bool isFinishedSync() = 0;

    //! @brief Run one step for all requests.
    virtual bool forward(decoder::Output& output, decoder::Input const& input) {
        forwardAsync(output, input);
        return isFinishedSync();
    }

    //! @brief Gather final results for all requests.
    virtual TensorPtr getFinalOutputIds() const = 0;

    //! @returns [batchSize, beamWidth, maxSequenceLength], all token ids, on gpu
    virtual TensorPtr getOutputIds() const = 0;

    //! @returns [batchSize, beamWidth], latests generated tokens (per beam), on gpu
    virtual TensorPtr getNewTokens() const = 0;

    //! @returns [1], number of finished sequences, in pinned host memory
    virtual TensorPtr getNbFinished() const = 0;

   protected:
    IStatefulGptDecoder() = default;
};

}  // namespace tensorrt_llm::runtime
