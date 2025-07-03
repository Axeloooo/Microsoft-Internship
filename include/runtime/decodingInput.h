/**
 * @file decodingInput.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief DecodingInput class for managing inputs to the decoding process in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <memory>

#include "common/cudaUtils.h"
#include "runtime/common.h"
#include "runtime/iTensor.h"

namespace tensorrt_llm::runtime {
class DecodingInput {
   public:
    using TensorPtr = std::shared_ptr<ITensor const>;

    DecodingInput(SizeType maxLength, SizeType batchSize, TensorPtr logits, TensorPtr endIds)
        : step{maxLength}, maxLength{maxLength}, batchSize{batchSize}, logits{std::move(logits)}, endIds{std::move(endIds)} {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->logits), "Invalid logits tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->endIds), "Invalid endIds tensor");
    }

    // mandatory parameters
    SizeType step;
    SizeType maxLength;
    SizeType batchSize;
    TensorPtr logits;  // [batchSize, beamWidth, vocabSizePadded], on gpu
    TensorPtr endIds;  // [batchSize * beamWidth], on gpu

    // optional parameters
    TensorPtr sequenceLimitLength;  // [batchSize], on gpu
    TensorPtr embeddingBias;        // [vocabSizePadded], on gpu
    TensorPtr lengths;              // [batchSize, beamWidth], on gpu
    TensorPtr badWordsList;         // [2, badWordsLength] or [batchSize, 2, badWordsLength], on gpu
    TensorPtr stopWordsList;        // [batchSize, 2, stopWordsLength], on gpu
    TensorPtr noRepeatNgramSize;    // [batchSize], on gpu

    // parameters for beam search
    TensorPtr cacheIndirection;  // [batchSize, beamWidth, maxSeqLen] - the k/v cache index for beam search, on gpu
};

}  // namespace tensorrt_llm::runtime
