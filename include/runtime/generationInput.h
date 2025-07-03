/**
 * @file generationInput.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief GenerationInput class for managing inputs to the generation process in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <optional>
#include <utility>

#include "common/cudaUtils.h"
#include "runtime/common.h"
#include "runtime/iTensor.h"

namespace tensorrt_llm::runtime {

class GenerationInput {
   public:
    using TensorPtr = ITensor::SharedPtr;

    explicit GenerationInput(SizeType const endId, SizeType const padId, TensorPtr ids, TensorPtr lengths, bool packed = false)
        : endId{endId}, padId{padId}, ids{std::move(ids)}, lengths{std::move(lengths)}, packed{packed} {
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor");
        TLLM_CHECK_WITH_INFO(static_cast<bool>(this->lengths), "Invalid lengths tensor");
    }

    // mandatory parameters
    SizeType endId;
    SizeType padId;
    TensorPtr ids;      // [1, packedLength] or [batchSize, maxInputLength], on gpu
    TensorPtr lengths;  // [batchSize], on gpu
    bool packed;        // indicates if ids are packed or padded to maxInputLength

    // optional parameters
    TensorPtr embeddingBiasOpt;            // [vocabSizePadded], on gpu
    TensorPtr badWordsList;                // [2, badWordsLength] or [batchSize, 2, badWordsLength], on gpu
    TensorPtr stopWordsList;               // [batchSize, 2, stopWordsLength], on gpu
    std::optional<SizeType> maxNewTokens;  // max number of tokens to generate
};

}  // namespace tensorrt_llm::runtime
