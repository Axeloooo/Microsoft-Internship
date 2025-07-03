/**
 * @file generationOutput.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief GenerationOutput class for managing outputs of the generation process in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <functional>
#include <utility>

#include "common/cudaUtils.h"
#include "runtime/common.h"
#include "runtime/iTensor.h"

namespace tensorrt_llm::runtime {

class GenerationOutput {
   public:
    using TensorPtr = ITensor::SharedPtr;

    using Callback = std::function<void(TensorPtr const& ids, SizeType step, bool finished)>;

    explicit GenerationOutput(TensorPtr ids) : ids{std::move(ids)} { TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor"); }

    // mandatory parameters
    TensorPtr ids;  // [batchSize, beamWidth, maxInputLength + maxNewTokens]

    // optional parameters
    TensorPtr logProbs;       // [request_output_length, batch_size * beam_width], must be float*, on gpu
    TensorPtr contextLogits;  // [batch_size, max_input_length, vocab_size_padded]

    // callbacks
    Callback onTokenGenerated;
};

}  // namespace tensorrt_llm::runtime
