/**
 * @file decodingOutput.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief DecodingOutput class for managing outputs of the decoding process in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <utility>

#include "common/cudaUtils.h"
#include "runtime/bufferManager.h"
#include "runtime/common.h"
#include "runtime/iTensor.h"

namespace tensorrt_llm::runtime {
class DecodingOutput {
   public:
    using TensorPtr = ITensor::SharedPtr;

    class BeamHypotheses {
       public:
        TensorPtr outputIdsTgt;        // [batchSize, 2 * beamWidth, maxSeqLen]
        TensorPtr sequenceLengthsTgt;  // [batchSize, 2 * beamWidth]
        TensorPtr cumLogProbs;         // [batchSize, 2 * beamWidth]
        TensorPtr normedScores;        // [batchSize, 2 * beamWidth]
        TensorPtr logProbs;            // [batchSize, 2 * beamWidth, maxSeqLen]
        TensorPtr minNormedScores;     // [batchSize]
        TensorPtr numBeams;            // [batchSize]
        TensorPtr isDone;              // [batchSize]

        void empty(BufferManager& manager);

        void reshape(SizeType batchSize, SizeType beamWidth, SizeType maxSequenceLength);

        void release();

        void init(BufferManager& manager, TokenIdType endId);

        BeamHypotheses slice(SizeType batchIndex, SizeType size) const;
    };

    static float constexpr kNegativeInfinity = -1e20f;

    explicit DecodingOutput(TensorPtr ids) : ids{std::move(ids)} { TLLM_CHECK_WITH_INFO(static_cast<bool>(this->ids), "Invalid ids tensor"); }

    // mandatory parameters
    TensorPtr ids;        // [batchSize, beamWidth, maxSeqLen], on gpu, must contain previously generated token ids for all
                          // steps before DecodingInput.step
    TensorPtr newTokens;  // [batchSize, beamWidth] on gpu.

    // optional parameters
    TensorPtr finished;     // [batchSize, beamWidth], mandatory in beam search and to determine whether to stop
                            // according to DecodingInput.sequenceLimitLength, on gpu
    TensorPtr finishedSum;  // [1], the sum of finished sequences, in pinned memory
    TensorPtr logProbs;     // [maxNewTokens, batchSize, beamWidth], must be float*, on gpu

    // mandatory parameters for beam search
    TensorPtr cumLogProbs;       // [batchSize, beamWidth], optional for sampling, on gpu
    TensorPtr parentIds;         // [batchSize, beamWidth, maxSeqLen], on gpu
    TensorPtr lengths;           // [batchSize, beamWidth], total sequence lengths including padding, on gpu
    TensorPtr cacheIndirection;  // [batchSize, beamWidth, maxSeqLen], k/v indirection for next generation step, on gpu

    BeamHypotheses beamHypotheses;
};

}  // namespace tensorrt_llm::runtime
