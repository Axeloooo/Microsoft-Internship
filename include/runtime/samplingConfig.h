/**
 * @file samplingConfig.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Configuration class for sampling parameters in TensorRT-LLM runtime.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <optional>
#include <vector>

#include "runtime/common.h"

namespace tensorrt_llm::runtime {

class SamplingConfig {
    using FloatType = float;

    template <typename T>
    using OptVec = std::optional<std::vector<T>>;

   public:
    explicit SamplingConfig(SizeType beamWidth = 1) : beamWidth{beamWidth} {}

    SizeType beamWidth;

    OptVec<FloatType> temperature;  // [1] or [batch_size] on cpu
    OptVec<SizeType> minLength;     // [1] or [batch_size] on cpu
    // repetitionPenalty and presencePenalty are mutually exclusive.
    OptVec<FloatType> repetitionPenalty;  // [1] or [batch_size] on cpu
    OptVec<FloatType> presencePenalty;    // [1] or [batch_size] on cpu

    // sampling layers
    OptVec<SizeType> topK;                  // [1] or [batch_size] on cpu
    OptVec<FloatType> topP;                 // [1] or [batch_size] on cpu
    OptVec<unsigned long long> randomSeed;  // [1] or [batch_size] on cpu
    OptVec<FloatType> topPDecay;            // [batch_size], must between [0, 1]
    OptVec<FloatType> topPMin;              // [batch_size], must between [0, 1]
    OptVec<SizeType> topPResetIds;          // [batch_size]

    // beam search layer
    std::optional<FloatType> beamSearchDiversityRate;
    std::optional<FloatType> lengthPenalty;
};

}  // namespace tensorrt_llm::runtime
