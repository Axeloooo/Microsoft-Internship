/**
 * @file gptModelConfig.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Configuration class for GPT models in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <NvInferRuntime.h>

#include "common/quantization.h"
#include "runtime/common.h"

namespace tensorrt_llm::runtime {

class GptModelConfig {
   public:
    enum class ModelVariant : std::int32_t {
        kGpt = 0,
        kGlm = 1,  // https://github.com/THUDM/GLM
    };

    constexpr explicit GptModelConfig(SizeType vocabSize, SizeType nbLayers, SizeType nbHeads, SizeType hiddenSize, nvinfer1::DataType dtype)
        : mVocabSize(vocabSize),
          mNbLayers(nbLayers),
          mNbHeads(nbHeads),
          mNbKvHeads(nbHeads),
          mHiddenSize(hiddenSize),
          mDataType(dtype),
          mUseGptAttentionPlugin(false),
          mInputPacked{false},
          mPagedKvCache{false},
          mTokensPerBlock{64},
          mQuantMode{common::QuantMode::none()},
          mMaxBatchSize(0),
          mMaxInputLen(0),
          mMaxOutputLen(0),
          mMaxNumTokens(std::nullopt),
          mComputeContextLogits(false),
          mModelVariant(ModelVariant::kGpt),
          mUseCustomAllReduce(false) {}

    [[nodiscard]] SizeType constexpr getVocabSize() const noexcept { return mVocabSize; }

    [[nodiscard]] SizeType constexpr getVocabSizePadded(SizeType worldSize) const noexcept { return (mVocabSize + worldSize - 1) / worldSize * worldSize; }

    [[nodiscard]] SizeType constexpr getNbLayers(SizeType pipelineParallelism = 1) const {
        TLLM_CHECK(mNbLayers % pipelineParallelism == 0);
        return mNbLayers / pipelineParallelism;
    }

    [[nodiscard]] SizeType constexpr getNbHeads() const noexcept { return mNbHeads; }

    [[nodiscard]] SizeType constexpr getNbKvHeads() const noexcept { return mNbKvHeads; }

    void constexpr setNbKvHeads(SizeType nbKvHeads) noexcept { mNbKvHeads = nbKvHeads; }

    [[nodiscard]] SizeType constexpr getHiddenSize() const noexcept { return mHiddenSize; }

    [[nodiscard]] SizeType constexpr getSizePerHead() const noexcept { return mHiddenSize / mNbHeads; }

    [[nodiscard]] nvinfer1::DataType constexpr getDataType() const noexcept { return mDataType; }

    [[nodiscard]] bool constexpr useGptAttentionPlugin() const noexcept { return mUseGptAttentionPlugin; }

    void constexpr useGptAttentionPlugin(bool useGptAttentionPlugin) noexcept { mUseGptAttentionPlugin = useGptAttentionPlugin; }

    [[nodiscard]] bool constexpr usePackedInput() const noexcept { return mInputPacked; }

    void constexpr usePackedInput(bool inputPacked) noexcept { mInputPacked = inputPacked; }

    [[nodiscard]] bool constexpr usePagedKvCache() const noexcept { return mPagedKvCache; }

    void constexpr usePagedKvCache(bool pagedKvCache) noexcept { mPagedKvCache = pagedKvCache; }

    [[nodiscard]] SizeType constexpr getTokensPerBlock() const noexcept { return mTokensPerBlock; }

    void constexpr setTokensPerBlock(SizeType TokensPerBlock) noexcept { mTokensPerBlock = TokensPerBlock; }

    [[nodiscard]] common::QuantMode constexpr getQuantMode() const noexcept { return mQuantMode; }

    void constexpr setQuantMode(common::QuantMode QuantMode) noexcept { mQuantMode = QuantMode; }

    [[nodiscard]] bool constexpr supportsInflightBatching() const noexcept { return mUseGptAttentionPlugin && mInputPacked && mPagedKvCache; }

    [[nodiscard]] SizeType constexpr getMaxBatchSize() const noexcept { return mMaxBatchSize; }

    void constexpr setMaxBatchSize(SizeType maxBatchSize) noexcept { mMaxBatchSize = maxBatchSize; }

    [[nodiscard]] SizeType constexpr getMaxInputLen() const noexcept { return mMaxInputLen; }

    void constexpr setMaxInputLen(SizeType maxInputLen) noexcept { mMaxInputLen = maxInputLen; }

    [[nodiscard]] SizeType constexpr getMaxOutputLen() const noexcept { return mMaxOutputLen; }

    void constexpr setMaxOutputLen(SizeType maxOutputLen) noexcept { mMaxOutputLen = maxOutputLen; }

    [[nodiscard]] std::optional<SizeType> constexpr getMaxNumTokens() const noexcept { return mMaxNumTokens; }

    void constexpr setMaxNumTokens(std::optional<SizeType> maxNumTokens) noexcept { mMaxNumTokens = maxNumTokens; }

    [[nodiscard]] bool constexpr computeContextLogits() const noexcept { return mComputeContextLogits; }

    void constexpr computeContextLogits(bool computeContextLogits) noexcept { mComputeContextLogits = computeContextLogits; }

    [[nodiscard]] ModelVariant getModelVariant() const { return mModelVariant; }

    void setModelVariant(ModelVariant modelVariant) { mModelVariant = modelVariant; }

    [[nodiscard]] bool constexpr useCustomAllReduce() const noexcept { return mUseCustomAllReduce; }

    void constexpr useCustomAllReduce(bool customAllReduce) noexcept { mUseCustomAllReduce = customAllReduce; }

   private:
    SizeType mVocabSize;
    SizeType mNbLayers;
    SizeType mNbHeads;
    SizeType mNbKvHeads;
    SizeType mHiddenSize;
    nvinfer1::DataType mDataType;
    bool mUseGptAttentionPlugin;
    bool mInputPacked;
    bool mPagedKvCache;
    SizeType mTokensPerBlock;
    common::QuantMode mQuantMode;
    SizeType mMaxBatchSize;
    SizeType mMaxInputLen;
    SizeType mMaxOutputLen;
    std::optional<SizeType> mMaxNumTokens;

    bool mComputeContextLogits;
    ModelVariant mModelVariant;
    bool mUseCustomAllReduce;
};

}  // namespace tensorrt_llm::runtime
