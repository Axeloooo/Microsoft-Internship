/**
 * @file gptJsonConfig.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Configuration class for GPT models in JSON format.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <filesystem>
#include <istream>
#include <string>
#include <utility>

#include "common/quantization.h"
#include "runtime/common.h"
#include "runtime/gptModelConfig.h"
#include "runtime/worldConfig.h"

namespace tensorrt_llm::runtime {

class GptJsonConfig {
   public:
    GptJsonConfig(std::string name, std::string precision, SizeType tensorParallelism, SizeType pipelineParallelism, GptModelConfig const& modelConfig)
        : mName(std::move(name)),
          mPrecision(std::move(precision)),
          mTensorParallelism{tensorParallelism},
          mPipelineParallelism{pipelineParallelism},
          mGptModelConfig(modelConfig) {}

    static GptJsonConfig parse(std::string const& json);

    static GptJsonConfig parse(std::istream& json);

    static GptJsonConfig parse(std::filesystem::path const& path);

    [[nodiscard]] GptModelConfig getModelConfig() const { return mGptModelConfig; }

    [[nodiscard]] std::string const& getName() const { return mName; }

    [[nodiscard]] std::string const& getPrecision() const { return mPrecision; }

    [[nodiscard]] SizeType constexpr getTensorParallelism() const { return mTensorParallelism; }

    [[nodiscard]] SizeType constexpr getPipelineParallelism() const { return mPipelineParallelism; }

    [[nodiscard]] SizeType constexpr getWorldSize() const { return mTensorParallelism * mPipelineParallelism; }

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig, std::string const& model) const;

    [[nodiscard]] std::string engineFilename(WorldConfig const& worldConfig) const { return engineFilename(worldConfig, getName()); }

   private:
    std::string const mName;
    std::string const mPrecision;
    SizeType const mTensorParallelism;
    SizeType const mPipelineParallelism;
    GptModelConfig const mGptModelConfig;
};

}  // namespace tensorrt_llm::runtime
