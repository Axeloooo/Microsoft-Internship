/**
 * @file tllmLogger.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Logger class for TensorRT-LLM runtime.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <NvInferRuntime.h>

namespace tensorrt_llm::runtime {

class TllmLogger : public nvinfer1::ILogger {
   public:
    void log(Severity severity, nvinfer1::AsciiChar const* msg) noexcept override;

    Severity getLevel();

    void setLevel(Severity level);
};

}  // namespace tensorrt_llm::runtime
