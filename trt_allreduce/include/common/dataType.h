/**
 * @file dataType.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for data types
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <NvInferRuntime.h>

#include <map>

#include "common/logger.h"

namespace trt_allreduce::common {

constexpr static size_t getDTypeSize(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kINT32:
            [[fallthrough]];
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kBF16:
            [[fallthrough]];
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kBOOL:
            [[fallthrough]];
        case nvinfer1::DataType::kUINT8:
            [[fallthrough]];
        case nvinfer1::DataType::kINT8:
            [[fallthrough]];
        case nvinfer1::DataType::kFP8:
            return 1;
        case nvinfer1::DataType::kINT4:
            return 1;  // INT4 is 4 bits, but return 1 byte for simplicity
        // case nvinfer1::DataType::kFP4:  // TODO: kFP4 not available in current TensorRT version
        //     TLLM_THROW("Cannot determine size of FP4 data type");
        default:
            TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    }
    return 0;
}

constexpr static size_t getDTypeSizeInBits(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kINT64:
            return 64;
        case nvinfer1::DataType::kINT32:
            [[fallthrough]];
        case nvinfer1::DataType::kFLOAT:
            return 32;
        case nvinfer1::DataType::kBF16:
            [[fallthrough]];
        case nvinfer1::DataType::kHALF:
            return 16;
        case nvinfer1::DataType::kBOOL:
            [[fallthrough]];
        case nvinfer1::DataType::kUINT8:
            [[fallthrough]];
        case nvinfer1::DataType::kINT8:
            [[fallthrough]];
        case nvinfer1::DataType::kFP8:
            return 8;
        case nvinfer1::DataType::kINT4:
            return 4;
        // case nvinfer1::DataType::kFP4:  // TODO: kFP4 not available in current TensorRT version
        //     return 4;
        default:
            TLLM_THROW("Unknown dtype %d", static_cast<int>(type));
    }
    return 0;
}

[[maybe_unused]] static std::string getDtypeString(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT:
            return "fp32";
            break;
        case nvinfer1::DataType::kHALF:
            return "fp16";
            break;
        case nvinfer1::DataType::kINT8:
            return "int8";
            break;
        case nvinfer1::DataType::kINT32:
            return "int32";
            break;
        case nvinfer1::DataType::kBOOL:
            return "bool";
            break;
        case nvinfer1::DataType::kUINT8:
            return "uint8";
            break;
        case nvinfer1::DataType::kFP8:
            return "fp8";
            break;
        case nvinfer1::DataType::kBF16:
            return "bf16";
            break;
        case nvinfer1::DataType::kINT64:
            return "int64";
            break;
        case nvinfer1::DataType::kINT4:
            return "int4";
            break;
        // case nvinfer1::DataType::kFP4:  // TODO: kFP4 not available in current TensorRT version
        //     return "fp4";
        //     break;
        default:
            throw std::runtime_error("Unsupported data type");
            break;
    }

    return "";
}

}  // namespace trt_allreduce::common
