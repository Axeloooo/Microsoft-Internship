/**
 * @file quantTypeUtils.cuh
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief  Utility functions for quantization types
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>

#include "common/cudaBf16Fallbacks.cuh"
#include "common/cudaFp8Utils.h"

namespace trt_allreduce {
namespace common {

template <typename T>
struct QuantTypeStaticVals;

template <>
struct QuantTypeStaticVals<int8_t> {
    static constexpr float MAX_VAL = 127.f;
    static constexpr float MIN_SCALING_FACTOR = 0.f;
    static constexpr float MIN_SCALING_FACTOR_RCP = FLT_MAX;
};

#ifdef ENABLE_FP8

template <>
struct QuantTypeStaticVals<__nv_fp8_e4m3> {
    static constexpr float MAX_VAL = 448.f;
    // Ref: https://github.com/pytorch/FBGEMM/blob/main/fbgemm_gpu/experimental/gen_ai/src/quantize/quantize.cu#L720
    static constexpr float MIN_SCALING_FACTOR = 1.0f / (448.f * 512.f);
    static constexpr float MIN_SCALING_FACTOR_RCP = (448.f * 512.f);
};

#endif  // ENABLE_FP8

}  // namespace common
}  // namespace trt_allreduce