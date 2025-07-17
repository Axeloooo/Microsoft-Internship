/**
 * @file quantization.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Header file for quantization utilities.
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "common/quantization.h"

namespace trt_allreduce {

enum class FP4QuantizationSFLayout {
    // Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
    // blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8 values.
    // The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and columns to
    // the nearest multiple of 4.
    //
    // The scale factor block rows map to data block rows in an interleaved pattern:
    // For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
    // Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
    //
    // Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
    SWIZZLED,
    // Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen kernels standard.
    LINEAR
};

#define PadUpFn(X, Y) ((X + Y - 1) / (Y) * (Y))

// totalCloumn should be in SFMatrix, not activation Matrix, so no sfVecSize needed.
inline int computeFP4SwizzledLayoutSFSize(int totalRow, int totalColumn) {
    int paddedRow = PadUpFn(totalRow, 128);
    int paddedColumn = PadUpFn(totalColumn, 4);
    return paddedRow * paddedColumn;
}

inline int computeFP4LinearLayoutSFSize(int totalRow, int totalColumn) { return totalRow * totalColumn; }

namespace kernels {

template <typename T>
void invokeQuantization(int8_t* dst, T const* src, int64_t const size, float const* scalePtr, cudaStream_t stream = 0, int maxGirdSize = 0);

template <typename T, typename QuantT>
void invokePerTokenQuantization(QuantT* dst, T const* src, int64_t const numRows, int64_t const numCols, float const* clampPtr, float* scalePtr, float* sumPtr,
                                trt_allreduce::common::QuantMode quantMode, cudaStream_t stream = 0);

template <typename T>
void invokeFP4Quantization(int m, int n, T const* input, float const* globalScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, FP4QuantizationSFLayout layout,
                           int multiProcessorCount, cudaStream_t stream = 0);

template <typename T>
void invokeBatchedFP4Quantization(int b, int m, int n, T const* input, float const* globalScale, int64_t* output, int32_t* SFOuput, bool useUE8M0, int multiProcessorCount,
                                  cudaStream_t stream = 0);

void invokeNVFP4BlockScaleInterleave(int b, int m, int n, uint8_t const* SFIn, uint8_t* SFOutput, int multiProcessorCount, cudaStream_t stream = 0);

}  // namespace kernels
}  // namespace trt_allreduce
