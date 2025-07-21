/**
 * @file envUtils.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief Utility functions for environment variable handling
 * @version 0.1
 * @date 2025-07-17
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once
#include <cstdint>
#include <optional>
#include <string>

namespace trt_allreduce::common {
// Useful when you want to inject some debug code controllable with env var.
std::optional<int32_t> getIntEnv(char const* name);

std::optional<size_t> getUInt64Env(char const* name);

// XQA kernels (optimized kernels for generation phase).
bool forceXQAKernels();

// Whether XQA JIT is enabled.
//
// Returns the value of TRTLLM_ENABLE_XQA_JIT env var. If such env var doesn't exist, std::nullopt is returned.
std::optional<bool> getEnvEnableXQAJIT();

// 0 means to use heuristics.
std::optional<int32_t> getEnvXqaBlocksPerSequence();

// Whether use tileSizeKv64 for multiCtasKvMode of trtllm-gen kernels.
bool getEnvUseTileSizeKv64ForTrtllmGen();

// Tune the number of blocks per sequence for accuracy/performance purpose.
bool getEnvMmhaMultiblockDebug();

int getEnvMmhaBlocksPerSequence();

int getEnvMmhaKernelBlockSize();

// Whether PDL is enabled.
bool getEnvEnablePDL();

bool getEnvUseUCXKvCache();

bool getEnvUseMPIKvCache();

std::string getEnvUCXInterface();

bool getEnvDisaggLayerwise();

bool getEnvParallelCacheSend();

bool getEnvRequestKVCacheConcurrent();

bool getEnvDisableKVCacheTransferOverlap();

bool getEnvEnableReceiveKVCacheParallel();

std::string getEnvKVCacheTransferOutputPath();

bool getEnvTryZCopyForKVCacheTransfer();

// Force deterministic behavior for all kernels.
bool getEnvForceDeterministic();

// Force deterministic behavior for MoE plugin.
bool getEnvForceDeterministicMOE();

// Force deterministic behavior for attention plugin.
bool getEnvForceDeterministicAttention();

// Force deterministic behavior for all reduce plugin.
bool getEnvForceDeterministicAllReduce();

// Return the workspace size for custom all reduce kernels.
// This only works when force deterministic is enabled.
size_t getEnvAllReduceWorkspaceSize();

size_t getEnvKVCacheRecvBufferCount();

bool getEnvKVCacheTransferUseAsyncBuffer();

size_t getEnvKVCacheSendMaxConcurrenceNum();

size_t getEnvMemSizeForKVCacheTransferBuffer();

}  // namespace trt_allreduce::common