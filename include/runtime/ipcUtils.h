/**
 * @file ipcUtils.h
 * @author Axel Omar Sanchez Peralta (t-axels@microsoft.com)
 * @brief IPC utilities for managing inter-process communication memory in TensorRT-LLM.
 * @version 0.1
 * @date 2025-07-03
 *
 * @copyright Copyright (c) 2025
 *
 */

#pragma once

#include "kernels/customAllReduceKernels.h"
#include "runtime/bufferManager.h"
#include "runtime/iTensor.h"
#include "runtime/worldConfig.h"

namespace tensorrt_llm::runtime {

void setPeerAccess(WorldConfig worldConfig, bool enable = true);

class IpcMemory {
   public:
    using TensorPtr = ITensor::SharedPtr;

    size_t static constexpr FLAGS_SIZE = kernels::MAX_ALL_REDUCE_BLOCKS * sizeof(uint32_t);

    IpcMemory(WorldConfig worldConfig, std::size_t bufferSize);
    ~IpcMemory();

    [[nodiscard]] const std::vector<void*>& getCommPtrsTensor() const { return mCommPtrs; }

   private:
    void allocateIpcMemory();
    void destroyIpcMemory();

    WorldConfig mWorldConfig;
    std::vector<void*> mCommPtrs;
    std::size_t mBufferSize;
    void* mBufferPtr;
};

}  // namespace tensorrt_llm::runtime
